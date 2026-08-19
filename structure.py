"""
Stage 1: Table Structure Learning
Detects rows, columns, and merge regions WITHOUT touching OCR.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CellRegion:
    row: int                  # logical row index (0-based, ALWAYS preserved)
    col: int                  # logical column index
    bbox: tuple                # (x0, y0, x1, y1) in image coords
    merge_flag: bool = False
    merge_id: Optional[int] = None
    ocr_text: str = ""
    has_information: bool = False

@dataclass
class MergeRegion:
    merge_id: int
    row_start: int
    row_end: int               # inclusive
    col_start: int
    col_end: int                # inclusive
    bbox: tuple                 # union bbox (x0, y0, x1, y1)


def cluster_1d(coords, gap=5):
    if len(coords) == 0:
        return []
    coords = sorted(coords)
    clusters = []
    current = [coords[0]]
    for c in coords[1:]:
        if c - current[-1] <= gap:
            current.append(c)
        else:
            clusters.append(int(np.mean(current)))
            current = [c]
    clusters.append(int(np.mean(current)))
    return clusters


class TableStructure:
    """
    Learns row boundaries, column boundaries, and per-column merge
    regions from a table image. Contains NO OCR logic.
    """

    def __init__(self, gray_img, line_presence_thresh=0.5, row_frac_thresh=0.3,
                 col_frac_thresh=0.5, header_position="bottom"):
        """
        header_position: which row band of the grid holds the column
        header text -- "bottom" (default) or "top". This is a property of
        the document layout (many BOM sheets build the table upward from
        a title block, so the header row ends up at the bottom of the
        image, i.e. the table reads bottom-up), not something inferred
        from content. No OCR happens here; this only records which row
        index later stages should treat as the header.
        """
        self.gray = gray_img
        self.h, self.w = gray_img.shape
        self.line_presence_thresh = line_presence_thresh
        self.row_frac_thresh = row_frac_thresh
        self.col_frac_thresh = col_frac_thresh
        self.header_position = header_position

        # Pixel tolerances below (cluster_1d's line-coordinate grouping gap,
        # and the +/-N px window _line_exists_at/_vline_exists_at search
        # around an expected wall position) were originally hardcoded
        # constants tuned against ~300 DPI crops (2243px-wide reference
        # table). At a meaningfully higher render DPI, every gridline is
        # proportionally more pixels wide/further apart, so a fixed pixel
        # tolerance becomes too tight -- real walls fall outside the search
        # window, read as "missing", and get unioned into false merges.
        # Scaling both by the image's own width relative to that reference
        # keeps wall/grid-line detection resolution-independent instead of
        # silently degrading whenever dpi changes.
        _REFERENCE_W = 2243
        self._px_scale = self.w / _REFERENCE_W
        self._cluster_gap = max(5, round(5 * self._px_scale))
        self._boundary_tol = max(2, round(2 * self._px_scale))
        # Adaptive-threshold binarization also uses a fixed local
        # neighborhood (blockSize) to decide dark-vs-light per pixel. At
        # higher DPI, gridlines/strokes are proportionally thicker, so a
        # too-small fixed blockSize can end up sampling only a line's own
        # interior, making it locally indistinguishable from background --
        # lines drop out or turn spotty, and wall detection misreads them
        # as missing, causing false merges. Scale it the same way; must
        # stay odd and >=3 for cv2.adaptiveThreshold.
        block_size = max(3, round(15 * self._px_scale))
        if block_size % 2 == 0:
            block_size += 1

        self.bin_img = cv2.adaptiveThreshold(
            ~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, -2
        )
        self._detect_horizontal_lines()
        self.Y = self._detect_rows()
        self.X = self._detect_columns_from_reference_row()
        self.n_rows = len(self.Y) - 1
        self.n_cols = len(self.X) - 1
        self.header_row_idx = (self.n_rows - 1) if header_position == "bottom" else 0

        self.cells: dict[tuple[int, int], CellRegion] = {}
        self.merge_regions: list[MergeRegion] = []
        self._build_grid()
        self._detect_merges()

    def _detect_horizontal_lines(self):
        horiz_size = max(5, self.w // 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
        lines = cv2.erode(self.bin_img, kernel)
        lines = cv2.dilate(lines, kernel)
        self.horiz_lines = lines

    def _detect_rows(self):
        row_sums = np.sum(self.horiz_lines, axis=1)
        y_rows = np.where(row_sums > (self.w * self.row_frac_thresh * 255))[0]
        Y = cluster_1d(list(y_rows), gap=self._cluster_gap)
        if len(Y) < 2:
            raise ValueError("Could not detect enough row lines to form a table.")
        return Y

    def _detect_columns_from_reference_row(self):
        """
        Column boundaries are calibrated ONCE from whichever row band
        (header or footer) is most fully gridded -- i.e. has the most
        detected vertical segments. This avoids relying on fragile
        per-row vertical lines that break under merged cells.
        """
        # Try multiple candidate reference bands (footer, header) and score
        # each by how EVENLY its detected boundaries are spaced -- a real
        # grid has fairly regular column widths, whereas noise-driven false
        # positives (e.g. OCR strokes mistaken for verticals in a
        # non-gridded text row) produce uneven, inflated boundary counts.
        # This replaces "pick whichever found the most lines", which was
        # picking up noise as if it were extra real columns.
        candidates = []
        bands = [
            (self.Y[-2], self.Y[-1]),  # footer row
            (self.Y[0], self.Y[1]),    # header/top row
        ]
        if len(self.Y) > 3:
            # a banner row at either edge (e.g. a full-width "PART
            # NO.-52164..." heading with no internal column dividers)
            # will score badly here and get skipped by regularity_score
            # below anyway -- these extra bands just mean the real
            # header/footer row one step in doesn't have to rely purely
            # on the OTHER edge's band happening to be well-formed
            bands.append((self.Y[1], self.Y[2]))     # second row from top
            bands.append((self.Y[-3], self.Y[-2]))   # second row from bottom
        for y0, y1 in bands:
            band = self.bin_img[y0:y1, :]
            vsize = max(5, (y1 - y0) // 2)
            vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vsize))
            vlines = cv2.dilate(cv2.erode(band, vkernel), vkernel)
            col_sums = np.sum(vlines, axis=0)
            xs = np.where(col_sums > ((y1 - y0) * self.col_frac_thresh * 255))[0]
            X = cluster_1d(list(xs), gap=self._cluster_gap)
            candidates.append(X)

        def regularity_score(X):
            if len(X) < 3:
                return -1  # too few boundaries to trust
            widths = np.diff(X)
            cv_score = np.std(widths) / (np.mean(widths) + 1e-6)
            return -cv_score  # higher score is better (lower variation)

        scored = [(regularity_score(X), X) for X in candidates]
        scored = [s for s in scored if s[0] > -1]
        if not scored:
            raise ValueError("Could not detect column boundaries from header/footer rows.")
        best = max(scored, key=lambda s: s[0])[1]
        return best

    def _build_grid(self):
        """Every logical (row, col) gets a CellRegion. Row count is NEVER reduced here."""
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                bbox = (self.X[c], self.Y[r], self.X[c + 1], self.Y[r + 1])
                self.cells[(r, c)] = CellRegion(row=r, col=c, bbox=bbox)

    def _line_exists_at(self, y_idx, col_i):
        """Is there a real horizontal rule at row-boundary y_idx, spanning column col_i?"""
        y = self.Y[y_idx]
        x0, x1 = self.X[col_i], self.X[col_i + 1]
        t = self._boundary_tol
        band = self.horiz_lines[max(0, y - t):y + t + 1, x0:x1]
        if band.size == 0:
            return False
        return (band.max(axis=0) > 0).mean() > self.line_presence_thresh

    def _vline_exists_at(self, x_idx, row_i):
        """Is there a real vertical rule at column-boundary x_idx, spanning row_i?"""
        x = self.X[x_idx]
        y0, y1 = self.Y[row_i], self.Y[row_i + 1]
        t = self._boundary_tol
        band = self.vert_lines[y0:y1, max(0, x - t):x + t + 1]
        if band.size == 0:
            return False
        return (band.max(axis=1) > 0).mean() > self.line_presence_thresh

    def _detect_vertical_lines_full(self):
        """Full-page vertical line mask, used for detecting missing walls
        BETWEEN columns within a single row (horizontal merges)."""
        vert_size = max(5, self.h // 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))
        lines = cv2.erode(self.bin_img, kernel)
        lines = cv2.dilate(lines, kernel)
        self.vert_lines = lines

    def _detect_merges(self):
        """
        General 2D block merge detection. A merge is NOT limited to "same
        column, multiple rows" -- it can also span multiple columns within
        one row (e.g. a "REFER DRAWING NO..." banner with no internal
        column rules), or both at once. We detect this by building two
        "wall" grids (is there a real rule to the right of cell (r,c)? is
        there a real rule below cell (r,c)?) and then union-find cells
        together wherever the wall between them is missing. Each resulting
        connected component is one rectangular merge block.
        """
        self._detect_vertical_lines_full()

        n_r, n_c = self.n_rows, self.n_cols
        # right_wall[r][c] = True if a real vertical rule separates (r,c) from (r,c+1)
        right_wall = [[True] * (n_c - 1) for _ in range(n_r)]
        for r in range(n_r):
            for c in range(n_c - 1):
                right_wall[r][c] = self._vline_exists_at(c + 1, r)

        # bottom_wall[r][c] = True if a real horizontal rule separates (r,c) from (r+1,c)
        bottom_wall = [[True] * n_c for _ in range(n_r - 1)]
        for r in range(n_r - 1):
            for c in range(n_c):
                bottom_wall[r][c] = self._line_exists_at(r + 1, c)

        # Union-Find over the grid
        parent = {}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for r in range(n_r):
            for c in range(n_c):
                parent[(r, c)] = (r, c)

        for r in range(n_r):
            for c in range(n_c - 1):
                if not right_wall[r][c]:
                    union((r, c), (r, c + 1))
        for r in range(n_r - 1):
            for c in range(n_c):
                if not bottom_wall[r][c]:
                    union((r, c), (r + 1, c))

        # group cells by connected component
        groups = {}
        for r in range(n_r):
            for c in range(n_c):
                root = find((r, c))
                groups.setdefault(root, []).append((r, c))

        merge_id_counter = 0
        for root, members in groups.items():
            rows = [m[0] for m in members]
            cols = [m[1] for m in members]
            row_start, row_end = min(rows), max(rows)
            col_start, col_end = min(cols), max(cols)
            is_merge = len(members) > 1
            if is_merge:
                x0, y0 = self.X[col_start], self.Y[row_start]
                x1, y1 = self.X[col_end + 1], self.Y[row_end + 1]
                region = MergeRegion(merge_id=merge_id_counter, row_start=row_start,
                                      row_end=row_end, col_start=col_start,
                                      col_end=col_end, bbox=(x0, y0, x1, y1))
                self.merge_regions.append(region)
            for (r, c) in members:
                cell = self.cells[(r, c)]
                cell.merge_flag = is_merge
                cell.merge_id = merge_id_counter if is_merge else None
            if is_merge:
                merge_id_counter += 1

    def summary(self):
        merged = [m for m in self.merge_regions]
        return (f"{self.n_rows} rows x {self.n_cols} cols | "
                f"{len(merged)} merge regions (rectangular blocks)")