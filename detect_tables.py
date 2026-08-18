"""
Stage 0: Multi-table detection on a PDF page.

Finds every table region on a page, then re-rasterizes ONLY that region
straight from the PDF at a high target DPI (crop-before-render, not
render-then-crop) so the downstream pipeline (structure.py etc.) never
has to work with the tiny row heights that were root-caused as the
actual failure mode on image.png (see project notes: 10px rows -> OCR
glyph loss; result.png at ~2x resolution fixed it with zero pipeline
code changes).

Three detection strategies, all run per page and merged:

  1. VECTOR (preferred, used whenever the page has drawn line/rect
     primitives). Reads the PDF's own vector geometry directly via
     PyMuPDF -- no thresholding, no pixel ambiguity, exact coordinates
     in PDF point-space. Table regions are found by unioning line
     segments whose bounding boxes touch/overlap (a real grid's
     horizontal and vertical rules all interconnect into one component;
     a stray title-block border does not reach that density) and
     keeping only components with enough distinct row/column lines to
     plausibly be a table, not just a box. A connected group is then
     split on genuine whitespace gaps (same adaptive-density idea used
     in raster mode) before being accepted as one table region, since a
     shared border/dimension line can bridge a real table into an
     unrelated diagram panel within the same connected component.

  2. RASTER GRID (used for scanned/embedded-image tables whose rules
     are still individually resolvable at preview DPI). Renders the
     page at a modest preview DPI and reuses the same morphological
     horizontal/vertical line-detection approach as structure.py,
     applied page-wide instead of table-wide, then finds connected
     components in the combined line mask.

  3. RASTER CORNER/BLOB (used for a scanned table that's genuinely
     detailed but crammed into a small physical area -- individual
     rules blur together at preview DPI and undercount below strategy
     2's line-count threshold, losing the table). Never tries to
     resolve individual rules; only finds the table's OUTER boundary.
     A densely-ruled table still shows up as one solid dark blob even
     when its rules can't be told apart, so this closes the ink mask
     and fits cv2.minAreaRect to the largest blob -- which by
     construction is always a true rectangle (adjacent sides at 90
     degrees, opposite sides equal length and parallel) -- giving 4
     corner points directly. If those corners indicate a rotated/skewed
     table, the high-DPI crop is perspective-warped straight using the
     corners before being saved, so structure.py (which has no
     rotation-handling) still sees a level grid. Internal row/column
     resolution is left entirely to structure.py, which runs later
     against the high-DPI crop -- exactly where that resolution is
     actually available.

Every detected region is also run through a lightweight CONTENT-TYPE
CLASSIFIER before being handed off. A rectangular grid with big cells
full of diagrams and captions (e.g. a 2x6 panel of illustrated assembly
steps) satisfies every geometric test above just as validly as a real
BOM table does -- right angles, equal/parallel sides, real ruling
lines -- so detection alone can't and shouldn't try to exclude it. But
sending it into the same OCR/cell-value extraction pipeline as a data
table would be meaningless: there's no "value" to read out of a
diagram. The classifier flags likely_diagram_grid vs likely_data_table
using two generic, content-shape signals (no domain wordlists, same
principle postprocess.py already follows) -- curve/embedded-image
density (diagrams are drawn with bezier curves and raster images; BOM
cells are just text and straight rules) and average cell size (a data
row is usually close to one text line tall; an illustration cell is
much bigger, it has to fit a whole diagram). This is a QA flag, not a
hard filter -- ambiguous cases are labeled "uncertain" rather than
guessed, same spirit as export_xlsx.py's existing length-outlier
flagging.

For every detected region, the ORIGINAL page (not the preview raster)
is re-rendered at `dpi` clipped to just that region's bbox -- this is
the "regenerate the image using dpi, pass only the required region"
step. Output is one crop per table plus a JSON manifest and an
annotated overview image per page, so a reviewer holding the actual
(confidential) PDF can sanity-check detection by eye without needing to
send any file back -- the overview/manifest only contain geometry, no
cell content.

Usage:
    python3 detect_tables.py <pdf_path> [output_dir]
        [--dpi=300] [--preview-dpi=150] [--min-lines=3] [--padding=6]

Output layout:
    <output_dir>/
        tables/page{P}_table{I}.png   -- one high-DPI crop per table
        overview_page{P}.png          -- preview page w/ boxes drawn, for review
        tables.json                   -- manifest (page, bbox, method, filename)
"""
import sys
import os
import json
import cv2
import numpy as np
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Vector-based detection
# ---------------------------------------------------------------------------

def _extract_line_segments(page):
    """
    Pull every straight line segment out of the page's vector drawings,
    from both explicit line items and rectangle edges (a rectangle is 4
    lines for this purpose). Curves/beziers are ignored -- table rules
    are straight lines, and treating a curve as a rule would be a false
    signal.

    Returns two lists of segments in PDF point-space:
      horiz: list of (y, x0, x1)
      vert:  list of (x, y0, y1)
    Segments below `min_len` points are dropped as noise (stray tick
    marks, dimension arrows, font-rendering artifacts).
    """
    min_len = 3.0
    horiz, vert = [], []

    def _classify(x0, y0, x1, y1):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if dx >= dy and dx >= min_len:
            horiz.append((round((y0 + y1) / 2, 1), min(x0, x1), max(x0, x1)))
        elif dy > dx and dy >= min_len:
            vert.append((round((x0 + x1) / 2, 1), min(y0, y1), max(y0, y1)))

    for d in page.get_drawings():
        for item in d.get("items", []):
            kind = item[0]
            if kind == "l":  # straight line: ("l", p1, p2)
                p1, p2 = item[1], item[2]
                _classify(p1.x, p1.y, p2.x, p2.y)
            elif kind == "re":  # rectangle: ("re", Rect)
                r = item[1]
                _classify(r.x0, r.y0, r.x1, r.y0)  # top
                _classify(r.x0, r.y1, r.x1, r.y1)  # bottom
                _classify(r.x0, r.y0, r.x0, r.y1)  # left
                _classify(r.x1, r.y0, r.x1, r.y1)  # right
            elif kind == "qu":  # quad, treat edges same as rect
                q = item[1]
                pts = [q.ul, q.ur, q.lr, q.ll]
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    _classify(a.x, a.y, b.x, b.y)
            # "c" (bezier curve) intentionally ignored -- not a table rule

    return horiz, vert


def _strip_thick_ink(bin_img, max_thickness=6):
    """Deprecated in favor of _remove_thick_line_components, kept only
    for reference -- an absolute raw-ink thickness cutoff doesn't
    generalize: a real scanned drawing's stroke thickness (ink bleed,
    photocopy artifacts) varies enough by source quality that a fixed
    pixel threshold either strips legitimate content on a heavier scan
    or misses the blob on a lighter one (confirmed empirically: this
    approach removed ~90% of ink, including real rules/text, on an
    actual scanned drawing). See _remove_thick_line_components for the
    fix that's actually used."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max_thickness, max_thickness))
    thick_ink = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(bin_img, thick_ink)


def _remove_thick_line_components(line_mask, axis, max_thickness=15):
    """
    A redaction scribble, ink stamp, or filled marker can survive the
    same long-kernel erosion test used to find a real rule -- erosion
    only checks CONTINUITY along the kernel's length (is this dark all
    the way across?), never THICKNESS in the perpendicular direction,
    so a wide/tall solid blob passes exactly as easily as a genuine
    thin rule does. This showed up directly on a real scanned drawing:
    a redaction mark next to a real underline got picked up as a
    spurious "line" and misdetected as a table.

    Fixing this by stripping thick ink from the raw image BEFORE line
    detection doesn't generalize (see _strip_thick_ink's docstring) --
    real scan stroke thickness varies too much by source quality to
    threshold reliably in raw-pixel space. Instead, this filters the
    RESULT: after erode/dilate produces a line mask, a genuine rule's
    mask stays thin (its thickness there is just the original rule's
    printed thickness, since dilation with a 1px-tall/wide kernel
    doesn't grow the perpendicular dimension) regardless of how heavy
    the surrounding scan noise is -- but a blob's mask is exactly as
    thick as the blob itself, which is categorically larger than any
    real rule/border even under realistic scan degradation. Removes
    any connected component in the mask whose thickness (height for a
    horizontal-line mask, width for a vertical-line mask) exceeds
    max_thickness.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
    out = np.zeros_like(line_mask)
    for label in range(1, n_labels):
        x, y, bw, bh, area = stats[label]
        thickness = bh if axis == "h" else bw
        if thickness <= max_thickness:
            out[labels == label] = 255
    return out


def _cluster_positions(values, tol=2.0):
    """Cluster nearby scalar positions (e.g. all the y's of horizontal
    lines) into distinct row/column boundary lines, same idea as
    structure.py's cluster_1d but on floats."""
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _seg_bbox(kind, seg, pad):
    if kind == "h":
        y, x0, x1 = seg
        return (x0 - pad, y - pad, x1 + pad, y + pad)
    else:
        x, y0, y1 = seg
        return (x - pad, y0 - pad, x + pad, y1 + pad)


def _bbox_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _rasterize_segment_group(members, segs, bbox, px_per_pt=4.0, thickness=2):
    """
    Render one connected group of vector segments into TWO small binary
    masks, local to its own bbox (origin at bbox's top-left, scaled by
    px_per_pt) -- one containing only horizontal segments, one
    containing only vertical segments. Kept separate (not combined
    into one mask) specifically so the whitespace-gap splitter can use
    cross-axis density -- see _split_by_cross_axis_whitespace's
    docstring for why a combined mask breaks on a real table with
    heavily merged cells.

    `thickness` pads each drawn segment by a couple of pixels so two
    segments that are meant to intersect (e.g. a rule's endpoint
    meeting a perpendicular rule) still touch in the mask even though
    _extract_line_segments only stores each segment's centerline.

    Returns (h_mask, v_mask, px_per_pt, pixel_offset) -- pixel_offset
    is how much the mask's content is shifted from (0,0) to leave room
    for the thickness padding, needed when mapping mask coordinates
    back to PDF point-space.
    """
    x0, y0, x1, y1 = bbox
    w = max(1, int(round((x1 - x0) * px_per_pt)))
    h = max(1, int(round((y1 - y0) * px_per_pt)))
    off = thickness
    h_mask = np.zeros((h + 2 * off, w + 2 * off), dtype=np.uint8)
    v_mask = np.zeros((h + 2 * off, w + 2 * off), dtype=np.uint8)
    for i in members:
        kind, seg = segs[i]
        if kind == "h":
            y, sx0, sx1 = seg
            p0 = (int(round((sx0 - x0) * px_per_pt)) + off, int(round((y - y0) * px_per_pt)) + off)
            p1 = (int(round((sx1 - x0) * px_per_pt)) + off, int(round((y - y0) * px_per_pt)) + off)
            cv2.line(h_mask, p0, p1, 255, thickness)
        else:
            x, sy0, sy1 = seg
            p0 = (int(round((x - x0) * px_per_pt)) + off, int(round((sy0 - y0) * px_per_pt)) + off)
            p1 = (int(round((x - x0) * px_per_pt)) + off, int(round((sy1 - y0) * px_per_pt)) + off)
            cv2.line(v_mask, p0, p1, 255, thickness)
    return h_mask, v_mask, px_per_pt, off


def detect_tables_vector(page, min_lines=3, bridge_gap=4.0):
    """
    Group line segments into connected components (segments whose
    padded bboxes touch/overlap get unioned -- this is exactly how a
    grid's horizontal and vertical rules all end up in one component,
    since every rule touches its neighbors at an intersection). Keep
    only components with at least `min_lines` distinct horizontal AND
    `min_lines` distinct vertical positions, which filters out a lone
    title-block rectangle or a single underline.

    A connected group can still legitimately span more than one real
    table -- e.g. a BOM table and an adjacent diagram/dimensioning
    panel that happen to share a border or dimension line within
    `bridge_gap` of each other. Raster mode already handles this exact
    failure mode with an adaptive whitespace-gap split
    (_split_by_cross_axis_whitespace); vector mode needs the same
    treatment, so each group is rasterized into small local masks
    (_rasterize_segment_group) and run through the identical splitter
    before being accepted as one or more table regions. n_h_lines /
    n_v_lines are recomputed per sub-region afterward, since the
    group-wide counts no longer apply once a group has been split.

    Returns list of dicts: {bbox, n_h_lines, n_v_lines}
    """
    horiz, vert = _extract_line_segments(page)
    segs = [("h", s) for s in horiz] + [("v", s) for s in vert]
    if not segs:
        return []

    n = len(segs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    bboxes = [_seg_bbox(k, s, bridge_gap) for k, s in segs]
    # naive O(n^2) overlap check -- fine at page-drawing scale (typically
    # low hundreds of segments, not thousands)
    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_overlap(bboxes[i], bboxes[j]):
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    tables = []
    for members in groups.values():
        h_ys = [segs[i][1][0] for i in members if segs[i][0] == "h"]
        v_xs = [segs[i][1][0] for i in members if segs[i][0] == "v"]
        n_h = len(_cluster_positions(h_ys))
        n_v = len(_cluster_positions(v_xs))
        if n_h < min_lines or n_v < min_lines:
            continue  # not enough grid density to plausibly be a table

        xs0, ys0, xs1, ys1 = [], [], [], []
        for i in members:
            kind, seg = segs[i]
            if kind == "h":
                y, x0, x1 = seg
                xs0.append(x0); xs1.append(x1); ys0.append(y); ys1.append(y)
            else:
                x, y0, y1 = seg
                xs0.append(x); xs1.append(x); ys0.append(y0); ys1.append(y1)
        group_bbox = (min(xs0), min(ys0), max(xs1), max(ys1))

        # A shared border/dimension line can bridge a real table into an
        # unrelated diagram panel within the same connected component --
        # split on genuine whitespace gaps before trusting "one connected
        # component" to mean "one table". Same cross-axis splitter raster
        # mode uses, run against a synthetic mask rasterized from this
        # group's own segments (vector mode has no page render to slice).
        h_mask, v_mask, scale, off = _rasterize_segment_group(members, segs, group_bbox)
        min_gap_px = max(20, int(round(25 * scale)))
        sub_rects = _split_by_cross_axis_whitespace(h_mask, v_mask, min_gap_px=min_gap_px)

        gx0, gy0 = group_bbox[0], group_bbox[1]
        for (sx0, sy0, sx1, sy1) in sub_rects:
            # map the pixel sub-rect back into PDF point space
            pb = (
                gx0 + (sx0 - off) / scale,
                gy0 + (sy0 - off) / scale,
                gx0 + (sx1 - off) / scale,
                gy0 + (sy1 - off) / scale,
            )

            # recompute n_h/n_v from only the segments that actually fall
            # inside this sub-rect -- the group-wide counts computed above
            # no longer apply once the group has been split into pieces
            sub_h = [
                segs[i][1][0] for i in members
                if segs[i][0] == "h" and pb[1] - 1.0 <= segs[i][1][0] <= pb[3] + 1.0
            ]
            sub_v = [
                segs[i][1][0] for i in members
                if segs[i][0] == "v" and pb[0] - 1.0 <= segs[i][1][0] <= pb[2] + 1.0
            ]
            sn_h = len(_cluster_positions(sub_h))
            sn_v = len(_cluster_positions(sub_v))
            if sn_h < min_lines or sn_v < min_lines:
                continue  # this sub-region alone isn't dense enough to be a table

            tables.append({"bbox": pb, "n_h_lines": sn_h, "n_v_lines": sn_v})

    return tables


# ---------------------------------------------------------------------------
# Whitespace-gap splitting (a shared outer sheet border/frame can connect
# genuinely distinct content -- e.g. a data table and unrelated diagram
# panels -- into one connected component; this splits them back apart)
# ---------------------------------------------------------------------------

def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter + 1e-6)


def _dedup_tables(tables, iou_threshold=0.5):
    """
    Removes near-duplicate detections of the same physical region --
    e.g. multiple raw connected components (from a border that's
    technically broken into several pieces) each independently
    resolving to overlapping sub-regions after whitespace-gap
    splitting. Keeps the first-seen of any mutually-overlapping group.
    """
    kept = []
    for t in tables:
        if any(_iou(t["bbox"], k["bbox"]) > iou_threshold for k in kept):
            continue
        kept.append(t)
    return kept


def _find_gap_ranges(density, min_gap_len, ink_threshold):
    """
    Given a 1D ink-density profile (pixel count per column or row),
    returns (start, end) index ranges where density stays at or below
    ink_threshold for at least min_gap_len consecutive positions -- a
    real blank margin, not just a momentary dip between two nearby rule
    lines.
    """
    gaps = []
    n = len(density)
    i = 0
    while i < n:
        if density[i] <= ink_threshold:
            j = i
            while j < n and density[j] <= ink_threshold:
                j += 1
            if j - i >= min_gap_len:
                gaps.append((i, j))
            i = j
        else:
            i += 1
    return gaps


def _split_by_cross_axis_whitespace(h_mask, v_mask, min_gap_px=25, gap_density_frac=0.3):
    """
    Splits a connected region into sub-rectangles wherever a wide band
    of comparatively low-density columns or rows separates otherwise
    distinct content -- e.g. a real data table and an unrelated
    diagram/dimensioning panel that happen to share an outer sheet
    border or a bridging line. "Touches the same border line" is weak
    evidence two panels are the same table; a real gap between their
    actual rule-work is much better evidence they're separate. Used by
    both detect_tables_raster (with real page-rendered h_lines/v_lines
    masks) and detect_tables_vector (with synthetic masks rasterized
    from vector segment geometry).

    Column gaps are found from HORIZONTAL-line ink only, and row gaps
    (within a column strip) from VERTICAL-line ink only -- deliberately
    NOT a combined h+v mask. A combined mask breaks specifically on a
    real table with heavily merged cells (e.g. a "reference drawing
    no." column where one label spans many rows): a full-height
    vertical column divider paints ink across the table's ENTIRE
    height, while a merged column's much sparser horizontal dividers
    only cross it a handful of times -- both are legitimately "inside
    the table", but the magnitude difference between them can be as
    large as the difference between "inside the table" and "genuinely
    outside it". A single percentile threshold over a combined mask
    can't tell those apart, and ends up reading the merged column's
    interior as a gap, shredding a real table into slivers too small
    to pass min_lines and losing it entirely (confirmed: this is
    exactly what happened on a real production BOM sheet with a long
    merged "reference drawing no." column).

    Using only horizontal-line ink to find column gaps sidesteps this:
    a real table's horizontal rules span its FULL width, including the
    outer top/bottom border, so every genuine column of the table
    (merged or not) gets crossed by at least that border and shows
    non-trivial horizontal-ink density -- density only drops near zero
    once you're actually past the table's physical edge, which is
    exactly the gap this needs to find. The same logic applies with
    axes swapped when finding row gaps within a column strip.

    The threshold is ADAPTIVE (a fraction of this component's own peak
    density on the relevant axis), not a fixed pixel count -- confirmed
    necessary on a real scanned drawing sheet: engineering drawings
    commonly carry full-width/height zone-marking or border lines
    (e.g. the margin tick marks/index letters on an ISO drawing sheet)
    that cross every column or row, so density in a genuine gap is
    never truly zero, it's just much lower than inside real table
    content. Comparing to this component's own peak density on that
    axis adapts to that automatically instead of guessing a universal
    absolute count.

    Splits columns first (side-by-side panels), then rows within each
    resulting strip (stacked panels). Returns a list of (x0,y0,x1,y1)
    sub-rectangles in the input masks' own coordinate space -- just
    [(0,0,w,h)] (unchanged) if no qualifying gap is found. h_mask and
    v_mask must be the same shape.
    """
    h, w = h_mask.shape

    col_density = (h_mask > 0).sum(axis=0)
    col_threshold = max(2, gap_density_frac * np.percentile(col_density, 85))
    col_gaps = _find_gap_ranges(col_density, min_gap_px, col_threshold)

    col_bounds = [0] + [(g0 + g1) // 2 for g0, g1 in col_gaps] + [w]
    col_bounds = sorted(set(col_bounds))

    sub_regions = []
    for i in range(len(col_bounds) - 1):
        cx0, cx1 = col_bounds[i], col_bounds[i + 1]
        if cx1 - cx0 < min_gap_px:
            continue
        v_strip = v_mask[:, cx0:cx1]
        row_density = (v_strip > 0).sum(axis=1)
        row_threshold = max(2, gap_density_frac * np.percentile(row_density, 85))
        row_gaps = _find_gap_ranges(row_density, min_gap_px, row_threshold)
        row_bounds = [0] + [(g0 + g1) // 2 for g0, g1 in row_gaps] + [h]
        row_bounds = sorted(set(row_bounds))
        for j in range(len(row_bounds) - 1):
            ry0, ry1 = row_bounds[j], row_bounds[j + 1]
            if ry1 - ry0 < min_gap_px:
                continue
            sub_regions.append((cx0, ry0, cx1, ry1))

    return sub_regions if sub_regions else [(0, 0, w, h)]


# ---------------------------------------------------------------------------
# Raster fallback detection (page has no usable vector lines)
# ---------------------------------------------------------------------------

def detect_tables_raster(page, preview_dpi=150, min_lines=3):
    """
    Render the page at a modest preview DPI and find grid-dense regions
    via the same morphological line-detection idea as structure.py
    (erode/dilate with wide kernels to isolate ruling lines), applied
    page-wide, then connected-components on the combined mask. Bboxes
    are converted back to PDF point-space (divide by dpi/72) so they're
    directly usable for a high-DPI re-render regardless of what preview
    resolution was used to find them.

    Each connected component is additionally passed through
    _split_by_cross_axis_whitespace before being accepted as a single
    table -- confirmed necessary on a real scanned drawing sheet, where
    a data table and unrelated diagram panels shared one outer sheet
    border and got detected as a single giant region spanning both.
    Uses h_lines and v_lines separately (not the combined grid) so a
    heavily merged column's sparse horizontal dividers don't get
    misread as a gap next to a full-height vertical divider -- see
    _split_by_cross_axis_whitespace's docstring.
    """
    scale = preview_dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    bin_img = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    h, w = bin_img.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w // 40), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, h // 40)))
    h_lines = cv2.dilate(cv2.erode(bin_img, h_kernel), h_kernel)
    v_lines = cv2.dilate(cv2.erode(bin_img, v_kernel), v_kernel)
    # drop any "line" that's actually a thick blob (redaction mark, ink
    # stamp) rather than a genuine thin rule -- see
    # _remove_thick_line_components's docstring
    h_lines = _remove_thick_line_components(h_lines, axis="h")
    v_lines = _remove_thick_line_components(v_lines, axis="v")
    grid = cv2.bitwise_or(h_lines, v_lines)

    # bridge small gaps between nearby table fragments (e.g. a faint or
    # broken rule) before finding connected components
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    grid_dilated = cv2.dilate(grid, bridge_kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid_dilated, connectivity=8)

    min_gap_px = max(20, int(25 * scale))  # ~25pt of real blank margin

    tables = []
    for label in range(1, n_labels):  # skip background label 0
        x, y, cw, ch, area = stats[label]
        if cw < 30 or ch < 30:
            continue  # too small to be a real table region

        # a shared outer frame can connect unrelated content into one
        # component -- split on genuine whitespace gaps before trusting
        # "connected" as "same table". Uses h_lines/v_lines separately
        # (cross-axis), not the combined grid -- see
        # _split_by_cross_axis_whitespace's docstring for why a combined
        # mask misreads a heavily merged column's interior as a gap.
        region_h_full = h_lines[y:y + ch, x:x + cw]
        region_v_full = v_lines[y:y + ch, x:x + cw]
        sub_rects = _split_by_cross_axis_whitespace(region_h_full, region_v_full, min_gap_px=min_gap_px)

        for (sx0, sy0, sx1, sy1) in sub_rects:
            gx0, gy0 = x + sx0, y + sy0
            scw, sch = sx1 - sx0, sy1 - sy0
            if scw < 30 or sch < 30:
                continue
            region_h = h_lines[gy0:gy0 + sch, gx0:gx0 + scw]
            region_v = v_lines[gy0:gy0 + sch, gx0:gx0 + scw]
            row_sums = np.sum(region_h, axis=1)
            col_sums = np.sum(region_v, axis=0)
            n_h = len(_cluster_positions(list(np.where(row_sums > (scw * 0.3 * 255))[0]), tol=3))
            n_v = len(_cluster_positions(list(np.where(col_sums > (sch * 0.5 * 255))[0]), tol=3))
            if n_h < min_lines or n_v < min_lines:
                continue
            bbox_pts = (gx0 / scale, gy0 / scale, (gx0 + scw) / scale, (gy0 + sch) / scale)
            tables.append({"bbox": bbox_pts, "n_h_lines": n_h, "n_v_lines": n_v})

    return _dedup_tables(tables)


# ---------------------------------------------------------------------------
# Corner-based blob detection (for tables too dense/small to resolve
# individual rules at preview resolution)
# ---------------------------------------------------------------------------

def detect_tables_corner_raster(page, preview_dpi=150, min_coverage=0.2,
                                 min_side_px=12, min_size_pt=25):
    """
    For a table that's genuinely detailed but crammed into a small
    physical area on the page, individual row/column rules can blur
    together at preview DPI -- morphological erosion (used by
    detect_tables_raster to isolate individual lines) can wipe out
    rules that are only 1-2px apart at preview scale, undercounting
    n_h_lines/n_v_lines below the min_lines threshold and losing the
    table entirely.

    This detector never tries to resolve individual rules at preview
    time at all. It only needs the table's OUTER boundary. It works
    off the same LINE masks as detect_tables_raster (erode/dilate with
    long thin kernels -- this is what already tells a straight rule
    apart from text glyphs and word gaps, regardless of how densely
    packed the rules are) rather than raw ink, then closes those lines
    into a solid blob and fits cv2.minAreaRect to the largest one --
    which by construction is always a true rectangle (adjacent sides at
    90 degrees, opposite sides equal length and parallel) -- giving 4
    corner points directly.

    Working from raw ink instead of line masks would also flag a boxed
    paragraph or a filled-in text block as a "table" (a hollow
    rectangle's OUTER contour covers its full area even though the
    interior is empty, and cv2.contourArea can't tell that apart from a
    genuinely filled blob). To reject those, this additionally requires
    that line-ink populates most of the interior height AND width of
    the candidate box (row_coverage / col_coverage), not just two thin
    bands at the top/bottom or left/right edges -- a real table (even
    one whose individual rules can't be told apart at this resolution)
    has rule-ink spread almost continuously through its interior; a
    lone bordered box only has ink at its 4 edges.

    Internal grid resolution is left entirely to structure.py, which
    runs later against the high-DPI crop this produces -- exactly where
    that resolution is actually available.

    min_size_pt enforces a minimum PHYSICAL footprint (PDF points, not
    preview pixels) in both dimensions. min_side_px alone isn't enough
    to rule out a single character stroke -- confirmed empirically on a
    real scanned drawing: once redaction-blob false positives were
    fixed (_remove_thick_line_components), individual digit strokes
    within ordinary text started passing the row/col-coverage check
    instead, since one straight stroke segment trivially clears
    min_coverage within its own tiny bounding box. A real table, even a
    minimal one, has to span multiple rows/columns and won't have a
    smaller side under a reasonable point threshold; a lone glyph will.

    Returns list of dicts: {corners: [(x,y)*4] in PDF points, angle,
    fill_ratio, row_coverage, col_coverage}.
    """
    scale = preview_dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    bin_img = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    h, w = bin_img.shape
    # shorter kernels than detect_tables_raster's page-relative sizing --
    # a small/dense table's individual rules may be short, and we're not
    # trying to count them individually here, just confirm "this is
    # line-like ink" as opposed to text
    line_len = max(10, min(w, h) // 60)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    h_lines = cv2.dilate(cv2.erode(bin_img, h_kernel), h_kernel)
    v_lines = cv2.dilate(cv2.erode(bin_img, v_kernel), v_kernel)
    # drop any "line" that's actually a thick blob (redaction mark, ink
    # stamp) rather than a genuine thin rule -- see
    # _remove_thick_line_components's docstring
    h_lines = _remove_thick_line_components(h_lines, axis="h")
    v_lines = _remove_thick_line_components(v_lines, axis="v")
    line_mask = cv2.bitwise_or(h_lines, v_lines)

    # close small gaps so a densely-ruled table's many close-together
    # lines fuse into one solid blob rather than staying as fragments
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # also close the raw ink (not just the line mask) with a bigger
    # kernel -- at extreme density one axis of rules can blur past
    # recovery entirely (e.g. row spacing collapses below 1px while
    # column spacing is still a few px apart -- an unrecoverable data
    # loss on that axis, same root cause as the original resolution
    # finding, not something any detector can fix). This still leaves
    # a solid dark rectangular footprint even when one axis' structure
    # is gone, which is what min_fill_ratio below checks for.
    ink_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, line_len))
    ink_closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, ink_close_kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (min_side_px * min_side_px):
            continue  # too small to be a real table blob, likely noise
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < min_side_px or ch < min_side_px:
            continue

        region_h = h_lines[y:y + ch, x:x + cw]
        region_v = v_lines[y:y + ch, x:x + cw]
        row_coverage = float(np.mean(np.any(region_h > 0, axis=1)))
        col_coverage = float(np.mean(np.any(region_v > 0, axis=0)))
        region_ink = bin_img[y:y + ch, x:x + cw]
        ink_fill_ratio = float(np.mean(region_ink > 0))

        # accept if EITHER axis shows real periodic rule structure
        # (a text paragraph shows this on neither axis -- individual
        # glyphs/words never form long continuous runs) AND the region
        # is a solid dark footprint overall, not just two thin edge
        # bands (which is what a hollow box/border alone would give)
        has_structure = row_coverage >= min_coverage or col_coverage >= min_coverage
        if not has_structure or ink_fill_ratio < 0.12:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        if rw < min_side_px or rh < min_side_px:
            continue
        if min(rw, rh) / scale < min_size_pt:
            continue  # smaller side too small physically to be a real table
        fill_ratio = area / (rw * rh + 1e-6)

        corners_px = cv2.boxPoints(rect)
        corners_pt = [(float(px) / scale, float(py) / scale) for px, py in corners_px]
        tables.append({"corners": corners_pt, "angle": float(angle),
                        "fill_ratio": round(float(fill_ratio), 3),
                        "row_coverage": round(row_coverage, 3),
                        "col_coverage": round(col_coverage, 3),
                        "ink_fill_ratio": round(ink_fill_ratio, 3)})

    return tables


def _corners_to_bbox(corners):
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# Content-type classification (data table vs. diagram/illustration grid)
# ---------------------------------------------------------------------------

def classify_table_content(page, bbox, n_h_lines=None, n_v_lines=None):
    """
    A rectangular grid of large illustrated cells (e.g. a 2x6 panel of
    assembly-step diagrams with captions) is geometrically a completely
    valid table by every test above -- it has real ruling lines, right
    angles, equal/parallel sides. But it isn't tabular DATA, and OCR'ing
    "cell values" out of diagrams for a BOM spreadsheet is meaningless.
    This flags that case using two generic signals, not a content
    wordlist:

      1. Curve/embedded-image density: a diagram is drawn with bezier
         curves and/or contains a raster image; a data cell is just
         text and straight rules. detect_tables_vector deliberately
         ignores curves entirely when finding table geometry ("c" items
         aren't table rules) -- here, their PRESENCE is exactly the
         useful signal, inverted.
      2. Average cell size: only computable when the caller has real
         row/column line counts (vector or raster-grid detections; the
         corner/blob detector doesn't resolve individual lines, so this
         signal is simply skipped for those -- curve/image density
         alone still applies). A BOM data row is usually close to one
         text line tall (~10-25pt); an illustration cell has to fit a
         whole diagram and is typically much larger.

    Returns a dict: {content_type: "likely_data_table" |
    "likely_diagram_grid" | "uncertain", n_curves, n_embedded_images,
    avg_row_height_pt, avg_col_width_pt}. This is a QA flag for a human
    to weigh, not a hard filter -- ambiguous evidence is reported as
    "uncertain" rather than guessed.
    """
    x0, y0, x1, y1 = bbox
    rect = fitz.Rect(x0, y0, x1, y1)

    n_curves = 0
    for d in page.get_drawings():
        for item in d.get("items", []):
            if item[0] != "c":
                continue
            pts = item[1:]
            if any(rect.contains(fitz.Point(p.x, p.y)) for p in pts if hasattr(p, "x")):
                n_curves += 1

    n_images = 0
    for img in page.get_images(full=True):
        try:
            img_rects = page.get_image_rects(img[0])
        except Exception:
            img_rects = []
        for ir in img_rects:
            if rect.intersects(ir):
                n_images += 1

    avg_row_height = avg_col_width = None
    if n_h_lines and n_h_lines > 1:
        avg_row_height = round((y1 - y0) / (n_h_lines - 1), 1)
    if n_v_lines and n_v_lines > 1:
        avg_col_width = round((x1 - x0) / (n_v_lines - 1), 1)

    diagram_evidence = 0
    data_evidence = 0

    if n_curves >= 4 or n_images >= 1:
        diagram_evidence += 1
    elif n_curves == 0 and n_images == 0:
        data_evidence += 1

    if avg_row_height is not None:
        if avg_row_height > 50:
            diagram_evidence += 1
        elif avg_row_height < 30:
            data_evidence += 1

    if diagram_evidence >= 1 and diagram_evidence > data_evidence:
        content_type = "likely_diagram_grid"
    elif data_evidence >= 1 and data_evidence > diagram_evidence:
        content_type = "likely_data_table"
    else:
        content_type = "uncertain"

    return {
        "content_type": content_type,
        "n_curves": n_curves,
        "n_embedded_images": n_images,
        "avg_row_height_pt": avg_row_height,
        "avg_col_width_pt": avg_col_width,
    }


def detect_tables_on_page(page, preview_dpi, min_lines):
    """
    Always run all THREE detectors, never just one-or-the-other. A
    single page can legitimately mix a vector-drawn table, a scanned
    table with a normal (resolvable) grid, and a scanned table so dense
    it can only be found as a blob:
      1. vector       -- exact PDF line geometry, preferred when available
      2. raster grid  -- line-counting on a preview raster, for scanned
                          tables whose rules are still resolvable at
                          preview DPI
      3. raster blob  -- outer-rectangle-only detection, for scanned
                          tables too dense/small to resolve individual
                          rules at preview DPI (see
                          detect_tables_corner_raster's docstring)
    Vector is preferred on overlap (exact geometry); raster grid is
    preferred over blob on overlap (it already knows real row/col
    counts, useful for the manifest/QA even though structure.py will
    redo this at full DPI regardless). A raster hit with no real
    overlap against anything already found is a genuinely separate
    table.
    """
    vector_found = detect_tables_vector(page, min_lines=min_lines)
    raster_found = detect_tables_raster(page, preview_dpi=preview_dpi, min_lines=min_lines)
    corner_found = detect_tables_corner_raster(page, preview_dpi=preview_dpi)

    combined = [{"bbox": t["bbox"], "corners": None,
                 "n_h_lines": t["n_h_lines"], "n_v_lines": t["n_v_lines"],
                 "method": "vector"} for t in vector_found]

    for rt in raster_found:
        if any(_iou(rt["bbox"], c["bbox"]) > 0.3 for c in combined):
            continue  # same table already found via vector, more precisely
        combined.append({"bbox": rt["bbox"], "corners": None,
                          "n_h_lines": rt["n_h_lines"], "n_v_lines": rt["n_v_lines"],
                          "method": "raster_fallback"})

    for ct in corner_found:
        cbbox = _corners_to_bbox(ct["corners"])
        if any(_iou(cbbox, c["bbox"]) > 0.3 for c in combined):
            continue  # already found by a more informative detector
        combined.append({"bbox": cbbox, "corners": ct["corners"],
                          "n_h_lines": None, "n_v_lines": None,
                          "method": "raster_corner_blob",
                          "angle": ct["angle"], "fill_ratio": ct["fill_ratio"]})

    return combined


def _find_content_edge(ink_1d, ascending, isolation_gap_px=40, min_run_ink=5,
                        max_isolated_run_px=15):
    """
    Scan a 1D ink-density profile inward from one end and return the
    index where SUSTAINED table content begins.

    A plain "stop at the first nonzero sample" rule breaks when the
    detected bbox swept up something outside the real table -- e.g. a
    sheet's zone-divider/frame rule merged in via a shared-border
    detection error. That line sits far from the real table with a
    long genuinely blank run on both sides, but a naive scan still
    treats it as "the edge" and refuses to trim past it, leaving all
    the blank space behind it uncropped.

    Here, hitting ink only counts as the true edge if a further
    isolation_gap_px window ahead also carries real ink (min_run_ink
    total) -- i.e. the content keeps going. An isolated hairline with
    nothing else nearby is skipped as noise, and the scan resumes
    trimming past it. A genuine merged-cell edge is never isolated
    like this: it always sits directly against the rest of the
    table's ongoing structure, so it's never skipped.
    """
    n = len(ink_1d)
    step = 1 if ascending else -1
    idx = 0 if ascending else n - 1

    while 0 <= idx < n:
        if ink_1d[idx] > 0:
            edge = idx
            # walk to the far end of this contiguous ink run (the line's
            # own thickness) so the lookahead below checks what's BEYOND
            # it, not the run's own pixels
            run_end = idx
            while 0 <= run_end + step < n and ink_1d[run_end + step] > 0:
                run_end += step
            run_width = abs(run_end - idx) + 1
            if run_width > max_isolated_run_px:
                # substantial width -- this is real table structure
                # (rows/columns/text stack up into one wide run when
                # summed), never a stray single rule. Accept immediately,
                # regardless of what is or isn't further beyond it.
                return edge
            lo = run_end + step
            hi = lo + step * isolation_gap_px
            lo, hi = min(lo, hi), max(lo, hi)
            lo = max(0, lo)
            hi = min(n, hi + 1)
            if ink_1d[lo:hi].sum() > min_run_ink:
                return edge  # sustained content -- this is the real edge
            # isolated hairline with nothing nearby: skip past it and
            # the already-scanned gap, keep trimming inward
            idx = hi if ascending else lo - 1
            continue
        idx += step

    return idx  # fully blank all the way in this direction


def _trim_whitespace_margins(crop_bgr, safety_px=2,
                              isolation_gap_px=40, min_run_ink=5):
    """
    Shrink a table crop to its actual ink extent by trimming blank
    rows/columns inward from each of the 4 outer edges, stopping once
    SUSTAINED content is found (see _find_content_edge) -- not merely
    the first pixel of ink, which can be an isolated stray line rather
    than the table itself.

    This only ever removes outside-in whitespace: true blank padding
    plus any isolated noise line and the blank space around it. It
    can never reach into the table's interior: a blank interior
    spacer row/column that exists because of a merged cell (e.g. a
    multi-row "REFER DRAWING NO." block with no per-row divider or
    per-row text) sits directly adjacent to the rest of the table's
    real content, so it is never mistaken for isolated noise and is
    never the outermost row/column once real content is found.

    A small safety_px is added back on each side after trimming so an
    anti-aliased outer rule isn't clipped at exactly zero margin.

    Returns (trimmed_crop, (top, bottom, left, right)) where the tuple
    is the pixel offsets actually cut, for bbox bookkeeping.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)

    h, w = bin_img.shape
    row_ink = bin_img.sum(axis=1)
    col_ink = bin_img.sum(axis=0)

    top = _find_content_edge(row_ink, True, isolation_gap_px, min_run_ink)
    bottom = _find_content_edge(row_ink, False, isolation_gap_px, min_run_ink) + 1
    left = _find_content_edge(col_ink, True, isolation_gap_px, min_run_ink)
    right = _find_content_edge(col_ink, False, isolation_gap_px, min_run_ink) + 1

    top = max(0, min(top, h - 1) - safety_px)
    bottom = min(h, max(bottom, top + 1) + safety_px)
    left = max(0, min(left, w - 1) - safety_px)
    right = min(w, max(right, left + 1) + safety_px)

    return crop_bgr[top:bottom, left:right], (top, bottom, left, right)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def detect_and_render(pdf_path, output_dir, dpi=600, preview_dpi=300,
                       min_lines=3, padding_pt=6.0):
    os.makedirs(output_dir, exist_ok=True)
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    manifest = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_rect = page.rect

        found = detect_tables_on_page(page, preview_dpi=preview_dpi, min_lines=min_lines)
        n_vector = sum(1 for t in found if t["method"] == "vector")
        n_raster = sum(1 for t in found if t["method"] == "raster_fallback")
        n_blob = sum(1 for t in found if t["method"] == "raster_corner_blob")

        print(f"[page {page_idx}] tables_found={len(found)} "
              f"(vector={n_vector}, raster_fallback={n_raster}, raster_corner_blob={n_blob})")

        overview_pix = page.get_pixmap(matrix=fitz.Matrix(preview_dpi / 72.0, preview_dpi / 72.0))
        overview_img = np.frombuffer(overview_pix.samples, dtype=np.uint8).reshape(
            overview_pix.height, overview_pix.width, overview_pix.n)
        overview_bgr = cv2.cvtColor(overview_img[:, :, :3], cv2.COLOR_RGB2BGR).copy()
        prev_scale = preview_dpi / 72.0

        for t_idx, table in enumerate(found):
            x0, y0, x1, y1 = table["bbox"]
            # pad, then clip to page bounds
            x0 = max(page_rect.x0, x0 - padding_pt)
            y0 = max(page_rect.y0, y0 - padding_pt)
            x1 = min(page_rect.x1, x1 + padding_pt)
            y1 = min(page_rect.y1, y1 + padding_pt)
            clip_rect = fitz.Rect(x0, y0, x1, y1)

            # crop-before-rasterize: render ONLY this region, at full
            # target DPI, straight from the PDF -- not a downscaled
            # render that gets cropped afterward
            zoom = dpi / 72.0
            crop_pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip_rect)
            crop_img = np.frombuffer(crop_pix.samples, dtype=np.uint8).reshape(
                crop_pix.height, crop_pix.width, crop_pix.n)
            crop_bgr = cv2.cvtColor(crop_img[:, :, :3], cv2.COLOR_RGB2BGR)

            warped = False
            if table["method"] == "raster_corner_blob" and abs(table["angle"] % 90) > 1.0:
                # corner-blob detection can find a rotated/skewed table;
                # straighten it via perspective warp using the actual 4
                # corners (rescaled from PDF points into this crop's own
                # pixel space) so structure.py sees level rows/columns
                # rather than a tilted grid it has no rotation-handling for
                src_pts = np.float32([[(cx - x0) * zoom, (cy - y0) * zoom]
                                       for cx, cy in table["corners"]])
                out_w = int(round(max(
                    np.linalg.norm(src_pts[0] - src_pts[1]),
                    np.linalg.norm(src_pts[2] - src_pts[3]))))
                out_h = int(round(max(
                    np.linalg.norm(src_pts[1] - src_pts[2]),
                    np.linalg.norm(src_pts[3] - src_pts[0]))))
                out_w, out_h = max(out_w, 10), max(out_h, 10)
                dst_pts = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                crop_bgr = cv2.warpPerspective(crop_bgr, M, (out_w, out_h))
                warped = True

            crop_bgr, (trim_top, trim_bottom, trim_left, trim_right) = \
                _trim_whitespace_margins(crop_bgr)
            if not warped:
                # pixel trim -> pt trim (only meaningful in un-warped,
                # axis-aligned point-space; a warped crop's coordinate
                # space is synthetic to the perspective transform, so
                # its bbox_pdf_points already can't be trusted precisely)
                x0 = x0 + trim_left / zoom
                y0 = y0 + trim_top / zoom
                x1 = x0 + (trim_right - trim_left) / zoom
                y1 = y0 + (trim_bottom - trim_top) / zoom

            fname = f"page{page_idx}_table{t_idx}.png"
            fpath = os.path.join(tables_dir, fname)
            cv2.imwrite(fpath, crop_bgr)
            out_h, out_w = crop_bgr.shape[:2]

            classification = classify_table_content(
                page, (x0, y0, x1, y1),
                n_h_lines=table["n_h_lines"], n_v_lines=table["n_v_lines"])

            manifest.append({
                "page": page_idx,
                "table_index": t_idx,
                "method": table["method"],
                "content_type": classification["content_type"],
                "content_signals": {k: v for k, v in classification.items() if k != "content_type"},
                "detected_grid": {"n_h_lines": table["n_h_lines"], "n_v_lines": table["n_v_lines"]},
                "bbox_pdf_points": [round(v, 1) for v in (x0, y0, x1, y1)],
                "render_dpi": dpi,
                "perspective_warped": warped,
                "crop_filename": os.path.join("tables", fname),
                "crop_size_px": [out_w, out_h],
            })

            # draw this table's box on the preview overview image --
            # diagram grids get their own color so a reviewer can spot
            # them at a glance without opening the manifest
            px0, py0, px1, py1 = [int(v * prev_scale) for v in (x0, y0, x1, y1)]
            if classification["content_type"] == "likely_diagram_grid":
                color = (0, 165, 255)  # orange -- flagged, likely not BOM data
            elif table["method"] == "raster_corner_blob":
                color = (255, 128, 0)
            else:
                color = (0, 0, 255)
            label = f"table {t_idx} ({table['method']})"
            if classification["content_type"] != "likely_data_table":
                label += f" [{classification['content_type']}]"
            cv2.rectangle(overview_bgr, (px0, py0), (px1, py1), color, 2)
            cv2.putText(overview_bgr, label, (px0, max(12, py0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        overview_path = os.path.join(output_dir, f"overview_page{page_idx}.png")
        cv2.imwrite(overview_path, overview_bgr)

    manifest_path = os.path.join(output_dir, "tables.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{len(manifest)} table(s) detected across {len(doc)} page(s).")
    print(f"Crops:    {tables_dir}/")
    print(f"Overview: {output_dir}/overview_page*.png")
    print(f"Manifest: {manifest_path}")
    return manifest


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    flags = {"--dpi", "--preview-dpi", "--min-lines", "--padding"}
    args = [a for a in sys.argv[1:] if not any(a.startswith(f) for f in flags)]
    flag_args = [a for a in sys.argv[1:] if any(a.startswith(f) for f in flags)]
    opts = dict(a.split("=", 1) for a in flag_args)

    pdf_path = args[0]
    output_dir = args[1] if len(args) > 1 else "table_detection_output"
    dpi = int(opts.get("--dpi", 300))
    preview_dpi = int(opts.get("--preview-dpi", 150))
    min_lines = int(opts.get("--min-lines", 3))
    padding = float(opts.get("--padding", 6))

    detect_and_render(pdf_path, output_dir, dpi=dpi, preview_dpi=preview_dpi,
                       min_lines=min_lines, padding_pt=padding)


if __name__ == "__main__":
    main()