"""Generates a harder synthetic multi-table PDF to validate
detect_tables.py without touching any real/confidential data. Includes:
  - three real tables of very different shapes/sizes (including one
    tiny 2x2 edge case)
  - several deliberate FALSE-POSITIVE TRAPS: underlined text, a boxed
    paragraph, flanking vertical rules around text, and stacked
    underlines -- all things that superficially involve straight lines
    near text but are not tables and must NOT be detected as one
  - a scanned/rasterized table pasted onto the same page as vector
    tables, to test that mixed vector+raster pages are both covered
"""
import fitz
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

PAGE_W, PAGE_H = A4


def draw_grid(c, x0, y0, x1, y1, n_rows, n_cols):
    row_h = (y1 - y0) / n_rows
    col_w = (x1 - x0) / n_cols
    for r in range(n_rows + 1):
        y = y0 + r * row_h
        c.line(x0, y, x1, y)
    for cidx in range(n_cols + 1):
        x = x0 + cidx * col_w
        c.line(x, y0, x, y1)


def make_base_pdf(path):
    c = canvas.Canvas(path, pagesize=A4)
    c.setLineWidth(1)

    # --- Real tables, varied sizes ---
    # Table 0: 8 rows x 6 cols
    draw_grid(c, 60, 620, 500, 780, n_rows=8, n_cols=6)
    # Table 1: 12 rows x 4 cols (tall/narrow, more rows)
    draw_grid(c, 60, 380, 340, 600, n_rows=12, n_cols=4)
    # Table 2: tiny 2x2 edge case (minimum grid density: exactly 3 h + 3 v lines)
    draw_grid(c, 400, 380, 500, 440, n_rows=2, n_cols=2)

    # --- False-positive traps ---
    c.setFont("Helvetica", 9)

    # Trap 1: underlined heading -- one line under text, no verticals
    c.drawString(60, 340, "Section Heading (underlined, not a table)")
    c.line(60, 335, 280, 335)

    # Trap 2: boxed paragraph -- a rectangle around body text (2 h-lines,
    # 2 v-lines from the 4 edges -- below the 3x3 density threshold)
    c.rect(60, 260, 240, 60)
    c.drawString(65, 300, "This paragraph is boxed for emphasis.")
    c.drawString(65, 288, "It is NOT tabular data, just bordered text.")
    c.drawString(65, 276, "Detector must not flag this as a table.")

    # Trap 3: flanking vertical rules around a text block (2 v-lines, 0 h-lines)
    c.line(340, 260, 340, 340)
    c.line(500, 260, 500, 340)
    c.drawString(350, 310, "Text flanked by two")
    c.drawString(350, 296, "vertical rules only --")
    c.drawString(350, 282, "still not a grid.")

    # Trap 4: several stacked short underlines (3 h-lines, 0 v-lines --
    # tempting because it clears the h-line count alone, but must fail
    # on the v-line requirement)
    for i, y in enumerate([200, 188, 176]):
        c.drawString(60, y + 3, f"List item {i+1} underlined")
        c.line(60, y, 220, y)

    # Title block -- single box, should never register as a table
    c.rect(60, 60, 200, 100)
    c.setFont("Helvetica", 8)
    c.drawString(65, 145, "TITLE BLOCK - NOT A TABLE")

    c.save()


def add_scanned_table_overlay(base_path, out_path):
    """
    Simulates a page that mixes vector-drawn tables with one
    scanned/embedded-image table -- e.g. a photocopied insert pasted
    into an otherwise digitally-drawn sheet. Renders a small ruled
    table to a raster image, then stamps that image (not vector lines)
    onto a copy of the base PDF.
    """
    # build a standalone small table, rasterize it
    tmp = canvas.Canvas("_scan_src.pdf", pagesize=(220, 140))
    draw_grid(tmp, 10, 10, 210, 130, n_rows=5, n_cols=3)
    tmp.save()

    src_doc = fitz.open("_scan_src.pdf")
    pix = src_doc[0].get_pixmap(matrix=fitz.Matrix(4, 4))  # rasterize at 4x
    pix.save("_scan_src.png")

    doc = fitz.open(base_path)
    page = doc[0]
    # place the scanned table image in the empty area near top-right
    target_rect = fitz.Rect(340, 700, 540, 790)
    page.insert_image(target_rect, filename="_scan_src.png")
    doc.save(out_path)


def add_dense_scanned_table(base_path, out_path, target_rect_pts, n_rows=45, n_cols=6, angle_deg=0):
    """
    Simulates your actual failure case: a genuinely detailed table
    (many rows) squeezed into a small physical footprint, scanned/
    embedded as an image rather than vector lines -- so at preview DPI
    the individual row rules are only ~1-2px apart and blur together,
    which is exactly what breaks strategy 2 (raster grid line-counting)
    and is what the corner/blob detector (strategy 3) exists for.

    target_rect_pts: (x0,y0,x1,y1) placement on the base page, in PDF
    points -- caller picks a free area so this doesn't overlap existing
    content. angle_deg optionally rotates the source table before
    embedding, to also test the perspective-straightening path.
    """
    x0, y0, x1, y1 = target_rect_pts
    src_w, src_h = (x1 - x0), (y1 - y0)
    pad = 8
    tmp = canvas.Canvas("_dense_src.pdf", pagesize=(src_w, src_h))
    draw_grid(tmp, pad, pad, src_w - pad, src_h - pad, n_rows=n_rows, n_cols=n_cols)
    tmp.save()

    src_doc = fitz.open("_dense_src.pdf")
    # rasterize at a fairly high internal resolution (this represents
    # "the details ARE there if you zoom in", per your description --
    # the source scan isn't inherently low quality, it's just that the
    # table occupies a small footprint on the page)
    pix = src_doc[0].get_pixmap(matrix=fitz.Matrix(8, 8))
    pix.save("_dense_src.png")

    if angle_deg:
        img = cv2.imread("_dense_src.png", cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255, 255))
        cv2.imwrite("_dense_src.png", rotated)

    doc = fitz.open(base_path)
    page = doc[0]
    page.insert_image(fitz.Rect(x0, y0, x1, y1), filename="_dense_src.png")
    doc.save(out_path)


def add_diagram_grid(base_path, out_path, target_rect_pts, n_rows=2, n_cols=6, blank_page=True):
    """
    Simulates a 2x6 panel of illustrated cells (e.g. assembly-step
    diagrams with captions) -- geometrically a completely valid grid
    (real ruling lines, right angles, equal/parallel sides), but NOT
    tabular data. Each cell gets a vector-drawn circle+line "diagram"
    (bezier curves, which detect_tables_vector deliberately ignores as
    table rules but classify_table_content specifically looks for) plus
    a short caption, so this exercises the content-type classifier: the
    grid should still be DETECTED (it's real geometry) but flagged
    likely_diagram_grid rather than silently treated as BOM data.

    blank_page=True builds this on its own fresh page rather than
    layering onto base_path's already-busy layout, so the grid's own
    detection/classification can be checked in isolation.
    """
    x0, y0, x1, y1 = target_rect_pts
    if blank_page:
        doc = fitz.open()
        doc.new_page(width=A4[0], height=A4[1])
    else:
        doc = fitz.open(base_path)
    page = doc[0]
    cw, ch = (x1 - x0) / n_cols, (y1 - y0) / n_rows

    shape = page.new_shape()
    for r in range(n_rows + 1):
        y = y0 + r * ch
        shape.draw_line(fitz.Point(x0, y), fitz.Point(x1, y))
    for c_ in range(n_cols + 1):
        x = x0 + c_ * cw
        shape.draw_line(fitz.Point(x, y0), fitz.Point(x, y1))
    shape.finish(width=1)

    for r in range(n_rows):
        for c_ in range(n_cols):
            cx, cy = x0 + (c_ + 0.5) * cw, y0 + (r + 0.4) * ch
            radius = min(cw, ch) * 0.28
            diag_shape = page.new_shape()
            diag_shape.draw_circle(fitz.Point(cx, cy), radius)  # bezier curves
            diag_shape.draw_line(fitz.Point(cx - radius, cy), fitz.Point(cx + radius, cy))
            diag_shape.finish(width=1)
            diag_shape.commit()
            page.insert_text(fitz.Point(x0 + c_ * cw + 4, y0 + (r + 1) * ch - 6),
                              f"Step {r * n_cols + c_ + 1}", fontsize=7)
    shape.commit()
    doc.save(out_path)


def main():
    make_base_pdf("test_multi_table.pdf")
    add_scanned_table_overlay("test_multi_table.pdf", "test_multi_table_mixed.pdf")
    # free zone on the page: right margin strip, x[505,590] y[600,780],
    # doesn't overlap table0 (x<=500), table1, table2, or any trap
    dense_zone = (505, 600, 590, 780)
    add_dense_scanned_table("test_multi_table.pdf", "test_multi_table_dense.pdf",
                             dense_zone, n_rows=45, n_cols=6, angle_deg=0)
    add_dense_scanned_table("test_multi_table.pdf", "test_multi_table_dense_rotated.pdf",
                             dense_zone, n_rows=45, n_cols=6, angle_deg=7)
    add_diagram_grid("test_multi_table.pdf", "test_multi_table_diagram_grid.pdf",
                      (60, 60, 535, 780), n_rows=2, n_cols=6, blank_page=True)
    print("wrote test_multi_table.pdf (vector only)")
    print("wrote test_multi_table_mixed.pdf (vector tables + one embedded scanned table)")
    print("wrote test_multi_table_dense.pdf (vector tables + one small/dense scanned table)")
    print("wrote test_multi_table_dense_rotated.pdf (same, scanned table rotated 7deg)")
    print("wrote test_multi_table_diagram_grid.pdf (2x6 illustrated grid, not BOM data)")


if __name__ == "__main__":
    main()
