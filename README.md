# table_detection — Stage 0, ahead of the main extraction pipeline

This folder is the new front-end stage that runs BEFORE `main.py`'s
5-stage extraction pipeline (structure.py -> ocr -> header.py ->
postprocess.py -> export_xlsx.py) described in the top-level README.

## What it does

`detect_tables.py` takes a PDF (which may contain multiple tables on one
or more pages, mixed with diagrams/title blocks/free text) and produces
one high-DPI, correctly-cropped PNG per table -- ready to feed straight
into `main.py <crop.png> output_dir`.

Three detection strategies run on every page and get merged:

1. **Vector** -- reads the PDF's own drawn line/rect geometry directly
   (exact coordinates, no thresholding). Preferred whenever available.
2. **Raster grid** -- for scanned/embedded-image tables whose rules are
   still individually resolvable at preview DPI; reuses the same
   morphological line-detection idea as `structure.py`, applied
   page-wide.
3. **Raster corner/blob** -- for a table that's genuinely detailed but
   crammed into a small physical area, where individual rules blur
   together at preview DPI. Finds only the table's OUTER boundary (a
   dense table still shows up as one solid blob even when its rules
   can't be told apart) via `cv2.minAreaRect`, which is always a true
   rectangle by construction. Straightens a rotated/skewed result via
   perspective warp before saving.

Every detected table is also run through `classify_table_content()`,
which flags `likely_data_table` vs. `likely_diagram_grid` vs.
`uncertain` -- e.g. a 2x6 panel of illustrated assembly-step diagrams is
a geometrically valid grid too, but isn't BOM data and shouldn't be fed
through OCR cell-value extraction the same way. This is a QA flag, not a
hard filter, in the same spirit as `export_xlsx.py`'s existing
length-outlier flagging.

## Usage

```bash
pip install pymupdf --break-system-packages
python3 table_detection/detect_tables.py <your.pdf> table_detection_output --dpi=300
```

Output:
```
table_detection_output/
    tables/pageP_tableI.png   -- one high-DPI crop per table, feed into main.py
    overview_pageP.png        -- full page with detected tables boxed + labeled,
                                  for a human to sanity-check (no cell content,
                                  safe to share even from a confidential source PDF)
    tables.json                -- bbox, detection method, content_type, grid info
```

Then run the existing pipeline against each crop:
```bash
python3 main.py table_detection_output/tables/page0_table0.png output_dir --engine=paddle
```

## tests/

`make_test_pdf.py` generates a set of synthetic multi-table PDFs (no
real/confidential data) used to validate detection before running
against the actual source PDF:

- `test_multi_table.pdf` -- 3 real tables of varying size (incl. a tiny
  2x2 edge case) + deliberate false-positive traps (underlined text, a
  boxed paragraph, flanking vertical rules, a title block) that must
  NOT be detected as tables.
- `test_multi_table_mixed.pdf` -- same page plus one table embedded as
  a scanned image (mixed vector + raster content on one page).
- `test_multi_table_dense.pdf` / `test_multi_table_dense_rotated.pdf` --
  a small, very dense scanned table (stress-tests the corner/blob
  detector); the rotated variant also stress-tests the
  perspective-straightening path.
- `test_multi_table_diagram_grid.pdf` -- a 2x6 illustrated panel, to
  confirm it's detected as a grid but correctly flagged
  `likely_diagram_grid` rather than treated as BOM data.

Rerun `python3 make_test_pdf.py` inside `tests/` any time detection
logic changes, to confirm nothing regressed before trusting it against
a real (confidential) PDF.
