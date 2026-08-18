"""
Single entry point: extraction + crop export.

Usage:
    python3 main.py <input_image_path> [output_dir] [--engine=paddle|tesseract] [--vocab=on|off]

What it does, in one run:
  1. Detects table structure (rows/columns/merges), geometry only --
     structure.py
  2. Runs targeted OCR per cell/merge block -- ocr_paddle.py (default) or
     ocr_targeted.py (Tesseract)
  2.5. Reads the column names/count from the header row's own OCR text
     and normalizes them against a vocabulary -- header.py +
     header_vocab.py. No fixed/hardcoded column list -- whatever the
     sheet's header row actually says becomes the schema.
  3. Applies domain-aware corrections -- postprocess.py
  4. Exports the final Excel file, in correct reading order -- export_xlsx.py
  5. Exports every individual cell/merge-block crop as its own image
  6. Prints a summary

Engines:
    --engine=paddle     (default) PaddleOCR -- CPU-friendly, deep-learning
                        based, generally more accurate than Tesseract on
                        small/low-contrast text.
    --engine=tesseract  Classical OCR via pytesseract.

Header position:
    --header=bottom     (default) The header row -- and therefore the
                        table's reading order -- is at the BOTTOM of the
                        grid. Many BOM sheets build the table upward from
                        a title block, so the sheet reads bottom-up.
    --header=top        Header row is the first (topmost) grid row, table
                        reads top-down as usual.

Vocabulary:
    --vocab=on          (default) Column names are normalized against
                        header_vocab.py's known column list (e.g.
                        "SR.No." -> "SL"), and vocabulary-likeness is
                        used as one signal for locating the header row.
    --vocab=off         Disables vocabulary use everywhere in this run.
                        Column names come out as exactly the cleaned raw
                        OCR text read off the header row -- no
                        normalization, no vocabulary-based header-row
                        scoring. Useful for seeing the pipeline's raw
                        output on an unfamiliar sheet, or for auditing
                        what OCR actually read before any vocabulary
                        correction is applied.
"""
import sys
import os
import csv
import cv2

from structure import TableStructure
from ocr_mapping import classify_confusion_matrix
from postprocess import correct_cell
from export_xlsx import export
from header import build_header

try:
    from ocr_targeted import run_targeted_ocr
except ImportError:
    run_targeted_ocr = None

try:
    from ocr_paddle import run_targeted_ocr_paddle, paddle_available
except ImportError:
    run_targeted_ocr_paddle = None
    paddle_available = lambda: False


# ---------- Stage 1-4: extraction pipeline ----------

def run_extraction(img_path, out_xlsx, col_names=None, engine="paddle",
                    header_position="bottom", use_vocab=True):
    """
    header_position: "bottom" (default) or "top". These BOM sheets build
    the table upward from a title block at the bottom of the page, so
    the header row -- and the column names/count -- are read from the
    bottom of the grid, and the table itself reads bottom-up. Pass
    col_names explicitly to skip header discovery and force a specific
    schema instead.

    use_vocab: True (default) normalizes discovered column names
    against header_vocab.py's known column list (e.g. "SR.No." ->
    "SL") and uses vocabulary-likeness as one signal for locating the
    header row. False disables vocabulary use everywhere in this run --
    column names come out as exactly the cleaned raw OCR text, and
    header-row detection falls back to the vocabulary-free
    row-similarity signal / header_position instead. Banner-row
    skipping (see header.py) is pure geometry and applies either way.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    struct = TableStructure(gray, header_position=header_position)
    print(f"[1/5] Structure: {struct.summary()} | "
          f"naive header row (pre-banner-skip): {struct.header_row_idx}")

    if engine == "paddle":
        if run_targeted_ocr_paddle is None or not paddle_available():
            raise RuntimeError(
                "engine='paddle' requires the paddleocr package. Install it with: "
                "`pip install -r requirements.txt --break-system-packages`."
            )
        print("[2/5] Running PaddleOCR per cell/merge-block "
              "(first call loads models, may take a moment)...")
        run_targeted_ocr_paddle(gray, struct)
    else:
        if run_targeted_ocr is None:
            raise RuntimeError(
                "engine='tesseract' requires the pytesseract package (and the "
                "tesseract-ocr system binary). Install with: "
                "`pip install pytesseract --break-system-packages` and "
                "`sudo apt-get install -y tesseract-ocr`."
            )
        print("[2/5] Running Tesseract OCR per cell/merge-block...")
        run_targeted_ocr(gray, struct)
    counts = classify_confusion_matrix(struct)
    print(f"[2/5] OCR complete. Confusion matrix: {counts}")

    # Column names/count are no longer hardcoded -- read from the header
    # row's own OCR text (already produced by Stage 2 above) and, unless
    # use_vocab=False, normalized against the vocabulary. data_row_order
    # also corrects for the table reading bottom-up: the row closest to
    # the header is logically first, not the row at the top of the image.
    if col_names is None:
        col_names, data_row_order, banner_rows = build_header(struct, use_vocab=use_vocab)
        print(f"[2/5] Header discovered ({'vocab-normalized' if use_vocab else 'raw OCR text'}): {col_names}")
    else:
        col_names = col_names[:struct.n_cols]
        all_rows = [r for r in range(struct.n_rows) if r != struct.header_row_idx]
        data_row_order = (sorted(all_rows, reverse=True)
                           if struct.header_row_idx == struct.n_rows - 1
                           else sorted(all_rows))

    data_row_set = set(data_row_order)
    for (r, c), cell in struct.cells.items():
        if r not in data_row_set:
            continue  # header row (and any detected duplicate header row) -- not a data value
        if cell.ocr_text:
            cell.ocr_text = correct_cell(col_names[c], cell.ocr_text)
    print("[3/5] Domain-aware corrections applied.")

    export(struct, out_xlsx, col_names, row_order=data_row_order)
    print(f"[4/5] Excel saved: {out_xlsx}")

    return struct, gray


# ---------- Stage 5: crop export ----------

def export_crops(struct, gray, images_dir, manifest_path):
    os.makedirs(images_dir, exist_ok=True)
    h_img, w_img = gray.shape

    def crop_bbox(bbox, pad=3):
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w_img, x1 + pad), min(h_img, y1 + pad)
        return gray[y0:y1, x0:x1]

    rows = []
    for region in struct.merge_regions:
        crop = crop_bbox(region.bbox)
        fname = f"merge_{region.merge_id}_r{region.row_start}-{region.row_end}_c{region.col_start}-{region.col_end}.png"
        cv2.imwrite(os.path.join(images_dir, fname), crop)
        text = struct.cells[(region.row_start, region.col_start)].ocr_text
        rows.append({"filename": fname, "this_pipeline_ocr_text": text})

    for (r, c), cell in struct.cells.items():
        if cell.merge_flag:
            continue
        crop = crop_bbox(cell.bbox)
        fname = f"cell_r{r}_c{c}.png"
        cv2.imwrite(os.path.join(images_dir, fname), crop)
        rows.append({"filename": fname, "this_pipeline_ocr_text": cell.ocr_text})

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "this_pipeline_ocr_text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[5/5] Exported {len(rows)} crops to {images_dir}")
    return rows


# ---------- Orchestration ----------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # optional trailing --engine=paddle|tesseract (default: paddle),
    # --header=bottom|top (default: bottom, since these BOM sheets read
    # bottom-up), and --vocab=on|off (default: on; off shows raw OCR
    # text as column names everywhere, with no vocabulary normalization
    # or vocabulary-based header-row scoring anywhere in the run)
    flags = {"--engine", "--header", "--vocab"}
    args = [a for a in sys.argv[1:] if not any(a.startswith(f) for f in flags)]
    flag_args = [a for a in sys.argv[1:] if any(a.startswith(f) for f in flags)]
    opts = dict(a.split("=", 1) for a in flag_args)
    engine = opts.get("--engine", "paddle")
    header_position = opts.get("--header", "bottom")
    use_vocab = opts.get("--vocab", "on") != "off"

    img_path = args[0]
    out_dir = args[1] if len(args) > 1 else "output"
    os.makedirs(out_dir, exist_ok=True)

    xlsx_path = os.path.join(out_dir, "extracted.xlsx")
    struct, gray = run_extraction(img_path, xlsx_path, engine=engine,
                                   header_position=header_position, use_vocab=use_vocab)

    images_dir = os.path.join(out_dir, "crops")
    manifest_path = os.path.join(out_dir, "manifest.csv")
    export_crops(struct, gray, images_dir, manifest_path)

    print(f"\nAll outputs saved in: {out_dir}/")

if __name__ == "__main__":
    main()