"""
Full-page extraction: Stage 0 table detection -> per-table
recognition -> extraction -> one shared Excel workbook.

For every table detect_tables.py finds on a page (skipping anything
already flagged likely_diagram_grid), this checks ONLY the table's
topmost row and its bottommost row -- never a row in the middle --
against two known patterns:

  A. A full-width banner reading "PART NO.-<number>" (a parent/
     assembly part-number heading, e.g. "PART NO.-52164XXXXXX" above a
     sub-parts table). Sheet name: the extracted number.

  B. The known pipe-BOM header schema, exactly:
        SL | PART DESCRIPTION | QTY | TML PART NO |
        PIPE CUT LENGTH(mm) | BUNCH & LOOSE | COLOUR CODING |
        REFERENCE DRG NO.
     (header_vocab.KNOWN_BOM_SCHEMA). Sheet name: "Pipe BOM".

Whichever edge (top or bottom) a match is found at fixes that table's
reading direction for extraction, same convention as header.py/
main.py elsewhere.

A table matching NEITHER pattern is skipped entirely -- not extracted,
no sheet created. This is a strict allowlist, not a best-effort
catch-all: an unrecognized table (a different BOM layout, a title
block, a dimensioning callout table, etc.) is intentionally left out
rather than guessed at.

Usage:
    python3 extract_page.py <pdf_or_image_path> [output_dir]
        [--engine=paddle|tesseract] [--vocab=on|off]
"""
import sys
import os
import cv2
from openpyxl import Workbook

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_detection"))
from detect_tables import detect_and_render  # noqa: E402  (Stage 0)

from structure import TableStructure  # noqa: E402
from ocr_mapping import classify_confusion_matrix  # noqa: E402
from postprocess import correct_cell  # noqa: E402
from export_xlsx import export  # noqa: E402
from header import build_header, is_banner_row  # noqa: E402
from header_vocab import match_part_no_banner, match_column_name, KNOWN_BOM_SCHEMA  # noqa: E402

try:
    from ocr_targeted import run_targeted_ocr
except ImportError:
    run_targeted_ocr = None

try:
    from ocr_paddle import run_targeted_ocr_paddle, paddle_available
except ImportError:
    run_targeted_ocr_paddle = None
    paddle_available = lambda: False


def _run_ocr(gray, struct, engine):
    if engine == "paddle":
        if run_targeted_ocr_paddle is None or not paddle_available():
            raise RuntimeError(
                "engine='paddle' requires the paddleocr package. Install it with: "
                "`pip install -r requirements.txt --break-system-packages`."
            )
        run_targeted_ocr_paddle(gray, struct)
    else:
        if run_targeted_ocr is None:
            raise RuntimeError(
                "engine='tesseract' requires the pytesseract package (and the "
                "tesseract-ocr system binary). Install with: "
                "`pip install pytesseract --break-system-packages` and "
                "`sudo apt-get install -y tesseract-ocr`."
            )
        run_targeted_ocr(gray, struct)


def classify_table(struct):
    """
    Returns (accepted, header_position, sheet_name, match_kind).

    Checks ONLY row 0 (top) and row n_rows-1 (bottom) -- a match
    anywhere else doesn't count, same "header sits at one end, never
    the middle" principle used throughout header.py. Pattern A is
    checked first at both edges, then pattern B at both edges, so a
    table matching neither returns (False, None, None, None).
    """
    if struct.n_rows == 0 or struct.n_cols == 0:
        return False, None, None, None

    edges = [("top", 0), ("bottom", struct.n_rows - 1)]

    # Pattern A: full-width "PART NO.-<number>" banner at either edge
    for position, r in edges:
        if not is_banner_row(struct, r):
            continue
        part_no = match_part_no_banner(struct.cells[(r, 0)].ocr_text)
        if part_no:
            return True, position, part_no, "banner"

    # Pattern B: the known pipe-BOM schema, exactly, at either edge
    for position, r in edges:
        row_names = [match_column_name(struct.cells[(r, c)].ocr_text, use_vocab=True)
                     for c in range(struct.n_cols)]
        if row_names == KNOWN_BOM_SCHEMA:
            return True, position, "Pipe BOM", "schema"

    return False, None, None, None


def extract_page(input_path, output_dir, engine="paddle", use_vocab=True):
    os.makedirs(output_dir, exist_ok=True)
    detect_dir = os.path.join(output_dir, "detected_tables")

    print(f"[Stage 0] Detecting tables in {input_path} ...")
    manifest = detect_and_render(input_path, detect_dir)

    wb = Workbook()
    accepted_count = 0
    last_sheet_name = None

    for entry in manifest:
        label = f"page{entry['page']}_table{entry['table_index']}"

        if entry["content_type"] == "likely_diagram_grid":
            print(f"  {label}: skipped (flagged as diagram, not a data table)")
            continue

        crop_path = os.path.join(detect_dir, entry["crop_filename"])
        img = cv2.imread(crop_path)
        if img is None:
            print(f"  {label}: skipped (couldn't read crop)")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # header_position is arbitrary here -- it only feeds a
        # last-resort fallback inside header.py's _detect_header_row
        # that a table strong enough to pass classify_table() below
        # never actually reaches (its matched edge already scores far
        # above the vocabulary threshold on its own).
        struct = TableStructure(gray, header_position="top")
        _run_ocr(gray, struct, engine)
        classify_confusion_matrix(struct)

        accepted, header_position, sheet_name, match_kind = classify_table(struct)
        if not accepted:
            print(f"  {label}: skipped (matches neither known table pattern)")
            continue

        # align header_row_idx with the edge the match was actually
        # found at, so build_header's bottom-up/top-down convention
        # (and its own internal fallback, if ever reached) agrees with
        # what classify_table already determined
        struct.header_row_idx = struct.n_rows - 1 if header_position == "bottom" else 0

        col_names, data_row_order, banner_rows = build_header(struct, use_vocab=use_vocab)

        data_row_set = set(data_row_order)
        for (r, c), cell in struct.cells.items():
            if r not in data_row_set:
                continue
            if cell.ocr_text:
                cell.ocr_text = correct_cell(col_names[c], cell.ocr_text)

        wb = export(struct, col_names=col_names, row_order=data_row_order,
                    workbook=wb, sheet_name=sheet_name)
        last_sheet_name = wb.sheetnames[-1]
        accepted_count += 1
        print(f"  {label}: ACCEPTED ({match_kind} match, header at {header_position}) "
              f"-> sheet '{last_sheet_name}' ({len(data_row_order)} data rows)")

    if accepted_count == 0:
        print("\nNo tables matched either known pattern -- nothing to export.")
        return None

    out_xlsx = os.path.join(output_dir, "extracted.xlsx")
    wb.save(out_xlsx)
    print(f"\n{accepted_count} table(s) extracted into {len(wb.sheetnames)} sheet(s): {out_xlsx}")
    return out_xlsx


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    flags = {"--engine", "--vocab"}
    args = [a for a in sys.argv[1:] if not any(a.startswith(f) for f in flags)]
    flag_args = [a for a in sys.argv[1:] if any(a.startswith(f) for f in flags)]
    opts = dict(a.split("=", 1) for a in flag_args)
    engine = opts.get("--engine", "paddle")
    use_vocab = opts.get("--vocab", "on") != "off"

    input_path = args[0]
    output_dir = args[1] if len(args) > 1 else "extract_page_output"
    extract_page(input_path, output_dir, engine=engine, use_vocab=use_vocab)


if __name__ == "__main__":
    main()
