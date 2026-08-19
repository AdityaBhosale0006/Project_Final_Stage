"""
Full-PDF extraction: Stage 0 table detection -> lightweight per-table
classification -> full extraction (via main.py, unmodified) -> one
Excel FILE per page, containing one worksheet per recognized table on
that page.

For every table detect_tables.py finds on EVERY page of the input PDF
(skipping anything already flagged likely_diagram_grid), this checks
ONLY the table's topmost row and its bottommost row -- never a row in
the middle, never the table's full interior -- against two known
patterns, doing the SMALLEST possible amount of OCR to decide:

  A. A full-width banner reading "PART NO.-<number>" (a parent/
     assembly part-number heading, e.g. "PART NO.-52164XXXXXX" above a
     sub-parts table). One OCR read (the whole banner merge block).
     Sheet name: the extracted number.

  B. The known pipe-BOM header schema, exactly:
        SL | PART DESCRIPTION | QTY | TML PART NO |
        PIPE CUT LENGTH(mm) | BUNCH & LOOSE | COLOUR CODING |
        REFERENCE DRG NO.
     (header_vocab.KNOWN_BOM_SCHEMA). One OCR read per column of that
     row. Sheet name: "Pipe BOM".

Whichever edge (top or bottom) a match is found at fixes that table's
reading direction ("--header direction") for full extraction.

Every table gets an `extraction` field: True (matched a known
pattern), False (matched neither), or "error" (matched a pattern but
extraction itself raised -- see below). True: the Stage 0 crop is kept
on disk, renamed to reflect what it matched, and handed to
main.run_extraction for a full OCR + extraction pass. False: the crop
is REMOVED from disk entirely -- it never goes through full OCR, and
produces no sheet. This is a strict allowlist, not a best-effort
catch-all: an unrecognized table (a different BOM layout, a title
block, a dimensioning callout table, etc.) is intentionally dropped
rather than guessed at.

Processing is grouped and saved PER PAGE, not deferred to one save at
the very end: as soon as every table on a given page has been
classified and (for accepted ones) extracted, that page's workbook is
written to <output_dir>/page{P}.xlsx immediately, before moving on to
the next page. On a large multi-page PDF this means a failure on page
7 doesn't lose the work already done and saved for pages 0-6. A single
table failing extraction (a corrupt crop, an OCR engine error, etc.)
is caught, logged with extraction="error" in the manifest, and does
NOT abort the page or the rest of the run -- every other table on
every page still gets processed. A page with zero accepted tables
produces no xlsx at all.

Usage:
    python3 extract_page.py <pdf_path> [output_dir]
        [--engine=paddle|tesseract] [--vocab=on|off]
        [--dpi=300] [--preview-dpi=150]
"""
import sys
import os
import shutil
import json
import traceback
import cv2
from collections import defaultdict
from openpyxl import Workbook

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_detection"))
from detect_tables import detect_and_render  # noqa: E402  (Stage 0)

from structure import TableStructure  # noqa: E402
from header import is_banner_row  # noqa: E402
from header_vocab import match_part_no_banner, match_column_name, KNOWN_BOM_SCHEMA  # noqa: E402
from export_xlsx import _sanitize_sheet_name  # noqa: E402
import main as extraction  # noqa: E402  (main.py's own, unmodified run_extraction)

try:
    from ocr_targeted import _ocr_crop, _preprocess_crop
except ImportError:
    _ocr_crop = _preprocess_crop = None

try:
    from ocr_paddle import _ocr_crop_paddle
except ImportError:
    _ocr_crop_paddle = None


def _read_crop(gray, bbox, engine, pad=2):
    """One single-region OCR read, bypassing the full per-table
    pipeline entirely -- the primitive both engines' full pipelines
    already use per cell/merge-block internally, called directly here
    for a lightweight classification-only read."""
    h, w = gray.shape
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return ""
    crop = gray[y0:y1, x0:x1]
    if engine == "paddle":
        if _ocr_crop_paddle is None:
            raise RuntimeError("engine='paddle' requires the paddleocr package.")
        return _ocr_crop_paddle(crop)
    if _ocr_crop is None or _preprocess_crop is None:
        raise RuntimeError("engine='tesseract' requires pytesseract + tesseract-ocr.")
    return _ocr_crop(_preprocess_crop(crop))


def classify_table(gray, struct, engine):
    """
    Returns (accepted, header_position, sheet_name, match_kind).

    Reads ONLY row 0 (top) and row n_rows-1 (bottom) -- a match
    anywhere else doesn't count, same "header sits at one end, never
    the middle" principle used throughout header.py. Deliberately does
    NOT write into struct.cells and never reuses this struct for
    extraction: a table that gets accepted is handed off to
    main.run_extraction, which reads the (renamed) crop fresh and
    re-derives every cell -- including these same edge-row cells --
    via its own correct, unmodified per-block/majority-vote pipeline.
    This function exists purely to decide accept/reject as cheaply as
    possible, using the smallest number of OCR calls that can do it.
    """
    if struct.n_rows == 0 or struct.n_cols == 0:
        return False, None, None, None

    for position, r in [("top", 0), ("bottom", struct.n_rows - 1)]:
        if is_banner_row(struct, r):
            # one read: the whole full-width merge block for this row
            region = next(m for m in struct.merge_regions
                          if m.row_start == r and m.row_end == r
                          and m.col_start == 0 and m.col_end == struct.n_cols - 1)
            text = _read_crop(gray, region.bbox, engine)
            part_no = match_part_no_banner(text)
            print(f"    [debug] {position} row is a full-width banner, OCR read: {text!r} "
                  f"-> part_no={part_no!r}")
            if part_no:
                return True, position, part_no, "banner"
            continue  # a full-width row can't also be a distinct-column header

        row_names = [match_column_name(_read_crop(gray, struct.cells[(r, c)].bbox, engine),
                                        use_vocab=True)
                     for c in range(struct.n_cols)]
        if len(row_names) == len(KNOWN_BOM_SCHEMA):
            matches = sum(a == b for a, b in zip(row_names, KNOWN_BOM_SCHEMA))
            ratio = matches / len(KNOWN_BOM_SCHEMA)
            print(f"    [debug] {position} row schema check: {matches}/{len(KNOWN_BOM_SCHEMA)} "
                  f"({ratio:.0%}) -> {row_names}")
            # >=75% positional agreement, not exact equality -- an 8-cell
            # OCR pass on a real (sometimes small/blurry) header row only
            # needs ONE cell to misread past the fuzzy-match threshold to
            # fail strict equality, discarding an otherwise perfectly
            # good, correctly-structured table. Confirmed on a real BOM
            # table: structure detection correctly found all 8 columns,
            # yet strict equality rejected it anyway. Requiring most --
            # not all -- positions to match keeps this just as precise
            # (an unrelated table matching 6+ of 8 positions against
            # this specific vocabulary, in this specific order, by
            # coincidence is not realistic) while tolerating a cell or
            # two of genuine OCR noise.
            if ratio >= 0.75:
                return True, position, "Pipe BOM", "schema"
        else:
            print(f"    [debug] {position} row: n_cols={len(row_names)} != "
                  f"{len(KNOWN_BOM_SCHEMA)} known columns -> {row_names}")

    return False, None, None, None


def _renamed_crop_path(entry, sheet_name, match_kind):
    """Accepted-crop naming: keeps the original page/table index for
    traceability, adds extraction status + what it matched."""
    safe_name = _sanitize_sheet_name(sheet_name, existing=[]).replace(" ", "_")
    new_fname = f"page{entry['page']}_table{entry['table_index']}_extract-true_{match_kind}_{safe_name}.png"
    return os.path.join("tables", new_fname)


def _classify_and_prepare(entry, detect_dir, engine):
    """
    Stages 1-3 for one detected table: read crop, classify off its top/
    bottom row, and (if accepted) rename the crop on disk / (if
    rejected or unreadable) remove it. Mutates `entry` in place with
    extraction/header_position/match_kind/sheet_name. Returns the new
    crop path for an accepted table, or None.

    Every detected table reaches classify_table() below, REGARDLESS of
    Stage 0's own content_type flag (likely_data_table /
    likely_diagram_grid / uncertain). That flag is explicitly a QA
    hint for a human reviewer (see detect_tables.py's own docstring),
    not a hard filter -- treating it as one was confirmed to silently
    drop a real, valid table: on a page that's one big scanned image,
    EVERY table's bbox trivially "intersects" that page-covering
    image, so the classifier's embedded-image signal fires as a false
    positive for every table regardless of content, and a couple points
    of row-height noise either side of its 30pt cutoff was the entire
    difference between one real table surviving and an identical
    neighboring one being discarded before ever reaching our own,
    far more precise banner/schema check. content_type is still
    recorded on the entry for visibility, just never used to skip.
    """
    label = f"page{entry['page']}_table{entry['table_index']}"
    old_crop_path = os.path.join(detect_dir, entry["crop_filename"])

    img = cv2.imread(old_crop_path)
    if img is None:
        entry["extraction"] = False
        print(f"  {label}: extraction=False (couldn't read crop)")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        struct = TableStructure(gray, header_position="top")  # arbitrary; see classify_table
        accepted, header_position, sheet_name, match_kind = classify_table(gray, struct, engine)
    except Exception as e:
        # a crop that isn't cleanly gridded enough for structure.py to
        # even read its top/bottom row (e.g. ValueError: "Could not
        # detect column boundaries...") must be treated as a rejected
        # table, not a fatal error -- one ungridded false-positive
        # detection from Stage 0 must not abort the rest of the page/run
        entry["extraction"] = "error"
        entry["error"] = f"{type(e).__name__}: {e}"
        print(f"  {label}: extraction=error (classification failed: {type(e).__name__}: {e})")
        if os.path.exists(old_crop_path):
            os.remove(old_crop_path)
        return None

    entry["extraction"] = accepted
    entry["header_position"] = header_position
    entry["match_kind"] = match_kind
    entry["sheet_name"] = sheet_name

    if not accepted:
        print(f"  {label}: extraction=False (matches neither known table pattern)")
        if os.path.exists(old_crop_path):
            os.remove(old_crop_path)
        return None

    new_rel_path = _renamed_crop_path(entry, sheet_name, match_kind)
    new_crop_path = os.path.join(detect_dir, new_rel_path)
    if os.path.exists(old_crop_path):
        shutil.move(old_crop_path, new_crop_path)
    entry["crop_filename"] = new_rel_path
    print(f"  {label}: extraction=True ({match_kind} match, header at {header_position}) "
          f"-> renamed to {os.path.basename(new_crop_path)}")
    return new_crop_path


def extract_pdf(pdf_path, output_dir, engine="paddle", use_vocab=True, dpi=300, preview_dpi=150):
    os.makedirs(output_dir, exist_ok=True)
    detect_dir = os.path.join(output_dir, "detected_tables")

    print(f"[Stage 0] Detecting tables across all pages of {pdf_path} "
          f"(dpi={dpi}, preview_dpi={preview_dpi}) ...")
    manifest = detect_and_render(pdf_path, detect_dir, dpi=dpi, preview_dpi=preview_dpi)

    # group by page so each page can be fully classified, extracted,
    # AND saved before moving on -- a crash partway through a large
    # multi-page PDF doesn't lose already-completed earlier pages
    entries_by_page = defaultdict(list)
    for entry in manifest:
        entries_by_page[entry["page"]].append(entry)

    saved = []
    total_accepted = 0
    total_errors = 0

    for page_idx in sorted(entries_by_page):
        page_entries = entries_by_page[page_idx]
        print(f"\n[Page {page_idx}] Classifying {len(page_entries)} detected table(s) "
              f"(top/bottom row only)...")

        wb = None
        page_accept_count = 0

        for entry in page_entries:
            new_crop_path = _classify_and_prepare(entry, detect_dir, engine)
            if new_crop_path is None:
                if entry.get("extraction") == "error":
                    total_errors += 1  # classification itself raised -- count it too
                continue  # rejected, or unreadable -- already logged

            label = f"page{entry['page']}_table{entry['table_index']}"
            try:
                wb = wb or Workbook()
                struct, gray, wb = extraction.run_extraction(
                    new_crop_path, engine=engine,
                    header_position=entry["header_position"],
                    use_vocab=use_vocab, workbook=wb,
                    sheet_name=entry["sheet_name"])
                page_accept_count += 1
                total_accepted += 1
            except Exception as e:
                # one bad table (corrupt crop, OCR engine failure, etc.)
                # must not take down the rest of this page or the run
                entry["extraction"] = "error"
                entry["error"] = f"{type(e).__name__}: {e}"
                total_errors += 1
                print(f"  {label}: extraction=error ({type(e).__name__}: {e})")
                traceback.print_exc()

        if wb is not None and page_accept_count > 0:
            out_xlsx = os.path.join(output_dir, f"page{page_idx}.xlsx")
            wb.save(out_xlsx)
            saved.append(out_xlsx)
            print(f"[Page {page_idx}] Saved {out_xlsx}: {page_accept_count} table(s), "
                  f"sheets: {wb.sheetnames}")
        else:
            print(f"[Page {page_idx}] No tables matched either known pattern -- no xlsx saved.")

    # refresh the manifest on disk with extraction/header_position/etc.
    # and the renamed/removed crop filenames, plus any per-table errors
    with open(os.path.join(detect_dir, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{len(saved)} page workbook(s) saved in {output_dir}/ "
          f"({total_accepted} table(s) extracted, {total_errors} error(s))")
    return saved


def extract_single_table(img_path, output_dir, engine="paddle", use_vocab=True):
    """
    For an image that's ALREADY a single cropped table (e.g. Stage 0's
    own output, or any other pre-cropped table image) -- skips
    detect_and_render entirely and reads the image at its own native
    pixel scale, exactly like main.py does.

    This exists because running such an image back through
    detect_and_render (as extract_pdf does) re-renders it via
    PyMuPDF's PDF-point-space pipeline. A standalone raster image has
    no DPI metadata, so PyMuPDF treats 1 pixel = 1 point -- requesting
    the usual dpi=300 there means rendering at 300/72 ~ 4.17x the
    crop's real size. That resolution mismatch was confirmed to
    corrupt structure.py's wall/merge detection (thresholds tuned for
    normal scale misfire at 4x blow-up), producing wildly wrong merge
    regions and multiple real rows' text getting OCR'd together into
    one cell -- a real failure seen on an actual re-run of a Stage 0
    crop through the full-PDF path. Reading the image directly here,
    with no rescaling step at all, avoids that failure mode entirely.
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    struct = TableStructure(gray, header_position="top")  # arbitrary; see classify_table
    print(f"Structure: {struct.summary()}")

    accepted, header_position, sheet_name, match_kind = classify_table(gray, struct, engine)
    if not accepted:
        print("extraction=False (matches neither known table pattern) -- nothing to export.")
        return None
    print(f"extraction=True ({match_kind} match, header at {header_position})")

    out_xlsx = os.path.join(output_dir, "extracted.xlsx")
    struct, gray, result = extraction.run_extraction(
        img_path, out_xlsx=out_xlsx, engine=engine,
        header_position=header_position, use_vocab=use_vocab, sheet_name=sheet_name)
    print(f"\nSaved: {out_xlsx} (sheet '{sheet_name}')")
    return out_xlsx


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    flags = {"--engine", "--vocab", "--precropped", "--dpi", "--preview-dpi"}
    args = [a for a in sys.argv[1:] if not any(a.startswith(f) for f in flags)]
    flag_args = [a for a in sys.argv[1:] if any(a.startswith(f) for f in flags)]
    opts = dict(a.split("=", 1) for a in flag_args)
    engine = opts.get("--engine", "paddle")
    use_vocab = opts.get("--vocab", "on") != "off"
    precropped = opts.get("--precropped", "false") == "true"
    dpi = int(opts.get("--dpi", 300))
    preview_dpi = int(opts.get("--preview-dpi", 150))

    input_path = args[0]
    output_dir = args[1] if len(args) > 1 else "extract_pdf_output"

    if precropped:
        extract_single_table(input_path, output_dir, engine=engine, use_vocab=use_vocab)
    else:
        extract_pdf(input_path, output_dir, engine=engine, use_vocab=use_vocab,
                    dpi=dpi, preview_dpi=preview_dpi)


if __name__ == "__main__":
    main()