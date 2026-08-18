"""
Header discovery.

Turns the OCR'd header row into an ordered list of column names and the
correct top-to-bottom reading order for the data rows -- instead of
assuming a fixed, hardcoded schema.

Which grid row IS the header is no longer a static assumption passed in
from outside (structure.py's header_position flag, decided before any
OCR has even run). It's determined here, from the OCR'd content of the
grid's two extremal rows (top and bottom), using two independent
signals:

  1. Vocabulary match -- does this row's text match known column-name
     concepts (header_vocab.COLUMN_ALIASES)? Works for this table
     family and any other sheet whose columns are already listed there.
  2. Row-to-row similarity -- do the top and bottom rows read as
     near-duplicates of EACH OTHER, regardless of what they say? This
     is vocabulary-free: it catches a sheet that prints its header at
     both the top and the bottom (a common BOM/engineering-drawing
     layout) even for a completely unfamiliar table whose column names
     aren't in the vocabulary at all.

If a duplicate header is detected this way, BOTH copies are excluded
from the data rows -- only one supplies the column names, but neither
leaks through as a bogus data row. If neither signal is confident (an
unfamiliar table, badly garbled OCR on both ends, etc.), this falls
back to structure.py's original static assumption rather than guessing
wrong from weak evidence.

Stage 2 (ocr_paddle.py / ocr_targeted.py) already OCR's every cell in
the grid, including these rows -- this stage does not run any
additional OCR. It only reads what's already there and interprets it.
"""
from header_vocab import match_column_name, header_likeness_score, row_similarity


def is_banner_row(struct, row_idx):
    """Public wrapper around _is_full_width_merge_row, for callers
    outside this module (e.g. a table-recognition classifier) that
    need to check a specific row without running full header
    detection."""
    return _is_full_width_merge_row(struct, row_idx)


def _row_texts(struct, row_idx):
    return [struct.cells[(row_idx, c)].ocr_text for c in range(struct.n_cols)]


def _is_full_width_merge_row(struct, row_idx):
    """
    True if `row_idx` is entirely ONE merge block spanning every column
    (col_start=0 to col_end=n_cols-1) with no internal vertical
    dividers -- e.g. a "PART NO.-52164..." banner sitting above the
    real header row. This is pure geometry (struct.merge_regions), not
    OCR content, so it's reliable even before/regardless of what the
    OCR'd text says.
    """
    if struct.n_cols <= 1:
        return False
    for m in struct.merge_regions:
        if (m.row_start == row_idx and m.row_end == row_idx
                and m.col_start == 0 and m.col_end == struct.n_cols - 1):
            return True
    return False


def _leading_banner_rows(struct):
    """
    Contiguous full-width merge rows starting at the very top of the
    grid (row 0, then 1, then 2, ...) -- e.g. a parent part-number
    heading above the real header. Only rows anchored to the top edge
    count: a full-width merge in the MIDDLE of the data (a genuine
    "REFER DRAWING NO." style note row) is real data, not a banner,
    and must never be swept up by this.
    """
    rows = []
    r = 0
    while r < struct.n_rows and _is_full_width_merge_row(struct, r):
        rows.append(r)
        r += 1
    return rows


def _trailing_banner_rows(struct):
    """Same idea, anchored to the bottom edge instead (e.g. a full-width
    footer note below the last data row)."""
    rows = []
    r = struct.n_rows - 1
    while r >= 0 and _is_full_width_merge_row(struct, r):
        rows.append(r)
        r -= 1
    return rows


def _names_from_row(struct, row_idx, use_vocab=True):
    col_names = []
    seen = {}
    for c, raw in enumerate(_row_texts(struct, row_idx)):
        name = match_column_name(raw, use_vocab=use_vocab) or f"Col{c + 1}"
        # keep column names unique (e.g. two blank/unreadable header
        # cells shouldn't collapse into one column downstream)
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        col_names.append(name)
    return col_names


def _detect_header_row(struct, vocab_threshold=0.5, duplicate_threshold=0.6, use_vocab=True):
    """
    Returns (header_row, duplicate_row_or_None, banner_rows).

    Only ever considers the grid's two EFFECTIVE extremal rows as header
    candidates -- a header, wherever it lives, sits at one end of the
    table, never in the middle. "Effective" means after skipping past
    any leading/trailing banner rows (see _leading_banner_rows):
    a full-width merge row like "PART NO.-52164..." isn't itself a
    header candidate, it just displaces where the true top/bottom edge
    of header-search actually starts. This banner-skip is pure geometry
    and applies regardless of use_vocab.

    use_vocab=False disables the vocabulary-likeness signal entirely
    (header_likeness_score is never called) -- header selection then
    relies only on the vocabulary-free row-similarity signal (duplicate
    header at both ends) and, failing that, falls back to whichever
    effective edge matches struct.header_position. Row-similarity
    itself needs no vocabulary: it only compares the two candidate
    rows' texts to EACH OTHER, not against any known column list.
    """
    banner_rows = set(_leading_banner_rows(struct)) | set(_trailing_banner_rows(struct))
    non_banner_rows = [r for r in range(struct.n_rows) if r not in banner_rows]

    if not non_banner_rows:
        # degenerate: every row read as a full-width merge (shouldn't
        # normally happen -- a real header always has internal column
        # dividers). Fall back to the raw extremes rather than erroring.
        top_row, bottom_row = 0, struct.n_rows - 1
        banner_rows = set()
    else:
        top_row, bottom_row = min(non_banner_rows), max(non_banner_rows)

    top_texts = _row_texts(struct, top_row)
    bottom_texts = _row_texts(struct, bottom_row)

    top_vocab = header_likeness_score(top_texts) if use_vocab else 0.0
    bottom_vocab = header_likeness_score(bottom_texts) if use_vocab else 0.0
    dup_score = row_similarity(top_texts, bottom_texts)

    if top_row != bottom_row and dup_score >= duplicate_threshold:
        # Both ends read as the same content -- a duplicated header.
        # Use whichever copy the vocabulary recognizes better (or the
        # bottom one, this table family's usual convention, as a
        # tie-break) to actually name the columns, and drop both from
        # the data.
        header_row = bottom_row if bottom_vocab >= top_vocab else top_row
        dup_row = top_row if header_row == bottom_row else bottom_row
        return header_row, dup_row, banner_rows

    if use_vocab and (top_vocab >= vocab_threshold or bottom_vocab >= vocab_threshold):
        header_row = top_row if top_vocab > bottom_vocab else bottom_row
        return header_row, None, banner_rows

    # Neither end reads confidently as a header (or use_vocab=False, so
    # this signal was never consulted) -- fall back to whichever
    # EFFECTIVE edge (post-banner-skip) matches struct.header_position,
    # not the raw static struct.header_row_idx, so the banner fix still
    # applies even with vocabulary disabled.
    header_row = top_row if struct.header_position == "top" else bottom_row
    return header_row, None, banner_rows


def build_header(struct, use_vocab=True):
    """
    Returns (col_names, data_row_order, banner_rows).

    col_names: one name per logical column (length == struct.n_cols).
    When use_vocab=True (default), names are taken from the detected
    header row's own OCR text and normalized against the known column
    vocabulary. When use_vocab=False, normalization is skipped
    everywhere -- col_names is exactly the cleaned raw OCR text read
    off the header row (or a positional "ColN" if that cell was
    blank), with no snapping to a known canonical name. Either way the
    number and identity of columns is discovered from the sheet, not
    assumed.

    data_row_order: struct row indices to treat as DATA, already in
    correct reading order, with the header row, any detected duplicate
    header row at the opposite end, and any leading/trailing full-width
    banner row (e.g. a "PART NO.-52164..." heading with no column
    structure of its own) all excluded. Because a header at the bottom
    of the image means the row directly above it is logically the
    first data row, that case reverses the image's own top-to-bottom
    row order.

    banner_rows: struct row indices identified as full-width banner
    rows (excluded from both header and data above). Returned rather
    than only used internally so a caller can still do something with
    that text -- e.g. recognizing/naming a table from its banner
    (header_vocab.match_part_no_banner) -- without it having to leak
    into the tabular data itself.
    """
    header_row, duplicate_row, banner_rows = _detect_header_row(struct, use_vocab=use_vocab)
    col_names = _names_from_row(struct, header_row, use_vocab=use_vocab)

    exclude = {header_row} | banner_rows
    if duplicate_row is not None:
        exclude.add(duplicate_row)

    all_rows = [r for r in range(struct.n_rows) if r not in exclude]
    if header_row == struct.n_rows - 1:
        # header at the bottom -> image reads bottom-to-top
        data_row_order = sorted(all_rows, reverse=True)
    else:
        # header at the top (possibly after a leading banner) -> image
        # already reads top-to-bottom
        data_row_order = sorted(all_rows)

    return col_names, data_row_order, banner_rows