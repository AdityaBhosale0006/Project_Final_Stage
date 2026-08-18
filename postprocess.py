"""
Post-processing / correction layer.

No domain word vocabulary here (no lists of known part-description
words like "PIPE"/"ASSY", no known-colour lists, etc.) -- correction is
purely structural/typographic: fixing characters that are visually
confusable with digits, and only where the cell already looks numeric.
A cell that isn't mostly digits is left exactly as OCR read it, since
"is this a valid word" can't be judged without a word list, and this
pipeline intentionally doesn't hardcode one.
"""
import re

# Common OCR digit/letter confusions seen in tesseract output on this
# font, based on real samples captured from the pipeline's own output.
# This is a character-level typographic table (which glyphs look like
# which), not a word/domain vocabulary -- it applies the same regardless
# of what column or document this is.
DIGIT_CONFUSION = {
    'O': '0', 'o': '0', 'Q': '0',
    'l': '1', 'I': '1', '|': '1',
    'S': '5', 's': '5',
    'B': '8',
    'Z': '2',
    'G': '6',
    'T': '7',
}


def _numeric_runs_preserving_decimals(orig, corrected):
    """
    Extracts digit runs from `corrected` (already DIGIT_CONFUSION-mapped),
    chaining two runs across a single '.' or ',' separator into ONE
    continuous number ONLY when the characters immediately either side
    of that separator were ALREADY digits in the ORIGINAL, unmapped
    text -- not merely digit-shaped after confusion-mapping a letter.

    This is what actually distinguishes a genuine decimal like "2.630"
    (both neighbors of the '.' were real digits already) from a
    truncated OCR read like "NO.517443900104" (the 'O' in "NO" maps to
    '0' via DIGIT_CONFUSION, but it was a LETTER, not a digit -- so the
    '.' there is an abbreviation period, not a decimal point). Chaining
    blindly on the confusion-mapped string alone can't tell these
    apart, and was confirmed gluing a converted stray letter onto an
    unrelated following number on a real extracted "REFERENCE DRG NO."
    cell. `orig` and `corrected` are always the same length and
    index-aligned, since DIGIT_CONFUSION only ever maps one character
    to exactly one character.
    """
    runs = []
    i, n = 0, len(corrected)
    while i < n:
        if not corrected[i].isdigit():
            i += 1
            continue
        start = i
        while i < n and corrected[i].isdigit():
            i += 1
        while (i < n and corrected[i] in '.,' and i + 1 < n
               and corrected[i + 1].isdigit()
               and orig[i - 1].isdigit() and orig[i + 1].isdigit()):
            i += 1  # consume the separator
            while i < n and corrected[i].isdigit():
                i += 1
        runs.append(corrected[start:i])
    return runs


def _normalize_if_numeric(text, min_digit_fraction=0.5):
    """
    Generic structural cleanup: if a cell is already mostly digits
    (after fixing common OCR letter/digit look-alikes), collapse it down
    to its longest digit run. If it's mostly letters/words, it's left
    completely untouched -- correcting free text requires knowing what
    the "right" word should be, which needs a word vocabulary, and this
    pipeline doesn't assume one.
    """
    if not text or not text.strip():
        return text

    # Classification gate uses the RAW text's own digit fraction -- letters
    # are not pre-mapped to digits here, or a free-text banner like
    # "REFER DRAWING NO.517443900141" gets misjudged as numeric just
    # because "DRAWING"/"NO" happen to contain I/G/O look-alikes.
    alnum_count = sum(ch.isalnum() for ch in text)
    raw_digit_count = sum(ch.isdigit() for ch in text)
    if alnum_count == 0 or (raw_digit_count / alnum_count) < min_digit_fraction:
        return text  # not numeric-looking -- leave as-is, don't guess

    # Only once a cell has already qualified as numeric do we apply the
    # confusion table, to clean up misread digits within it.
    corrected = ''.join(DIGIT_CONFUSION.get(ch, ch) for ch in text)

    # A pure \d+ run treats '.' and ',' as hard breaks, so a decimal or
    # thousands-separated value like "2.630" gets shattered into ['2',
    # '630'] and the longer-but-wrong fragment wins, silently dropping
    # the "2." -- confirmed as a real failure on an actual extracted
    # length value. Chaining digit groups across a single separator
    # keeps a genuine decimal/thousands number intact as one token --
    # but only when that separator's neighbors were ALREADY digits
    # before confusion-mapping (see _numeric_runs_preserving_decimals),
    # so a truncated label fragment like "NO.517443900104" (a
    # confusion-mapped 'O' next to an abbreviation period) doesn't get
    # glued into a fake "0.517443900104" the same way -- confirmed as a
    # real failure on an actual extracted REFERENCE DRG NO. value.
    runs = _numeric_runs_preserving_decimals(text, corrected)
    if not runs:
        return text
    return max(runs, key=len)


def correct_cell(column_name, text):
    """
    Applies the one generic, vocabulary-free correction this pipeline
    makes: normalizing cells that are already numeric-looking. Anything
    text-like (descriptions, colour names, drawing banners, single
    letters, etc.) passes through exactly as OCR read it -- no snapping
    to a fixed word list. column_name is accepted for API compatibility
    (callers pass whatever header.py discovered) but isn't required to
    decide the correction; it's purely content-shape based.
    """
    return _normalize_if_numeric(text)