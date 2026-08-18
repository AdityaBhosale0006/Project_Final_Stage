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
    # length value. Matching digit groups CHAINED by a single separator
    # keeps a genuine decimal/thousands number intact as one token,
    # while still falling back to plain digit runs when there's no
    # separator at all. This is still purely structural (no locale
    # assumption about which separator means what) -- it only ever
    # preserves whatever punctuation OCR already read between two digit
    # groups, never invents or reinterprets it.
    runs = re.findall(r'\d+(?:[.,]\d+)*', corrected)
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