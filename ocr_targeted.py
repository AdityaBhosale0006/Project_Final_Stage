"""
Enhanced Stage 2: instead of running OCR once over the whole page and
mapping words back by coordinate, OCR is run PER LEARNED BLOCK (each
merge region or standalone cell gets its own crop). This lets small
cells get upscaled much more aggressively and uses a single-line PSM
mode suited to short BOM values, which a single whole-page pass can't
do as effectively.

Structure is still learned completely independently beforehand (Stage 1
untouched) -- this only changes HOW Stage 2 gathers text, not how
structure/merges are determined.
"""
import cv2
import numpy as np
import pytesseract
from structure import TableStructure


def _preprocess_crop(gray_crop, target_height=48):
    h, w = gray_crop.shape
    if h == 0 or w == 0:
        return None
    scale = max(1.0, target_height / h)
    scale = min(scale, 8.0)
    big = cv2.resize(gray_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(big, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(big, 1.5, blur, -0.5, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(sharp)
    _, bin_crop = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_crop


def _ocr_crop(bin_crop, multiline=False):
    if bin_crop is None:
        return ""
    psm = 6 if multiline else 7
    txt = pytesseract.image_to_string(bin_crop, config=f'--psm {psm}').strip()
    if not txt and not multiline:
        txt = pytesseract.image_to_string(bin_crop, config='--psm 6').strip()
    return txt.replace('\n', ' ').strip()


import difflib


def _is_substantial(text):
    """Filters out tiny OCR noise fragments (stray '|', '.', single chars)
    that are common on blank scan regions and shouldn't count as 'real
    content' when deciding whether rows genuinely hold distinct values."""
    cleaned = ''.join(ch for ch in text if ch.isalnum())
    return len(cleaned) >= 3


import re


def _trailing_number(text):
    """Extracts the last contiguous digit run from a string, e.g.
    'NEFER teaving 0:5201c3910210' -> '210' (or '5201' if no later run --
    we want the LAST digit group, since that's where sequential BOM/drawing
    numbers typically vary)."""
    matches = re.findall(r'\d+', text)
    return matches[-1] if matches else None


def _rows_are_actually_same_value(row_texts):
    """
    A missing dividing line only proves the rows share column boundaries --
    it does NOT prove they share the same value, and it does NOT prove
    they don't. Several situations need to be told apart:
      1. A single merged value whose text sits vertically centered in the
         block -- only one row shows real text, the rest are blank. TRUE
         merge; propagate the one value.
      2. That same situation, but the centered text line happens to fall
         right on a row-slice boundary -- TWO adjacent rows each capture
         a different PARTIAL FRAGMENT of the one real value (not two real
         values). With only 1-2 conflicting reads, this is far more
         likely than a genuine coincidence of exactly 2 distinct values
         appearing with no dividing line between them. TRUE merge.
      3. Genuinely distinct sequential values (e.g. ...210, ...211, ...212,
         ...213, ...214) where MANY rows each hold their own real,
         differing text -- the defining signature of true per-row data is
         that MOST rows in the block are independently populated, not
         just one or two. Only this pattern should count as "NOT a merge".
      4. A single TRUE shared value where OCR noise misreads a digit on
         a minority of rows -- majority vote overrules the noisy minority.

    Practically: we only trust a "distinct values" conclusion when there
    are at least 3 substantial (non-noise) rows to compare -- below that,
    2 conflicting reads are treated as fragments of one value, not two
    real ones, and we default to merge (using the whole-block OCR read,
    not any single noisy per-row fragment).
    """
    substantial = [t for t in row_texts if _is_substantial(t)]
    if len(substantial) <= 2:
        return True  # too little signal to conclude real distinct values;
                      # likely one value fragmented across row slices

    # The docstring's own rule: genuine per-row data means MOST rows in
    # the block are independently populated -- not just "more than 2 in
    # absolute terms". A handful of substantial fragments out of a much
    # larger block (e.g. 3 of 7 rows) is the signature of ONE value
    # sliced into pieces by row boundaries, with the rest of the block
    # genuinely blank -- not a block of 7 real distinct values. Require a
    # real majority of the block's own rows to carry independent text
    # before trusting a "distinct values" conclusion at all.
    if len(substantial) / len(row_texts) <= 0.5:
        return True  # too sparse relative to the block to be genuine per-row data

    numbers = [_trailing_number(t) for t in substantial]
    if all(n is not None for n in numbers):
        from collections import Counter
        counts = Counter(numbers)
        most_common_num, most_common_count = counts.most_common(1)[0]
        majority_share = most_common_count / len(numbers)
        distinct_count = len(counts)

        if majority_share > 0.5:
            return True  # one number dominates -> true shared value, rest is noise
        if distinct_count == len(numbers):
            return False  # every row has its own unique number -> genuinely distinct
        # ambiguous middle ground -- fall through to text-similarity check

    ratios = []
    for i in range(len(substantial) - 1):
        ratio = difflib.SequenceMatcher(None, substantial[i], substantial[i + 1]).ratio()
        ratios.append(ratio)
    avg_ratio = sum(ratios) / len(ratios)
    return avg_ratio > 0.90


def run_targeted_ocr(gray_img, struct: TableStructure, pad=2):
    """
    For every merge block: OCR the whole block AND each individual row
    within it, then verify whether the rows actually share one value
    before propagating it. If rows differ meaningfully, each row keeps
    its own distinct OCR result instead of being overwritten with a
    single shared block-level value.
    Standalone (non-merged) cells are OCR'd independently as before.
    """
    h, w = gray_img.shape

    def crop_bbox(bbox):
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            return None
        return gray_img[y0:y1, x0:x1]

    for region in struct.merge_regions:
        is_tall = region.row_end > region.row_start
        is_wide = region.col_end > region.col_start

        if is_tall:
            # OCR each row's own slice within the block first (per-row check)
            row_texts = {}
            for r in range(region.row_start, region.row_end + 1):
                # use the block's column span but this row's own Y-slice
                row_bbox = (struct.X[region.col_start], struct.Y[r],
                            struct.X[region.col_end + 1], struct.Y[r + 1])
                crop = crop_bbox(row_bbox)
                bin_crop = _preprocess_crop(crop)
                row_texts[r] = _ocr_crop(bin_crop, multiline=False)

            texts_in_order = [row_texts[r] for r in range(region.row_start, region.row_end + 1)]
            if _rows_are_actually_same_value(texts_in_order):
                # genuine merge: prefer a fresh whole-block OCR for the
                # cleanest combined read...
                crop = crop_bbox(region.bbox)
                bin_crop = _preprocess_crop(crop)
                shared_text = _ocr_crop(bin_crop, multiline=True)
                if not shared_text:
                    # ...but the whole-block crop can fail on large, mostly
                    # blank spans even when the per-row pass already found
                    # fragments of the real text -- don't discard those
                    substantial_fragments = [t for t in texts_in_order if _is_substantial(t)]
                    shared_text = " ".join(substantial_fragments)
                for r in range(region.row_start, region.row_end + 1):
                    for c in range(region.col_start, region.col_end + 1):
                        cell = struct.cells[(r, c)]
                        cell.ocr_text = shared_text
                        cell.has_information = bool(shared_text)
            else:
                # rows genuinely differ -- keep each row's own distinct value,
                # do NOT collapse into one repeated string
                for r in range(region.row_start, region.row_end + 1):
                    text = row_texts[r]
                    for c in range(region.col_start, region.col_end + 1):
                        cell = struct.cells[(r, c)]
                        cell.ocr_text = text
                        cell.has_information = bool(text)
        else:
            # horizontal-only merge (single row, multiple columns) --
            # rows aren't in question here, so the original whole-block
            # OCR + propagate-across-columns behavior is correct as-is
            crop = crop_bbox(region.bbox)
            bin_crop = _preprocess_crop(crop)
            text = _ocr_crop(bin_crop, multiline=False)
            for r in range(region.row_start, region.row_end + 1):
                for c in range(region.col_start, region.col_end + 1):
                    cell = struct.cells[(r, c)]
                    cell.ocr_text = text
                    cell.has_information = bool(text)

    # Standalone (non-merged) cells, unchanged
    for (r, c), cell in struct.cells.items():
        if cell.merge_flag:
            continue
        crop = crop_bbox(cell.bbox)
        bin_crop = _preprocess_crop(crop)
        text = _ocr_crop(bin_crop, multiline=False)
        cell.ocr_text = text
        cell.has_information = bool(text)
