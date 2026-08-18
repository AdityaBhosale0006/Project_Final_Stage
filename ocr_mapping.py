"""
Stage 2: Independent OCR + mapping into the learned structure.
OCR NEVER determines structure -- it only produces (bbox, text),
which is then assigned into the CellRegions/MergeRegions built in Stage 1.
"""
import cv2
import numpy as np
from structure import TableStructure


def run_ocr_words(gray_img, upscale=3):
    """
    Runs OCR independently over the whole table image and returns a flat
    list of {bbox, text} -- no structural knowledge involved.
    """
    import pytesseract  # lazy import: only this function needs Tesseract,
                         # the rest of this module (classify_confusion_matrix,
                         # used regardless of OCR engine) doesn't.
    big = cv2.resize(gray_img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    _, big_bin = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data = pytesseract.image_to_data(big_bin, config='--psm 6', output_type=pytesseract.Output.DICT)

    words = []
    n = len(data['text'])
    for i in range(n):
        txt = data['text'][i].strip()
        conf = int(float(data['conf'][i])) if data['conf'][i] not in ('-1', '') else -1
        if not txt or conf < 0:
            continue
        # scale bbox back down to original image coordinates
        x = data['left'][i] / upscale
        y = data['top'][i] / upscale
        w = data['width'][i] / upscale
        h = data['height'][i] / upscale
        words.append({'bbox': (x, y, x + w, y + h), 'text': txt, 'conf': conf})
    return words


def word_center(word):
    x0, y0, x1, y1 = word['bbox']
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def locate_cell_for_point(px, py, struct: TableStructure):
    """Find which logical (row, col) a point falls into, via interval containment
    -- NOT bbox overlap with cell crops. This is what makes Case D correct:
    a row is owned by its Y-interval regardless of which column merges."""
    col = None
    for c in range(struct.n_cols):
        if struct.X[c] <= px < struct.X[c + 1]:
            col = c
            break
    row = None
    for r in range(struct.n_rows):
        if struct.Y[r] <= py < struct.Y[r + 1]:
            row = r
            break
    return row, col


def map_ocr_to_structure(words, struct: TableStructure):
    """
    Assigns each OCR word into its owning logical cell / merge region using
    row & column intervals (per spec: never rely on bbox overlap alone).
    Implements the four-case confusion matrix implicitly:
      - merge_flag True + word found  -> Case A
      - merge_flag True + no word     -> Case B (handled by leaving merge
        region present with empty text; a second pass below searches the
        full merge bbox range for stray words before giving up)
      - merge_flag False + word found -> Case C
      - merge_flag False + no word    -> Case D (row's other columns may
        still be merged; this cell's own row/col identity doesn't change)
    """
    # bucket words by their owning (row, col)
    assigned = {}
    unassigned_words = []
    for w in words:
        px, py = word_center(w)
        row, col = locate_cell_for_point(px, py, struct)
        if row is None or col is None:
            unassigned_words.append(w)  # word fell outside any known cell (e.g. margin noise)
            continue
        assigned.setdefault((row, col), []).append(w)

    # Build merge_id -> list of (row,col) member cells, in reading order
    merge_members = {}
    for (r, c), cell in struct.cells.items():
        if cell.merge_flag:
            merge_members.setdefault(cell.merge_id, []).append((r, c))

    # First pass: direct assignment (Case A / Case C)
    for (r, c), cell in struct.cells.items():
        cell_words = assigned.get((r, c), [])
        if cell_words:
            cell_words.sort(key=lambda w: (round(word_center(w)[1]), word_center(w)[0]))
            cell.ocr_text = " ".join(w['text'] for w in cell_words)
            cell.has_information = True

    # Second pass: for merge regions with no info directly assigned to their
    # anchor cell, search ALL member rows' word buckets before giving up (Case B)
    for region in struct.merge_regions:
        members = merge_members.get(region.merge_id, [])
        combined_words = []
        for (r, c) in members:
            combined_words.extend(assigned.get((r, c), []))
        if combined_words:
            combined_words.sort(key=lambda w: (round(word_center(w)[1]), word_center(w)[0]))
            text = " ".join(w['text'] for w in combined_words)
            for (r, c) in members:
                struct.cells[(r, c)].ocr_text = text
                struct.cells[(r, c)].has_information = True

    return unassigned_words


def classify_confusion_matrix(struct: TableStructure):
    """Returns counts for the four cases, for diagnostic/QA purposes."""
    counts = {'A_merge_info': 0, 'B_merge_missing': 0, 'C_single_info': 0, 'D_single_missing': 0}
    for cell in struct.cells.values():
        if cell.merge_flag and cell.has_information:
            counts['A_merge_info'] += 1
        elif cell.merge_flag and not cell.has_information:
            counts['B_merge_missing'] += 1
        elif not cell.merge_flag and cell.has_information:
            counts['C_single_info'] += 1
        else:
            counts['D_single_missing'] += 1
    return counts