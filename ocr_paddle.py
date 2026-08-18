"""
Stage 2 (PaddleOCR variant): identical structural traversal and same-value-
vs-distinct decision logic as ocr_targeted.py, but the leaf "read this crop"
step uses PaddleOCR (PP-OCR) instead of Tesseract.

Structure (rows/cols/merges) is still fully decided beforehand by
structure.py -- this file only changes HOW Stage 2 reads text out of each
already-known cell/merge-block crop.

Where this sits relative to the Tesseract engine:
  - No LLM, no external server needed -- CPU-friendly enough for a
    Codespace or a laptop.
  - Generally more accurate than Tesseract on small/low-contrast text
    (PP-OCR's detection+recognition models are deep-learning based, not
    classical thresholding + template matching).
  - There's no column-aware prompting -- PaddleOCR has no concept of "this
    column should be one of these 11 colours". Rely on postprocess.py's
    existing vocabulary-snapping to do that work instead.

CAUTION -- version sensitivity: PaddleOCR's Python API changed significantly
around the 3.x / PaddleX-based release (predict() returning rec_texts/
rec_scores, replacing the older ocr() call returning [bbox, (text, conf)]
per line). This module tries the modern predict() API first and falls back
to the legacy ocr() call if predict() isn't available.

CONFIRMED against a live install (paddlepaddle 3.3.1 / paddleocr 3.7.0,
Ubuntu 24 CPU): the default constructor crashes on every single crop with
    (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
    [pir::ArrayAttribute<pir::DoubleAttribute>]
This is a gap in that PaddlePaddle version's PIR (Paddle IR) executor when
paired with its oneDNN/MKL-DNN CPU backend -- some model-internal op takes
an array-of-doubles attribute that the oneDNN PIR instruction layer can't
yet convert. Fixed here by constructing PaddleOCR with enable_mkldnn=False,
which routes inference through the plain (non-oneDNN) CPU kernels instead.

----------------------------------------------------------------------------
WHY THIS FILE HAS TWO ENGINES (rec-only primary, full det+rec fallback)
----------------------------------------------------------------------------
Every crop this pipeline hands to OCR already has a KNOWN bounding box --
structure.py found it geometrically before any OCR ran. That means the
question "is there text somewhere in this image" (what the detection stage
answers) is never actually in doubt; the only real question is "what does
the text in this exact box say" (what the recognition stage answers).

Running the full predict() pipeline (detection -> recognition) on a crop
that's already tightly cropped to one cell wastes the detector's job on a
question we've already answered, and worse, actively breaks on sparse
crops: PP-OCR's detector (DB / differentiable binarization) filters out
connected components below a minimum size before it ever calls recognition.
A crop containing nothing but a single thin glyph (e.g. a lone "1" in a
QTY column) can fall below that size filter -- detection returns zero
boxes, recognition never runs, and the cell silently comes back as ""
instead of "1". This was confirmed directly: a crop containing only "1"
returned "", while an otherwise-identical crop containing "1" plus an
incidental stray mark (large enough to push the connected component over
the detector's size threshold) correctly returned "1". The recognition
model read the digit fine in both cases -- detection is what dropped it.

Fix: skip detection entirely for these crops and call PaddleOCR's
standalone recognition-only module (paddleocr.TextRecognition, part of
the PaddleX-based module set introduced in 3.x) directly on each crop.
There's no size filter to fall below because there's no "should I look
here at all" decision left to make.

The full det+rec PaddleOCR() engine is kept as a fallback, used whenever:
  (a) the TextRecognition module can't be imported/constructed on this
      install (older paddleocr versions predate the standalone modules),
  (b) the rec-only call raises, comes back low-confidence, or empty, or
  (c) the crop looks like it holds MORE THAN ONE text line (see below) --
      a single-line recognition model can't segment stacked lines, only
      a detector can.

----------------------------------------------------------------------------
WHY THIS IS TABLE-AGNOSTIC, NOT SPECIFIC TO THE PIPE-BOM SHEET
----------------------------------------------------------------------------
The rec-only-first idea only actually generalizes safely across arbitrary
table layouts if two assumptions from the original single-table version
are removed:

  1. "Every crop contains exactly one text line."
     Not true in general -- many tables wrap long cell text across 2+
     lines within a single cell. A recognition-only call fed a two-line
     crop will try to read it as one line and mangle it. Fix: before
     choosing an engine, run a cheap row-wise ink-projection check
     (_looks_multiline) on the crop. If it finds more than one distinct
     horizontal band of ink separated by a gap, treat it as multi-line
     and go straight to the full detection+recognition engine, which can
     find and read each line as its own box. Single-line crops (the
     common case, and the only case in the pipe-BOM sheet) still get the
     fast/robust rec-only path.

  2. "Every crop that isn't the one specific glyph we tested contains
     real text." Also not safe in general -- recognition models have no
     'nothing here' output, they always emit *some* string, even for
     pure noise or a genuinely blank cell. A sparser table (lots of
     legitimately empty cells) would start hallucinating phantom text
     where the old detection-based code correctly returned "". Fix: a
     cheap ink-density check (_is_blank_crop) runs first and short-
     circuits to "" without calling any model if the crop has
     essentially no dark pixels. Also, the rec-only result now carries
     its confidence score, and a low-confidence read is treated as a
     miss and handed to the fallback engine rather than trusted outright.

None of these checks reference column names, this table's row/column
counts, or anything else specific to the pipe-BOM sheet -- they operate
purely on crop pixel content, so the same file should behave reasonably
on a differently-shaped table dropped into the same structure.py +
ocr_paddle.py pipeline. That said, the numeric thresholds below
(dark-pixel fraction, ink-gap size, confidence cutoff) are heuristics,
not universal constants -- if you start running this against tables with
very different fonts/scan quality, spot-check a batch of crops and its
printed engine-choice output, and retune the module-level constants
below for that font/resolution.

Install (unchanged):
    pip install paddleocr --break-system-packages
"""
import cv2
import numpy as np

from ocr_targeted import _is_substantial, _rows_are_actually_same_value
from structure import TableStructure

_rec_engine = None   # recognition-only engine (primary path)
_full_engine = None  # full detection+recognition engine (fallback only)
_rec_engine_failed = False  # set True if rec-only construction/import fails,
                             # so we don't retry the failing import on every
                             # single crop -- fall straight to the full
                             # engine for the rest of this run instead.

# ---------------------------------------------------------------------------
# Tunable heuristic thresholds -- see the module docstring section on
# generalization for what each one guards against. These are starting
# points calibrated against a clean scanned/printed technical-drawing
# table; if you point this at noisier scans, handwriting, or a very
# different font/DPI, spot-check a batch of crops and adjust.
# ---------------------------------------------------------------------------
BLANK_DARK_FRAC_THRESHOLD = 0.005   # below this fraction of ink pixels,
                                     # treat the crop as blank -- skip OCR
                                     # entirely rather than let a model
                                     # hallucinate text from noise.
MULTILINE_MIN_ROW_INK_FRAC = 0.02   # a row counts as "has ink" once more
                                     # than this fraction of its width is
                                     # dark -- filters out speckle noise.
MULTILINE_MIN_GAP_ROWS = 2          # need at least this many consecutive
                                     # ink-free rows between two ink bands
                                     # before treating them as separate
                                     # lines (avoids splitting one line
                                     # into two over an accent/gap in a
                                     # character's own strokes).
MIN_REC_SCORE = 0.5                 # recognition-only confidence below
                                     # this is treated as an unreliable
                                     # read -- fall back to the full
                                     # detection+recognition engine
                                     # instead of trusting a low-confidence
                                     # single-line guess.


def _rec_model_name(lang):
    # PaddleX/PaddleOCR 3.x model naming. Adjust if your installed model
    # zoo uses different names -- check with:
    #   python3 -c "from paddleocr import TextRecognition; print(TextRecognition.__doc__)"
    # or the PaddleOCR 3.x model list docs.
    return "en_PP-OCRv4_mobile_rec" if lang == "en" else "PP-OCRv4_mobile_rec"


def _get_rec_engine(lang="en"):
    """Recognition-only engine -- no detection stage. Lazily built once,
    reused across all crops. Returns None (not a raised exception) if
    unavailable, so callers can cleanly fall back."""
    global _rec_engine, _rec_engine_failed
    if _rec_engine is not None:
        return _rec_engine
    if _rec_engine_failed:
        return None
    try:
        from paddleocr import TextRecognition
        try:
            _rec_engine = TextRecognition(model_name=_rec_model_name(lang))
        except TypeError:
            # Older/newer signature that doesn't take model_name the same
            # way -- fall back to the module's own default model.
            _rec_engine = TextRecognition()
        print("  [paddle ocr] recognition-only engine ready "
              f"(model={_rec_model_name(lang)})")
    except Exception as e:
        print(f"  [paddle ocr] recognition-only engine unavailable ({e}); "
              "falling back to full detection+recognition pipeline for "
              "every crop. Isolated single-glyph cells (e.g. lone digits) "
              "are at risk of coming back empty on this path -- see the "
              "module docstring for why.")
        _rec_engine_failed = True
        _rec_engine = None
    return _rec_engine


def _get_full_engine(lang="en"):
    """Full detection+recognition engine. Fallback only -- see module
    docstring for why this is no longer the primary path."""
    global _full_engine
    if _full_engine is not None:
        return _full_engine
    from paddleocr import PaddleOCR
    try:
        _full_engine = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    except TypeError:
        _full_engine = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)
    return _full_engine


def _is_blank_crop(gray_crop, dark_frac_threshold=BLANK_DARK_FRAC_THRESHOLD):
    """Generic (table-agnostic) check: does this crop have enough ink to
    plausibly contain text at all? Operates purely on pixel content --
    no knowledge of column identity, this table's layout, or anything
    else specific to one sheet. Otsu-binarize, measure the fraction of
    foreground (ink) pixels, and treat near-zero as blank.

    This matters for generalizing beyond the pipe-BOM sheet: that table
    happens to have almost no genuinely empty cells, so this case never
    came up there. Tables with legitimately blank cells would otherwise
    have those cells fed into the recognition-only model, which has no
    'nothing here' output and would hallucinate some string instead of
    correctly reporting no text."""
    if gray_crop is None or gray_crop.size == 0:
        return True
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_frac = float(np.count_nonzero(binary)) / binary.size
    return dark_frac < dark_frac_threshold


def _looks_multiline(gray_crop,
                      min_row_ink_frac=MULTILINE_MIN_ROW_INK_FRAC,
                      min_gap_rows=MULTILINE_MIN_GAP_ROWS):
    """Generic (table-agnostic) check: does this crop likely contain more
    than one stacked line of text? Uses a row-wise ink projection profile
    -- binarize, sum dark pixels per row, and count distinct bands of
    "this row has text" separated by a run of "this row is blank" rows.
    More than one band => probably multiple text lines.

    This matters for generalizing beyond the pipe-BOM sheet: every cell
    there happens to hold a single line, so a recognition-only call
    (which reads one line, not a paragraph) was always safe. A table
    with wrapped multi-line cell content would otherwise get those
    cells mangled by the single-line model. Flagging multi-line crops
    routes them to the full detection+recognition engine instead, which
    can find and read each line as its own box."""
    if gray_crop is None or gray_crop.size == 0:
        return False
    h, w = gray_crop.shape
    if h < 12 or w < 4:
        return False  # too small to meaningfully judge line count
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_ink_frac = (binary > 0).sum(axis=1).astype(float) / max(w, 1)
    has_ink = row_ink_frac > min_row_ink_frac

    bands = 0
    in_band = False
    blank_run = 0
    for v in has_ink:
        if v:
            if not in_band:
                bands += 1
                in_band = True
            blank_run = 0
        else:
            blank_run += 1
            if blank_run >= min_gap_rows:
                in_band = False
    return bands > 1


def _light_preprocess(gray_crop, target_height=96, max_scale=6.0):
    """Upscale + mild contrast boost. PP-OCR's recognition model is a deep
    net trained on natural images, so mild contrast enhancement helps more
    than classical OTSU binarization would."""
    h, w = gray_crop.shape
    if h == 0 or w == 0:
        return None
    scale = min(max(1.0, target_height / h), max_scale)
    big = cv2.resize(gray_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(big)
    # PaddleOCR expects a 3-channel image.
    return cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)


def _extract_texts_modern(result):
    """Modern predict() result: list of result objects, one per input image,
    each with .rec_texts / ["rec_texts"] holding recognized line strings.
    (Full det+rec engine output shape.)"""
    texts = []
    for page in result:
        rec_texts = None
        if hasattr(page, "rec_texts"):
            rec_texts = page.rec_texts
        elif isinstance(page, dict):
            rec_texts = page.get("rec_texts")
        if rec_texts:
            texts.extend(rec_texts)
    return texts


def _extract_texts_legacy(result):
    """Legacy ocr() result: list (per image) of list of [bbox, (text, conf)]."""
    texts = []
    if not result:
        return texts
    for line_result in (result[0] or []):
        try:
            text = line_result[1][0]
            texts.append(text)
        except (IndexError, TypeError):
            continue
    return texts


def _extract_text_recognition(result):
    """Standalone TextRecognition module output shape: list of result
    objects/dicts, one per input image, each carrying a single 'rec_text'
    + 'rec_score' -- there's no detection step, so there's exactly one
    text prediction per crop, not a list of lines. Returns a list of
    (text, score) pairs; score defaults to 1.0 if the install's output
    shape doesn't expose one (older/newer API variants)."""
    out = []
    for page in result:
        text, score = None, 1.0
        if hasattr(page, "rec_text"):
            text = page.rec_text
            score = getattr(page, "rec_score", 1.0)
        elif isinstance(page, dict):
            text = page.get("rec_text")
            score = page.get("rec_score", 1.0)
        if text:
            out.append((text, float(score) if score is not None else 1.0))
    return out


def paddle_available():
    """Upfront check so a missing install fails fast with one clear
    message, not mid-run on crop 1 of 243."""
    try:
        import paddleocr  # noqa: F401
        return True
    except ImportError:
        return False


def _ocr_crop_paddle_recognition_only(processed, lang="en"):
    """Primary path: recognition only, no detection. Returns (text, score)
    -- text is "" on failure/unavailability/no-output, in which case
    callers should fall back to the full engine. score is the minimum
    confidence across any text fragments returned (only one fragment is
    expected per crop in normal use, since this is a single-line call)."""
    engine = _get_rec_engine(lang=lang)
    if engine is None:
        return "", 0.0
    try:
        result = engine.predict(processed)
        pairs = _extract_text_recognition(result)
        pairs = [(t.strip(), s) for t, s in pairs if t and t.strip()]
        if not pairs:
            return "", 0.0
        text = " ".join(t for t, _ in pairs)
        min_score = min(s for _, s in pairs)
        return text, min_score
    except Exception as e:
        print(f"  [paddle ocr] recognition-only crop failed: {e}")
        return "", 0.0


def _ocr_crop_paddle_full(processed, lang="en"):
    """Fallback path: full detection+recognition pipeline."""
    try:
        engine = _get_full_engine(lang=lang)
        if hasattr(engine, "predict"):
            result = engine.predict(processed)
            texts = _extract_texts_modern(result)
        else:
            result = engine.ocr(processed)
            texts = _extract_texts_legacy(result)
        return " ".join(t.strip() for t in texts if t and t.strip())
    except Exception as e:
        print(f"  [paddle ocr] full-pipeline crop failed: {e}")
        return ""


def _ocr_crop_paddle(gray_crop, lang="en"):
    """Single-crop PaddleOCR call. Table-agnostic dispatch:

      1. Blank check on the RAW crop (before upscale/contrast, which can
         amplify faint noise into something that no longer reads as
         blank) -- if there's essentially no ink, return "" without
         calling any model. Avoids hallucinated text on genuinely empty
         cells, which the old detection-based code never had to worry
         about (a detector finding zero boxes on a blank crop is the
         correct, cheap answer already).
      2. Multi-line check on the RAW crop -- if it looks like more than
         one stacked text line, skip straight to the full detection+
         recognition engine, which can segment lines; a single-line
         recognition-only call would just mangle them together.
      3. Otherwise, try recognition-only first (fixes the dropped-
         isolated-glyph failure mode -- see module docstring) and accept
         its result only if confidence clears MIN_REC_SCORE.
      4. Any case that didn't return in steps 1-3 falls back to the full
         detection+recognition pipeline as a second attempt.

    Returns '' on any failure rather than raising, so one bad crop
    doesn't kill a 200+ crop run."""
    if gray_crop is None:
        return ""
    if _is_blank_crop(gray_crop):
        return ""

    processed = _light_preprocess(gray_crop)
    if processed is None:
        return ""

    if _looks_multiline(gray_crop):
        return _ocr_crop_paddle_full(processed, lang=lang)

    text, score = _ocr_crop_paddle_recognition_only(processed, lang=lang)
    if text and score >= MIN_REC_SCORE:
        return text

    # Recognition-only came back empty, low-confidence, or was
    # unavailable -- fall back to the full pipeline as a second attempt
    # before giving up.
    return _ocr_crop_paddle_full(processed, lang=lang)


def run_targeted_ocr_paddle(gray_img, struct: TableStructure, pad=2, lang="en", progress=True):
    """
    Drop-in replacement for ocr_targeted.run_targeted_ocr(). Same block
    traversal and same _rows_are_actually_same_value() decision for merge
    blocks -- only the leaf read call differs (PaddleOCR instead of
    Tesseract).
    """
    h, w = gray_img.shape

    def crop_bbox(bbox):
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            return None
        return gray_img[y0:y1, x0:x1]

    total_blocks = len(struct.merge_regions) + sum(1 for c in struct.cells.values() if not c.merge_flag)
    done = 0

    for region in struct.merge_regions:
        is_tall = region.row_end > region.row_start

        if is_tall:
            row_texts = {}
            for r in range(region.row_start, region.row_end + 1):
                row_bbox = (struct.X[region.col_start], struct.Y[r],
                            struct.X[region.col_end + 1], struct.Y[r + 1])
                crop = crop_bbox(row_bbox)
                row_texts[r] = _ocr_crop_paddle(crop, lang=lang)

            texts_in_order = [row_texts[r] for r in range(region.row_start, region.row_end + 1)]
            if _rows_are_actually_same_value(texts_in_order):
                crop = crop_bbox(region.bbox)
                shared_text = _ocr_crop_paddle(crop, lang=lang)
                if not shared_text:
                    substantial_fragments = [t for t in texts_in_order if _is_substantial(t)]
                    shared_text = " ".join(substantial_fragments)
                for r in range(region.row_start, region.row_end + 1):
                    for c in range(region.col_start, region.col_end + 1):
                        cell = struct.cells[(r, c)]
                        cell.ocr_text = shared_text
                        cell.has_information = bool(shared_text)
            else:
                for r in range(region.row_start, region.row_end + 1):
                    text = row_texts[r]
                    for c in range(region.col_start, region.col_end + 1):
                        cell = struct.cells[(r, c)]
                        cell.ocr_text = text
                        cell.has_information = bool(text)
        else:
            crop = crop_bbox(region.bbox)
            text = _ocr_crop_paddle(crop, lang=lang)
            for r in range(region.row_start, region.row_end + 1):
                for c in range(region.col_start, region.col_end + 1):
                    cell = struct.cells[(r, c)]
                    cell.ocr_text = text
                    cell.has_information = bool(text)

        done += 1
        if progress:
            print(f"  [paddle ocr] merge block {done}/{total_blocks}")

    for (r, c), cell in struct.cells.items():
        if cell.merge_flag:
            continue
        crop = crop_bbox(cell.bbox)
        text = _ocr_crop_paddle(crop, lang=lang)
        cell.ocr_text = text
        cell.has_information = bool(text)
        done += 1
        if progress and done % 20 == 0:
            print(f"  [paddle ocr] {done}/{total_blocks} blocks")