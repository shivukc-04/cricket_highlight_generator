"""Modality B - scoreboard OCR verification (EasyOCR on a user-defined ROI)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROI = Tuple[int, int, int, int]  # x, y, w, h

logger = logging.getLogger(__name__)
MIN_SCORE_CONF = 0.30   # below this the scoreboard is treated as "not visible"

SCORE_RE = re.compile(r"(?<!\d)(\d{1,3})\s*[/\\-]\s*(\d{1,3})(?!\d)")
OVERS_RE = re.compile(r"\b([0-4]?\d|50)[.,]([0-6])\b")
CALIB_RE = re.compile(r"\d+[\/\-]\d+")            # e.g. 123/4, 45-2
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 100, 100, 30, 30
BOUNDARY_TOKENS = {"4", "6", "W", "FOUR", "SIX", "WICKET", "OUT"}


@dataclass
class Reading:
    text: str = ""
    elements: List[str] = field(default_factory=list)
    runs: Optional[int] = None
    wickets: Optional[int] = None
    overs: Optional[float] = None
    score_text: str = ""                       # the matched score string, e.g. "142/3"
    bbox: Optional[Tuple[int, int, int, int]] = None   # (x, y, w, h) in full-frame pixels
    confidence: float = 0.0                    # OCR confidence of the score text

    def visible(self, min_conf: float = MIN_SCORE_CONF) -> bool:
        """True only if a valid score pattern was read with enough confidence."""
        return bool(self.score_text) and self.confidence >= min_conf

    @property
    def tokens(self) -> set:
        words = re.split(r"[\s|]+", self.text.upper()) + [e.strip().upper() for e in self.elements]
        return set(words) & BOUNDARY_TOKENS


@dataclass
class OCRResult:
    confirmed: bool
    reason: str
    before_text: str = ""
    after_text: str = ""
    visible: bool = True        # False -> discarded: scoreboard missing (replay / graphic / ad)


def parse_score(text: str) -> Tuple[Optional[int], Optional[int]]:
    m = SCORE_RE.search(text)
    if not m:
        return None, None
    a, b = int(m.group(1)), int(m.group(2))
    if b > 10 and a <= 10:          # "3/142" style (wickets/runs)
        a, b = b, a
    if b > 10:                      # neither side can be a wicket count -> not a score
        return None, None
    return a, b


def parse_overs(text: str) -> Optional[float]:
    m = OVERS_RE.search(text.replace(",", "."))
    return float(m.group(0).replace(",", ".")) if m else None


def grab_frame(cap: cv2.VideoCapture, t: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def clamp_roi(roi: ROI, frame_shape) -> Optional[ROI]:
    h, w = frame_shape[:2]
    x, y, rw, rh = [int(v) for v in roi]
    x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
    rw, rh = min(rw, w - x), min(rh, h - y)
    return (x, y, rw, rh) if rw > 4 and rh > 4 else None


def compare(before: Reading, after: Reading, min_run_delta: int = 4,
            max_run_delta: int = 36) -> Tuple[bool, str]:
    """Verified score event: runs jump, wicket falls, or boundary text appears."""
    if None not in (before.runs, after.runs, before.wickets, after.wickets):
        dr, dw = after.runs - before.runs, after.wickets - before.wickets
        if 1 <= dw <= 2 and dr >= 0:
            return True, f"wicket (+{dw}) {before.runs}/{before.wickets} -> {after.runs}/{after.wickets}"
        if dw == 0 and min_run_delta <= dr <= max_run_delta:
            return True, f"runs +{dr} ({before.runs}/{before.wickets} -> {after.runs}/{after.wickets})"
    new_tokens = after.tokens - before.tokens
    if new_tokens:
        return True, f"boundary text appeared: {', '.join(sorted(new_tokens))}"
    return False, "no scoreboard change"


class ScoreboardDetector:
    """Auto-calibrating scoreboard reader.

    ``calibrate(frame)`` runs EasyOCR on the FULL frame once, finds score-like text
    (``\\d+[/-]\\d+``), pads the box (-100/+100 px x, -30/+30 px y, clamped to the
    frame) and stores it as ``self.roi``. All later reads crop to that ROI first,
    so OCR runs on a tiny slice. The ROI can also be set manually (fallback).
    """

    def __init__(self, reader, roi: Optional[ROI] = None):
        self.reader = reader
        self.roi: Optional[ROI] = roi

    # ---- calibration ---------------------------------------------------- #
    def calibrate(self, frame) -> bool:
        if frame is None:
            return False
        h, w = frame.shape[:2]
        for bbox, text, _prob in self.reader.readtext(frame):
            if CALIB_RE.search(text):
                xs = [int(pt[0]) for pt in bbox]
                ys = [int(pt[1]) for pt in bbox]
                x0 = max(0, min(xs) - PAD_LEFT)
                y0 = max(0, min(ys) - PAD_TOP)
                x1 = min(w, max(xs) + PAD_RIGHT)
                y1 = min(h, max(ys) + PAD_BOTTOM)
                if x1 - x0 > 4 and y1 - y0 > 4:
                    self.roi = (x0, y0, x1 - x0, y1 - y0)
                    return True
        return False

    def calibrate_scan(self, cap: cv2.VideoCapture, duration: float,
                       start_t: Optional[float] = None, tries: int = 8):
        """Try ``start_t`` first, then evenly spaced frames (skips ads / full-screen
        graphics). Returns the time (s) of the frame that calibrated, or None."""
        times = [] if start_t is None else [float(start_t)]
        times += [duration * f for f in np.linspace(0.05, 0.95, tries)]
        for t in times:
            if self.calibrate(grab_frame(cap, t)):
                return t
        return None

    # ---- reading (always ROI-cropped) ----------------------------------- #
    def _ocr_roi(self, frame):
        """OCR the ROI crop only. Returns (results, roi, scale)."""
        if self.roi is None:
            raise ValueError("Scoreboard not calibrated: call calibrate(frame) or set roi.")
        roi = clamp_roi(self.roi, frame.shape)
        if roi is None:
            return [], None, 2.0
        x, y, w, h = roi
        crop = frame[y:y + h, x:x + w]
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return self.reader.readtext(crop), roi, 2.0

    @staticmethod
    def _best_score(results, roi, scale):
        """(score_text, bbox, confidence) of the best valid score in the OCR results."""
        def to_bbox(box):
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            return (int(roi[0] + min(xs) / scale), int(roi[1] + min(ys) / scale),
                    int((max(xs) - min(xs)) / scale), int((max(ys) - min(ys)) / scale))
        best = ("", None, 0.0)
        for box, text, prob in results:                       # a single element holds the score
            m = SCORE_RE.search(str(text))
            if m and parse_score(m.group(0))[0] is not None and float(prob) > best[2]:
                best = (m.group(0), to_bbox(box), float(prob))
        if best[0]:
            return best
        joined = " ".join(str(r[1]) for r in results)         # score split across elements
        m = SCORE_RE.search(joined)
        if m and parse_score(m.group(0))[0] is not None and results:
            conf = float(np.mean([float(r[2]) for r in results]))
            return (m.group(0), to_bbox(results[0][0]), conf)
        return best

    def read_score(self, frame):
        """Returns (score_text, bbox, confidence). ('', None, 0.0) if no valid score
        pattern such as '142/3' is visible in the ROI."""
        results, roi, scale = self._ocr_roi(frame)
        if roi is None:
            return "", None, 0.0
        return self._best_score(results, roi, scale)

    def read(self, frame) -> Reading:
        results, roi, scale = self._ocr_roi(frame)
        if roi is None:
            return Reading()
        elements = [str(r[1]) for r in results]
        text = " | ".join(elements)
        runs, wkts = parse_score(text)
        st, bb, cf = self._best_score(results, roi, scale)
        return Reading(text, elements, runs, wkts, parse_overs(text), st, bb, cf)

    def verify_event(self, cap: cv2.VideoCapture, t: float, duration: float,
                     before_sec: float = 8.0, after_sec: float = 10.0,
                     after_jitter: Sequence[float] = (-2.0, 0.0, 2.0),
                     min_run_delta: int = 4, require_visible: bool = True,
                     min_conf: float = MIN_SCORE_CONF) -> OCRResult:
        """Visibility filter first, then change detection.

        A candidate is discarded immediately when the broadcast scoreboard is not
        readable (no valid score pattern, or confidence < ``min_conf``) in the
        'before' frame, or in every 'after' frame - i.e. replays, crowd shots,
        full-screen graphics and ads can never be confirmed."""
        not_visible = "scoreboard not visible (replay/graphic/ad)"
        fb = grab_frame(cap, max(0.0, t - before_sec))
        if fb is None:
            return OCRResult(False, "could not read 'before' frame", visible=False)
        rb = self.read(fb)
        if require_visible and not rb.visible(min_conf):
            logger.info(f"Skipping candidate at {t:.1f}s: Scoreboard not visible (Replay/Graphic/Ad).")
            return OCRResult(False, not_visible + " before the event", rb.text, "", visible=False)

        last, any_visible = rb, False
        for j in after_jitter:
            fa = grab_frame(cap, min(max(0.0, duration - 0.5), t + after_sec + j))
            if fa is None:
                continue
            ra = self.read(fa)
            if require_visible and not ra.visible(min_conf):
                continue                       # replay / graphic frame: ignore this sample
            any_visible = True
            last = ra
            ok, reason = compare(rb, ra, min_run_delta)
            if ok:
                return OCRResult(True, reason, rb.text, ra.text)
        if require_visible and not any_visible:
            logger.info(f"Skipping candidate at {t:.1f}s: Scoreboard not visible (Replay/Graphic/Ad).")
            return OCRResult(False, not_visible + " after the event", rb.text, "", visible=False)
        return OCRResult(False, "no scoreboard change", rb.text, last.text)

ScoreboardOCR = ScoreboardDetector  # backwards-compatible alias
