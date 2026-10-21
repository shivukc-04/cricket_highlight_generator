"""🏏 Automated Cricket Highlight Engine (Streamlit).

Audio excitement  ->  scoreboard OCR verification  ->  CNN run-up alignment
->  strict interval de-duplication  ->  one continuous highlight MP4.

Run:  streamlit run app.py
"""
from __future__ import annotations

import atexit
import hashlib
import logging
import os
import shutil
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from utils import audio, classifier, merger, ocr

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Cricket Highlight Engine", page_icon="🏏", layout="wide")

LOCAL_MODEL_CANDIDATES = ["./weights/runup_model.h5", "./weights/cricket_runup_final.h5",
                          "./cricket_runup_final.h5"]


# =========================================================================== #
# Helpers: temp storage, caching
# =========================================================================== #
def workdir() -> str:
    if "workdir" not in st.session_state:
        d = tempfile.mkdtemp(prefix="cricket_hl_")
        st.session_state["workdir"] = d
        atexit.register(shutil.rmtree, d, ignore_errors=True)
    return st.session_state["workdir"]


def save_upload(uploaded, suffix: str) -> str:
    """Stream an uploaded file to disk once (keyed by name+size) and return its path."""
    key = hashlib.md5(f"{uploaded.name}:{uploaded.size}".encode()).hexdigest()[:12]
    path = os.path.join(workdir(), f"{key}{suffix}")
    if not os.path.exists(path):
        uploaded.seek(0)
        with open(path, "wb") as f:
            shutil.copyfileobj(uploaded, f, length=8 * 1024 * 1024)
    return path


@st.cache_resource(show_spinner="Loading run-up model…")
def cached_model(path: str, mtime: float):
    return classifier.load_runup_model(path)


@st.cache_resource(show_spinner="Loading EasyOCR (first run downloads models)…")
def cached_reader(gpu: bool):
    import easyocr
    return easyocr.Reader(["en"], gpu=gpu)


@st.cache_data(show_spinner=False)
def cached_audio_curve(path: str, size: int, mtime: float, window: float, smooth: float):
    return audio.compute_audio_curve(path, window_sec=window, smooth_sec=smooth)


@st.cache_data(show_spinner=False)
def video_info(path: str, mtime: float) -> dict:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV could not open this video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30.0 if not fps or np.isnan(fps) else float(fps)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return dict(fps=fps, frames=int(n), duration=float(n) / fps,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


def read_frame(path: str, t: float):
    cap = cv2.VideoCapture(path)
    try:
        return ocr.grab_frame(cap, t)
    finally:
        cap.release()


def resolve_model_path(uploaded) -> str | None:
    if uploaded is not None:
        return save_upload(uploaded, ".h5")
    for p in LOCAL_MODEL_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


# =========================================================================== #
# Sidebar
# =========================================================================== #
st.sidebar.title("🏏 Highlight Engine")

st.sidebar.subheader("Inputs")
vid_upload = st.sidebar.file_uploader("Match video", type=["mp4", "mkv"])
vid_local = st.sidebar.text_input("…or local video path (best for multi-GB files)", "")
mdl_upload = st.sidebar.file_uploader("Run-up model (.h5)", type=["h5"],
                                      help="Falls back to ./weights/runup_model.h5")

st.sidebar.subheader("Detection")
k = st.sidebar.slider("Audio sensitivity k (Z-score)", 1.0, 4.0, 2.0, 0.1)
runup_thr = st.sidebar.slider("Run-up probability threshold", 0.10, 0.90, 0.50, 0.05)

st.sidebar.subheader("Buffers & gaps (seconds)")
lead_buffer = st.sidebar.number_input("lead_buffer", 0.0, 30.0, 2.0, 0.5)
fallback_lead = st.sidebar.number_input("fallback_lead", 0.0, 60.0, 6.0, 0.5)
post_roll = st.sidebar.number_input("post_roll", 0.0, 60.0, 4.0, 0.5)
merge_gap = st.sidebar.number_input("merge_gap", 0.0, 60.0, 4.0, 0.5)

with st.sidebar.expander("Advanced"):
    min_event_duration = st.number_input("min_event_duration", 1.0, 60.0, 5.0, 0.5)
    peak_gap = st.number_input("Min gap between audio peaks (s)", 1.0, 60.0, 10.0, 1.0)
    max_candidates = st.number_input("Max audio candidates to verify (strongest first)", 5, 2000, 150, 5)
    use_ocr = st.checkbox("Use scoreboard OCR verification", True)
    ocr_gpu = st.checkbox("EasyOCR on GPU", False)
    ocr_before = st.number_input("OCR: seconds before peak", 1.0, 30.0, 8.0, 1.0)
    ocr_after = st.number_input("OCR: seconds after peak", 1.0, 40.0, 10.0, 1.0)
    require_visible = st.checkbox("Discard candidates where the scoreboard isn't visible", True,
                                  help="Drops replays, crowd shots, full-screen graphics and ads.")
    min_ocr_conf = st.slider("Scoreboard visibility: min OCR confidence", 0.0, 1.0, 0.30, 0.05)
    min_run_delta = st.number_input("OCR: min runs increase", 1, 6, 4, 1,
                                    help="4 = boundaries only. Lower to 1 to accept any scoring shot.")
    runup_lookback = st.number_input("CNN lookback window (s)", 5.0, 60.0, 10.0, 1.0,
                                     help="How far before each audio peak to search for the run-up. "
                                          "Raise this if the roar comes >10 s after the run-up begins.")
    norm = st.selectbox("CNN input normalisation", ["0to1", "-1to1", "none"], 0,
                        help="The training notebook used RGB / 255 → '0to1'.")
    engine = st.selectbox("Render engine", ["auto", "moviepy", "ffmpeg"], 0,
                          help="auto = MoviePy if installed, else ffmpeg (faster, less RAM).")

if st.sidebar.button("🧹 Clear temp files & results"):
    shutil.rmtree(workdir(), ignore_errors=True)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    st.rerun()

# --------------------------------------------------------------------------- #
# Resolve the video
# --------------------------------------------------------------------------- #
video_path = None
if vid_upload is not None:
    video_path = save_upload(vid_upload, os.path.splitext(vid_upload.name)[1] or ".mp4")
elif vid_local.strip():
    p = vid_local.strip().strip('"')
    if os.path.isfile(p):
        video_path = p
    else:
        st.sidebar.error("That path doesn't exist.")

st.title("🏏 Automated Cricket Highlight Engine")
st.caption("Audio roar → scoreboard OCR → bowler run-up CNN → de-duplicated single reel")

if video_path is None:
    st.info("Upload a match video (or paste a local path) in the sidebar to begin.")
    st.stop()

try:
    vinfo = video_info(video_path, os.path.getmtime(video_path))
except Exception as e:
    st.error(f"Could not read the video: {e}")
    st.stop()

# --------------------------------------------------------------------------- #
# ROI state (reset when the video changes)
# --------------------------------------------------------------------------- #
W, H = vinfo["width"], vinfo["height"]
vkey = f"{video_path}:{W}x{H}"
if st.session_state.get("roi_vkey") != vkey:
    st.session_state.update(roi_vkey=vkey, roi_x=0, roi_y=int(H * 0.85),
                            roi_w=int(W * 0.40), roi_h=int(H * 0.15), roi_msg="",
                            preview_t=min(60, int(vinfo["duration"] // 2)),
                            video_path=video_path, roi_calibrated=False, auto_tried=False)
    st.session_state.pop("results", None)
st.session_state["video_path"] = video_path
st.session_state["roi_x"] = int(np.clip(st.session_state["roi_x"], 0, W - 1))
st.session_state["roi_y"] = int(np.clip(st.session_state["roi_y"], 0, H - 1))
st.session_state["roi_w"] = int(np.clip(st.session_state["roi_w"], 1, W))
st.session_state["roi_h"] = int(np.clip(st.session_state["roi_h"], 1, H))


def run_calibration(prefer_t: float | None) -> None:
    """ScoreboardDetector.calibrate() on the preferred frame, then a scan of other frames."""
    try:
        reader = cached_reader(bool(st.session_state.get("_ocr_gpu", False)))
        det = ocr.ScoreboardDetector(reader)
        cap = cv2.VideoCapture(st.session_state["video_path"])
        try:
            t = det.calibrate_scan(cap, vinfo["duration"], prefer_t)
        finally:
            cap.release()
        if t is not None:
            x, y, w, h = det.roi
            st.session_state.update(roi_x=x, roi_y=y, roi_w=w, roi_h=h, roi_calibrated=True,
                                    preview_t=int(min(t, max(1, vinfo["duration"] - 2))),
                                    roi_msg=f"✅ Scoreboard auto-calibrated at {merger.fmt_time(t)} "
                                            f"→ ROI {det.roi}. Check the green box.")
        else:
            st.session_state.update(roi_calibrated=False, roi_msg=(
                "❌ Auto-calibration found no score-like text (e.g. 123/4). Open 'Manual ROI' below "
                "and set the box with the sliders, or pick a frame showing the scoreboard and retry."))
    except Exception as e:  # noqa: BLE001
        st.session_state.update(roi_calibrated=False,
                                roi_msg=f"❌ Auto-calibration failed ({e}). Use the manual ROI sliders.")


def _calibrate_cb():
    run_calibration(float(st.session_state["preview_t"]))


st.session_state["_ocr_gpu"] = ocr_gpu
# Automatic calibration once per video (before any ROI widgets are created)
if use_ocr and not st.session_state["auto_tried"]:
    st.session_state["auto_tried"] = True
    with st.spinner("Calibrating scoreboard (EasyOCR on full frames)…"):
        run_calibration(None)

model_path = resolve_model_path(mdl_upload)

tab1, tab2, tab3 = st.tabs(["1 · Video & Scoreboard Setup", "2 · Processing & Diagnostics",
                            "3 · Final Reel & Export"])

# =========================================================================== #
# Tab 1 - ROI
# =========================================================================== #
with tab1:
    c1, c2 = st.columns([3, 2])
    with c2:
        st.markdown(f"**Video:** {W}×{H} · {vinfo['fps']:.2f} fps · {merger.fmt_time(vinfo['duration'])}")
        st.markdown(f"**Model:** `{model_path}`" if model_path else
                    "**Model:** ⚠️ none found — will use audio + fallback lead only")
        st.slider("Calibration / preview frame (seconds)", 0, max(1, int(vinfo["duration"]) - 1),
                  key="preview_t")
        st.button("🎯 Calibrate Scoreboard", on_click=_calibrate_cb,
                  help="Runs EasyOCR on the full frame, finds text like 123/4 and pads the box.")
        if st.session_state.get("roi_msg"):
            (st.success if st.session_state.get("roi_calibrated") else st.warning)(st.session_state["roi_msg"])
        with st.expander("Manual ROI (fallback)", expanded=not st.session_state.get("roi_calibrated")):
            st.slider("roi_x", 0, W - 1, key="roi_x")
            st.slider("roi_y", 0, H - 1, key="roi_y")
            st.slider("roi_w", 1, W, key="roi_w")
            st.slider("roi_h", 1, H, key="roi_h")

    roi = (st.session_state["roi_x"], st.session_state["roi_y"],
           st.session_state["roi_w"], st.session_state["roi_h"])
    frame = read_frame(video_path, float(st.session_state["preview_t"]))
    with c1:
        if frame is None:
            st.warning("Could not read a frame at that time.")
        else:
            shown = frame.copy()
            x, y, w, h = roi
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), max(2, W // 400))
            st.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), caption="Green box = scoreboard ROI (confirm it covers the score)",
                     use_container_width=True)
    if frame is not None:
        cc1, cc2 = st.columns([3, 2])
        crop_roi = ocr.clamp_roi(roi, frame.shape)
        if crop_roi:
            x, y, w, h = crop_roi
            cc1.image(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB),
                      caption="ROI crop (what OCR sees)")
            if cc2.button("Test OCR on this crop"):
                try:
                    r = ocr.ScoreboardDetector(cached_reader(ocr_gpu), crop_roi).read(frame)
                    cc2.code(r.text or "(nothing read)")
                    cc2.write(f"Parsed → runs: `{r.runs}` wickets: `{r.wickets}` overs: `{r.overs}`")
                except Exception as e:  # noqa: BLE001
                    cc2.error(f"OCR failed: {e}")
        else:
            cc2.warning("ROI is empty or outside the frame.")

# =========================================================================== #
# Pipeline
# =========================================================================== #
def run_pipeline() -> None:
    prog = st.progress(0.0, text="Starting…")
    duration, vfps = vinfo["duration"], vinfo["fps"]
    log = st.container()
    notes: list[str] = []

    # ---- Stage 1: audio -------------------------------------------------- #
    prog.progress(0.02, text="① Extracting audio & computing energy Z-scores…")
    curve = cached_audio_curve(video_path, os.path.getsize(video_path),
                               os.path.getmtime(video_path), audio.WINDOW_SEC, 0.5)
    peaks = audio.detect_peaks(curve, k=k, min_gap_sec=peak_gap)
    total_peaks = len(peaks)
    if total_peaks > max_candidates:
        peaks = sorted(sorted(peaks, key=lambda p: -p.z)[: int(max_candidates)], key=lambda p: p.time)
        notes.append(f"{total_peaks} audio peaks found; verifying the strongest {int(max_candidates)}.")
    prog.progress(0.15, text=f"① {len(peaks)} candidate audio peaks")
    if not peaks:
        prog.empty()
        st.warning("No audio peaks above the threshold — lower the sensitivity k.")
        return

    events = [dict(t=p.time, z=p.z, ocr=None, ocr_reason="", conf=None, source="", rs=None)
              for p in peaks]

    # ---- Stage 2: OCR ---------------------------------------------------- #
    roi_ok = ocr.clamp_roi(roi, (H, W)) if use_ocr else None
    ocr_active = False
    if use_ocr and roi_ok:
        try:
            scorer = ocr.ScoreboardDetector(cached_reader(ocr_gpu), roi_ok)
            ocr_active = True
        except Exception as e:  # noqa: BLE001
            notes.append(f"⚠️ EasyOCR unavailable ({e}); continuing without OCR verification.")
    elif use_ocr:
        notes.append("⚠️ Scoreboard ROI is invalid; continuing without OCR verification.")

    if ocr_active:
        cap = cv2.VideoCapture(video_path)
        try:
            for i, ev in enumerate(events):
                prog.progress(0.15 + 0.40 * i / len(events),
                              text=f"② OCR check {i + 1}/{len(events)} @ {merger.fmt_time(ev['t'])}")
                res = scorer.verify_event(cap, ev["t"], duration, ocr_before, ocr_after,
                                          min_run_delta=int(min_run_delta),
                                          require_visible=require_visible, min_conf=min_ocr_conf)
                ev["ocr"], ev["ocr_reason"] = res.confirmed, res.reason
                ev["hidden"] = not res.visible
        finally:
            cap.release()
        n_hidden = sum(1 for e in events if e.get("hidden"))
        if n_hidden:
            notes.append(f"🚫 {n_hidden} of {len(events)} candidates discarded: scoreboard not visible "
                         "(replay / crowd shot / graphic / ad).")
        selected = [e for e in events if e["ocr"]]
        if not selected:
            prog.empty()
            st.session_state["results"] = dict(events=events, merged=[], out=None, notes=notes,
                                               curve=_curve_preview(curve), duration=duration,
                                               total_peaks=total_peaks)
            st.warning("No audio peak was confirmed by the scoreboard. Check the ROI (Tab 1), "
                       "lower the OCR 'min runs increase', or disable OCR in Advanced.")
            return
    else:
        selected = events
    prog.progress(0.55, text=f"② {len(selected)} events selected")

    # ---- Stage 3: CNN run-up alignment ----------------------------------- #
    model, minfo = None, None
    if model_path:
        try:
            model, minfo = cached_model(model_path, os.path.getmtime(model_path))
            notes += [f"ℹ️ Model loaded via **{minfo.mode}**, input {minfo.input_size}."] + \
                     [f"⚠️ {w}" for w in minfo.warnings]
        except Exception as e:  # noqa: BLE001
            notes.append(f"⚠️ Run-up model unavailable ({e}). Falling back to audio + "
                         f"{fallback_lead:.1f}s fallback lead.")
    else:
        notes.append("⚠️ No .h5 model provided — using audio + fallback lead.")

    cap = cv2.VideoCapture(video_path)
    try:
        for i, ev in enumerate(selected):
            prog.progress(0.55 + 0.30 * i / len(selected),
                          text=f"③ Run-up alignment {i + 1}/{len(selected)}")
            start, conf = None, None
            if model is not None:
                try:
                    times, probs = classifier.score_window(cap, model, minfo.input_size, norm,
                                                           ev["t"], runup_lookback, 2.0)
                    start, conf = classifier.locate_runup_start(times, probs, runup_thr)
                except Exception as e:  # noqa: BLE001
                    notes.append(f"⚠️ CNN prediction failed ({type(e).__name__}: {str(e)[:120]}); "
                                 "using fallback lead for all remaining events.")
                    model = None
            ev["conf"] = conf
            if start is not None:
                ev["rs"], ev["source"] = start, "CNN"
            else:
                ev["rs"], ev["source"] = max(0.0, ev["t"] - fallback_lead), "Fallback"
    finally:
        cap.release()

    # ---- Stage 4: dedupe + render ---------------------------------------- #
    prog.progress(0.86, text="④ Merging intervals (de-duplication)…")
    raws = merger.build_raw_intervals([e["t"] for e in selected], [e["rs"] for e in selected],
                                      duration, lead_buffer, post_roll)
    merged = merger.merge_intervals(raws, merge_gap, min_event_duration, duration)
    for ev, raw in zip(selected, raws):
        ev["window"] = merger.window_for(raw, merged)

    out_path = os.path.join(workdir(), f"highlights_{int(time.time())}.mp4")
    prog.progress(0.88, text=f"⑤ Rendering {len(merged)} clips (each cut once)…")
    try:
        used = merger.render_reel(video_path, out_path, merged, engine,
                                  progress_cb=lambda f: prog.progress(0.88 + 0.11 * f,
                                                                      text=f"⑤ Rendering… {f:.0%}"))
        notes.append(f"ℹ️ Rendered with **{used}**.")
    except Exception as e:  # noqa: BLE001
        out_path = None
        st.error(f"Rendering failed: {e}")
    prog.progress(1.0, text="Done ✅")

    st.session_state["results"] = dict(events=events, merged=merged, out=out_path, notes=notes,
                                       curve=_curve_preview(curve), duration=duration,
                                       total_peaks=total_peaks)


def _curve_preview(curve: audio.AudioCurve) -> pd.DataFrame:
    step = max(1, len(curve.times) // 4000)
    return pd.DataFrame({"Smoothed RMS": curve.smooth[::step], "Z-score": curve.z[::step]},
                        index=pd.Index(curve.times[::step].round(1), name="seconds"))


# =========================================================================== #
# Tab 2 - processing & diagnostics
# =========================================================================== #
with tab2:
    st.write(f"k = **{k}** · run-up threshold = **{runup_thr}** · "
             f"OCR {'on' if use_ocr else 'off'} · model {'found' if model_path else 'missing'}")
    if st.button("▶️ Run highlight pipeline", type="primary"):
        try:
            run_pipeline()
        except Exception as e:  # noqa: BLE001
            st.exception(e)

    res = st.session_state.get("results")
    if res:
        for n in res["notes"]:
            st.markdown(n)
        merged = res["merged"]
        reel_len = sum(e - s for s, e in merged)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Audio peaks", res["total_peaks"])
        m2.metric("Events used", sum(1 for e in res["events"] if e.get("window")) or 0)
        m3.metric("Final clips (no overlaps)", len(merged))
        m4.metric("Reel length", merger.fmt_time(reel_len))

        rows = []
        for e in res["events"]:
            w = e.get("window")
            rows.append({
                "Timestamp": f"{merger.fmt_time(e['t'])} ({e['t']:.1f}s)",
                "Peak Z-Score": round(e["z"], 2),
                "OCR Confirmed": "Skipped" if e["ocr"] is None else ("Yes" if e["ocr"] else "No"),
                "OCR note": e["ocr_reason"],
                "Run-Up Confidence": None if e["conf"] is None else round(e["conf"], 2),
                "Run-up source": e["source"] or "—",
                "Final Merged Start": merger.fmt_time(w[0]) if w else "—",
                "Final Merged End": merger.fmt_time(w[1]) if w else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with st.expander("Audio energy & Z-score"):
            st.line_chart(res["curve"])

# =========================================================================== #
# Tab 3 - final reel
# =========================================================================== #
with tab3:
    res = st.session_state.get("results")
    if res and res.get("out") and os.path.exists(res["out"]):
        st.video(res["out"])
        with open(res["out"], "rb") as f:
            st.download_button("⬇️ Download Highlight MP4", f, file_name="cricket_highlights.mp4",
                               mime="video/mp4", type="primary")
        with st.expander("Clip list"):
            st.dataframe(pd.DataFrame(
                [{"#": i + 1, "Start": merger.fmt_time(s), "End": merger.fmt_time(e),
                  "Length (s)": round(e - s, 1)} for i, (s, e) in enumerate(res["merged"])]),
                hide_index=True, use_container_width=True)
    else:
        st.info("Run the pipeline in Tab 2 to render your highlight reel.")
