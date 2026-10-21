"""Interval de-duplication (2-pass union merge) and single-pass rendering."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Sequence, Tuple

from .audio import ffmpeg_exe

Interval = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Interval logic
# --------------------------------------------------------------------------- #
def build_raw_intervals(event_times: Sequence[float], runup_starts: Sequence[float],
                        duration: float, lead_buffer: float, post_roll: float) -> List[Interval]:
    """Start_i = max(0, RunUp_i - lead_buffer); End_i = min(duration, T_i + post_roll)."""
    raw = []
    for t, rs in zip(event_times, runup_starts):
        s = max(0.0, rs - lead_buffer)
        e = min(duration, t + post_roll)
        if e > s:
            raw.append((s, e))
    return raw


def _union(intervals: Sequence[Interval], gap: float) -> List[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: x[0])
    out = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s <= out[-1][1] + gap:                 # Start_{i+1} <= End_i + merge_gap
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def merge_intervals(intervals: Sequence[Interval], merge_gap: float = 4.0,
                    min_duration: float = 5.0, duration: Optional[float] = None) -> List[Interval]:
    """Pass 1: sort + union merge. Pass 2: enforce min duration and re-merge until
    stable, because lengthening a short window can make it touch its neighbour."""
    merged = _union(intervals, merge_gap)
    for _ in range(len(merged) + 2):
        grown = []
        for s, e in merged:
            if e - s < min_duration:
                e = s + min_duration
                if duration is not None and e > duration:
                    e = duration
                    s = max(0.0, e - min_duration)
            grown.append((s, e))
        new = _union(grown, merge_gap)
        if new == merged:
            break
        merged = new
    assert_disjoint(merged)
    return merged


def assert_disjoint(intervals: Sequence[Interval]) -> None:
    for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
        if s2 <= e1:
            raise ValueError(f"Overlapping clips after merge: {(s1, e1)} / {(s2, e2)}")


def window_for(raw: Interval, merged: Sequence[Interval]) -> Optional[Interval]:
    mid = (raw[0] + raw[1]) / 2
    for s, e in merged:
        if s - 1e-6 <= mid <= e + 1e-6:
            return (s, e)
    return None


def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _import_moviepy():
    try:                                    # MoviePy 1.x
        from moviepy.editor import VideoFileClip, concatenate_videoclips
    except Exception:                       # MoviePy 2.x
        from moviepy import VideoFileClip, concatenate_videoclips
    return VideoFileClip, concatenate_videoclips


def moviepy_available() -> bool:
    try:
        _import_moviepy()
        return True
    except Exception:
        return False


def _subclip(clip, s, e):
    return clip.subclipped(s, e) if hasattr(clip, "subclipped") else clip.subclip(s, e)


def render_moviepy(src: str, out: str, intervals: Sequence[Interval], tmpdir: str) -> None:
    VideoFileClip, concatenate_videoclips = _import_moviepy()
    video, subs, final = None, [], None
    try:
        video = VideoFileClip(src)
        dur = video.duration
        subs = [_subclip(video, s, min(e, dur)) for s, e in intervals]
        final = concatenate_videoclips(subs)
        final.write_videofile(
            out, codec="libx264", audio_codec="aac", preset="veryfast",
            temp_audiofile=os.path.join(tmpdir, "mp_tmp_audio.m4a"),
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            logger=None)
    finally:                                 # release handles immediately (Windows)
        for c in [final, *subs, video]:
            try:
                if c is not None:
                    c.close()
            except Exception:
                pass


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.strip()[-600:]}")


def render_ffmpeg(src: str, out: str, intervals: Sequence[Interval], tmpdir: str,
                  progress_cb=None) -> None:
    """Cut every window exactly once (re-encoded, uniform params), then concat-copy."""
    exe = ffmpeg_exe()
    parts = []
    try:
        for i, (s, e) in enumerate(intervals):
            p = os.path.join(tmpdir, f"part_{i:04d}.mp4")
            _run([exe, "-y", "-v", "error", "-ss", f"{s:.3f}", "-i", src, "-t", f"{e - s:.3f}",
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                  "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2", p])
            parts.append(p)
            if progress_cb:
                progress_cb((i + 1) / len(intervals))
        lst = os.path.join(tmpdir, "concat.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for p in parts:
                f.write("file '{}'\n".format(p.replace("\\", "/").replace("'", r"'\''")))
        _run([exe, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
              "-c", "copy", "-movflags", "+faststart", out])
    finally:
        for p in parts:
            try:
                os.remove(p)
            except OSError:
                pass


def render_reel(src: str, out: str, intervals: Sequence[Interval], engine: str = "auto",
                progress_cb=None) -> str:
    """Render the non-overlapping intervals into one MP4. Returns the engine used."""
    if not intervals:
        raise ValueError("No intervals to render.")
    assert_disjoint(intervals)
    if engine == "auto":
        engine = "moviepy" if moviepy_available() else "ffmpeg"
    tmp = tempfile.mkdtemp(prefix="cricket_render_")
    try:
        if engine == "moviepy":
            render_moviepy(src, out, intervals, tmp)
        else:
            render_ffmpeg(src, out, intervals, tmp, progress_cb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return engine
