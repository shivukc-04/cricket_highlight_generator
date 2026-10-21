"""Modality A - audio excitement detection (crowd roars, commentary shouts).

Pipeline: video -> temp WAV (ffmpeg) -> 100 ms RMS windows -> moving-average
smoothing -> first difference -> Z-score -> peaks above threshold ``k``.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

TARGET_SR = 16000          # mono 16 kHz is plenty for energy analysis
WINDOW_SEC = 0.1           # 100 ms RMS windows


@dataclass
class AudioCurve:
    window_sec: float
    times: np.ndarray      # start time (s) of every window
    rms: np.ndarray        # raw RMS energy
    smooth: np.ndarray     # moving-average-smoothed RMS
    z: np.ndarray          # Z-score of the first difference of `smooth`


@dataclass
class AudioPeak:
    time: float
    z: float
    rms: float


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def extract_wav(video_path: str, wav_path: str, sr: int = TARGET_SR) -> None:
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-i", video_path, "-vn",
           "-ac", "1", "-ar", str(sr), "-acodec", "pcm_s16le", wav_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(wav_path):
        raise RuntimeError(f"Audio extraction failed: {proc.stderr.strip()[:500]}")


def _rms_windows(wav_path: str, window_sec: float) -> np.ndarray:
    """Block-wise RMS so a 3-hour match never has to sit in RAM as float32."""
    sr, data = wavfile.read(wav_path, mmap=True)
    if data.ndim > 1:
        data = data[:, 0]
    hop = max(1, int(round(sr * window_sec)))
    n = len(data) // hop
    if n == 0:
        raise RuntimeError("Audio track is empty or too short.")
    scale = 32768.0 if data.dtype == np.int16 else 1.0
    try:
        import librosa  # optional
    except Exception:
        librosa = None

    rms = np.empty(n, dtype=np.float32)
    block = 2000  # windows per block
    for s in range(0, n, block):
        e = min(n, s + block)
        chunk = np.asarray(data[s * hop:e * hop], dtype=np.float32) / scale
        if librosa is not None:
            rms[s:e] = librosa.feature.rms(
                y=chunk, frame_length=hop, hop_length=hop, center=False)[0][: e - s]
        else:
            rms[s:e] = np.sqrt(np.mean(chunk.reshape(-1, hop) ** 2, axis=1))
    del data
    return rms


def compute_audio_curve(video_path: str, window_sec: float = WINDOW_SEC,
                        smooth_sec: float = 0.5) -> AudioCurve:
    """Extract audio to a temp WAV (always deleted) and return the Z-score curve."""
    with tempfile.TemporaryDirectory(prefix="cricket_audio_") as td:
        wav = os.path.join(td, "audio.wav")
        extract_wav(video_path, wav)
        rms = _rms_windows(wav, window_sec)   # memmap is released inside

    w = max(1, int(round(smooth_sec / window_sec)))
    smooth = np.convolve(rms, np.ones(w) / w, mode="same").astype(np.float32)
    diff = np.diff(smooth, prepend=smooth[0])
    z = (diff - diff.mean()) / (diff.std() + 1e-9)
    times = np.arange(len(rms), dtype=np.float64) * window_sec
    return AudioCurve(window_sec, times, rms, smooth, z.astype(np.float32))


def detect_peaks(curve: AudioCurve, k: float = 2.0, min_gap_sec: float = 10.0,
                 volume_gate: bool = True) -> List[AudioPeak]:
    """Flag sudden audio spikes whose first-difference Z-score exceeds ``k``.

    ``min_gap_sec`` keeps only the strongest peak inside any window of that
    length; ``volume_gate`` drops spikes that happen while the absolute volume
    is below the match average (random quiet noises).
    """
    dist = max(1, int(round(min_gap_sec / curve.window_sec)))
    idx, _ = find_peaks(curve.z, height=k, distance=dist)
    mean_vol = float(curve.smooth.mean())
    peaks = []
    for i in idx:
        if volume_gate and curve.smooth[i] < mean_vol:
            continue
        peaks.append(AudioPeak(float(curve.times[i]), float(curve.z[i]), float(curve.smooth[i])))
    return peaks
