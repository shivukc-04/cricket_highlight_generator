# Automated Multimodal IR Highlight Generator (Cricket Highlight Engine)

An automated video summarization engine that correlates audio spectrogram spikes (**Librosa**) with visual computer vision cues (**custom CNNs**) to detect crowd cheer peaks and player action events from raw video recordings, stitching them into frame-accurate, synchronized 1080p highlight reels.

---

## Key Features

* **Multimodal Feature Fusion**: Combines audio signal processing (decibel/FFT analysis) and visual computer vision detection (CNN feature extraction).
* **Audio Peak Detection**: Identifies crowd cheering noise spikes and commentary energy bursts using short-time Fourier transforms (STFT) in Librosa.
* **Visual Cue Recognition**: Detects key visual actions such as bowler run-ups, boundary signals, and player celebration events.
* **Peak Correlation & False Positive Filtering**: Heuristic interval-merging algorithm filters out non-event audio spikes (e.g., loud studio commentary or commercials) by requiring cross-modal alignment.
* **Temporal Interval Merging**: Dynamically merges adjacent and overlapping candidate clips to produce smooth, non-choppy video transitions.
* **Automated Video Assembly**: Frame-accurate video extraction and audio-video multiplexing using FFmpeg and OpenCV.

---

## System Architecture & Workflow

```
+-------------------------------------------------------------------+
|                        Raw Video Recording                        |
+-------------------------------------------------------------------+
                                  |
                +-----------------+-----------------+
                |                                   |
                v                                   v
    [ Audio Stream Extraction ]         [ Visual Frame Processing ]
                |                                   |
                v                                   v
     (Librosa Spectrogram /             (Custom CNN Feature
      FFT Decibel Peak Detection)         Action Cue Detection)
                |                                   |
                +-----------------+-----------------+
                                  |
                                  v
              [ Multimodal Correlation Engine ]
              - Thresholding & Peak Alignment
              - False-Positive Noise Filtering
                                  |
                                  v
              [ Temporal Interval Merging ]
              - Overlap Union & Buffer Insertion
                                  |
                                  v
              [ Video Multiplexing & Stitching ]
              (FFmpeg / OpenCV Frame Reconstruction)
                                  |
                                  v
             +-----------------------------------------+
             | Synchronized 1080p Highlight Reel Output |
             +-----------------------------------------+
```

---

## Tech Stack & Dependencies

* **Programming Language**: Python 3.10+
* **Audio Signal Processing**: `librosa`, `scipy`, `numpy`
* **Computer Vision & Deep Learning**: `opencv-python`, `torch` / `torchvision` (Custom CNNs)
* **Video Encoding & Processing**: `ffmpeg-python`, `moviepy`
* **Visualization & Analytics**: `matplotlib`

---

## Project Structure

```
Cricket-Highlight-Generator/
├── data/
│   ├── raw_videos/                     # Source full-match video recordings
│   └── output_highlights/              # Output summarized highlight reels
├── models/
│   └── cnn_action_detector.pth         # Pre-trained CNN weights for action cues
├── src/
│   ├── audio_analysis.py               # Librosa spectrogram & audio peak detector
│   ├── visual_analysis.py              # Frame extraction & CNN inference pipeline
│   ├── interval_merger.py              # Temporal clip merging & heuristic filter
│   └── video_stitcher.py              # FFmpeg video reconstruction module
├── generate_highlights.py             # Main pipeline CLI entry point
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

---

## Setup & Running Guide

### Prerequisites
1. **Python 3.10+** installed.
2. **FFmpeg** installed on your system path.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Highlight Generation Pipeline
Execute the main pipeline on a match video:
```bash
python generate_highlights.py     --input data/raw_videos/match_recording.mp4     --output data/output_highlights/match_highlights.mp4     --audio-threshold 0.75     --min-clip-duration 5     --max-clip-duration 15
```

---

## Performance Characteristics

* **Summarization Speed**: Processes a full 3-hour match in minutes through frame sampling and parallelized audio FFT extraction.
* **Frame Accuracy**: Audio-video multiplexing maintains exact frame sync down to the millisecond.
* **Storage Optimization**: Reduces video file size by ~85-90% while preserving all key match events.
