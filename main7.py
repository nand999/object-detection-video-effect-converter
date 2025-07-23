#!/usr/bin/env python3
"""main.py — overlay beat-synced squares and connecting lines on a video.

Usage example:
    python main.py -i sample_data/playing_dead.mp4 -o output_with_boxes.mp4
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import tempfile
from pathlib import Path
import uuid

import cv2
import librosa
import moviepy.editor as mpy
import numpy as np
import torch  # For YOLOv5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_audio(video_path: Path, sr: int = 22050) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / "temp_audio.wav"
    clip = mpy.VideoFileClip(str(video_path))
    clip.audio.write_audiofile(str(wav_path), fps=sr, logger=None, verbose=False)
    return wav_path


def _detect_onsets(wav_path: Path, sr: int = 22050) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sr)
    return librosa.onset.onset_detect(y=y, sr=sr, units="time")


class Square:
    def __init__(self, born_at: float, x: int, y: int, size: int, idx: int):
        self.born_at = born_at
        self.x = x
        self.y = y
        self.size = size
        self.idx = idx

    def age(self, now: float) -> float:
        return now - self.born_at


class TrackedPoint:
    def __init__(
        self,
        pos: tuple[float, float],
        life: int,
        size: int,
        label: str,
        font_scale: float,
        text_color: tuple[int, int, int],
        vertical: bool,
    ):
        self.pos = np.array(pos, dtype=np.float32)
        self.life = life
        self.size = size
        self.label = label
        self.font_scale = font_scale
        self.text_color = text_color
        self.vertical = vertical


model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model_yolo.eval()


def render_tracked_effect(
    video_in: Path,
    video_out: Path,
    *,
    fps: float | None,
    pts_per_beat: int,
    ambient_rate: float,
    jitter_px: float,
    life_frames: int,
    min_size: int,
    max_size: int,
    neighbor_links: int,
    orb_fast_threshold: int,
    bell_width: float,
    seed: int | None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    clip = mpy.VideoFileClip(str(video_in))
    if fps is None:
        fps = clip.fps

    wav_path = _extract_audio(video_in)
    onset_times = _detect_onsets(wav_path)
    logging.info("%d onsets detected", len(onset_times))

    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=orb_fast_threshold)

    active: list[TrackedPoint] = []
    onset_idx = 0
    prev_gray: np.ndarray | None = None

    def make_frame(t: float):
        nonlocal prev_gray, onset_idx, active
        frame = clip.get_frame(t).copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        f_idx = int(round(t * fps))

        # Object detection (put first)
        results = model_yolo(frame)
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            label = model_yolo.names[int(cls)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Track existing points
        if prev_gray is not None and active:
            prev_pts = np.array([p.pos for p in active], dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=(21, 21), maxLevel=3)
            new_active: list[TrackedPoint] = []
            for tp, new_pt, ok in zip(active, next_pts.reshape(-1, 2), status.reshape(-1)):
                if not ok:
                    continue
                x, y = new_pt
                if 0 <= x < w and 0 <= y < h and tp.life > 0:
                    tp.pos = new_pt
                    tp.life -= 1
                    if jitter_px > 0:
                        tp.pos += np.random.normal(0, jitter_px, size=2)
                        tp.pos[0] = np.clip(tp.pos[0], 0, w - 1)
                        tp.pos[1] = np.clip(tp.pos[1], 0, h - 1)
                    new_active.append(tp)
            active = new_active

        # Beat-synced spawning
        while onset_idx < len(onset_times) and t >= onset_times[onset_idx]:
            kps = orb.detect(gray, None)
            kps = sorted(kps, key=lambda k: k.response, reverse=True)
            target_spawn = random.randint(1, pts_per_beat)
            spawned = 0
            for kp in kps:
                if spawned >= target_spawn:
                    break
                x, y = kp.pt
                if any(np.linalg.norm(tp.pos - (x, y)) < 10 for tp in active):
                    continue
                size = _sample_size_bell(min_size, max_size, bell_width)
                r = random.random()
                if r < 0.33:
                    label = ''.join(random.choices('ABCDEF0123456789', k=6))
                elif r < 0.66:
                    label = str(random.randint(1, 999))
                else:
                    label = str(uuid.uuid4())[:8]

                font_scale = random.uniform(1.0, 1.8)
                text_color = (255, 255, 255)
                vertical = random.random() < 0.25

                active.append(TrackedPoint((x, y), life_frames, size, label, font_scale, text_color, vertical))
                spawned += 1
            logging.info("Spawned %d points", spawned)
            onset_idx += 1

        # Ambient random noise
        if ambient_rate > 0:
            noise_n = np.random.poisson(ambient_rate / fps)
            for _ in range(noise_n):
                x = random.uniform(0, w)
                y = random.uniform(0, h)
                size = _sample_size_bell(min_size, max_size, bell_width)
                label_choices = [
                    ''.join(random.choices('ABCDEF0123456789', k=6)),
                    str(random.randint(1, 999)),
                    str(uuid.uuid4())[:8],
                ]
                label = random.choice(label_choices)
                font_scale = random.uniform(1.0, 1.8)
                text_color = (255, 255, 255)
                vertical = random.random() < 0.25
                active.append(TrackedPoint((x, y), life_frames, size, label, font_scale, text_color, vertical))

        # Draw edges between points
        coords = [tp.pos for tp in active]
        for i, p in enumerate(coords):
            dists = [(j, np.linalg.norm(p - coords[j])) for j in range(len(coords)) if j != i]
            dists.sort(key=lambda x: x[1])
            for j, _ in dists[:neighbor_links]:
                cv2.line(frame, tuple(p.astype(int)), tuple(coords[j].astype(int)), (255, 255, 255), 1)

        # Draw beat squares
        for tp in active:
            x, y = tp.pos
            s = tp.size
            tl = (int(x - s // 2), int(y - s // 2))
            br = (int(x + s // 2), int(y + s // 2))
            roi = frame[tl[1]:br[1], tl[0]:br[0]]
            if roi.size:
                frame[tl[1]:br[1], tl[0]:br[0]] = 255 - roi
            cv2.rectangle(frame, tl, br, (255, 255, 255), 1)
            if tp.vertical:
                y_cursor = tl[1] + 2
                line_height = int(12 * tp.font_scale)
                for ch in tp.label:
                    cv2.putText(frame, ch, (tl[0] + 2, y_cursor), cv2.FONT_HERSHEY_PLAIN, tp.font_scale, tp.text_color, 1, cv2.LINE_AA)
                    y_cursor += line_height
                    if y_cursor > br[1] - 2:
                        break
            else:
                cv2.putText(frame, tp.label, (tl[0] + 2, br[1] - 4), cv2.FONT_HERSHEY_PLAIN, tp.font_scale, tp.text_color, 1, cv2.LINE_AA)

        prev_gray = gray
        return frame

    out_clip = mpy.VideoClip(make_frame, duration=clip.duration)
    out_clip = out_clip.set_audio(clip.audio)
    out_clip.write_videofile(str(video_out), fps=fps, codec="libx264", audio_codec="aac")


def _sample_size_bell(min_s: int, max_s: int, width_div: float = 6.0) -> int:
    mean = (min_s + max_s) / 2.0
    sigma = (max_s - min_s) / width_div
    for _ in range(10):
        val = np.random.normal(mean, sigma)
        if min_s <= val <= max_s:
            return int(val)
    return int(np.clip(val, min_s, max_s))


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", type=Path, default=Path("sample_data/vid2.mp4"))
    p.add_argument("-o", "--output", type=Path, default=Path("main7_1.mp4"))
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--life-frames", type=int, default=10)
    p.add_argument("--pts-per-beat", type=int, default=20)
    p.add_argument("--ambient-rate", type=float, default=5.0)
    p.add_argument("--jitter-px", type=float, default=0.5)
    p.add_argument("--min-size", type=int, default=70)
    p.add_argument("--max-size", type=int, default=90)
    p.add_argument("--neighbor-links", type=int, default=3)
    p.add_argument("--orb-fast-threshold", type=int, default=20)
    p.add_argument("--bell-width", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s:%(message)s")
    render_tracked_effect(
        video_in=args.input,
        video_out=args.output,
        fps=args.fps,
        pts_per_beat=args.pts_per_beat,
        ambient_rate=args.ambient_rate,
        jitter_px=args.jitter_px,
        life_frames=args.life_frames,
        min_size=args.min_size,
        max_size=args.max_size,
        neighbor_links=args.neighbor_links,
        orb_fast_threshold=args.orb_fast_threshold,
        bell_width=args.bell_width,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
