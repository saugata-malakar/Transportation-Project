"""
stage2_frame_sampling.py
Section 3.2 Frame Sampling + Section 3.3 Preparation Workflow

Given the raw video, extracts frames at the recommended rate:
  - Baseline: 1 frame / 2s
  - Fast-changing scenes: 1 fps, applied locally around detected scene changes
  - Refinement: near-duplicate frames removed via perceptual hashing

Also performs the video-segment-level split described in 3.3, so that
adjacent frames of the same scene/person never cross train/test.

IMPORTANT — real drone footage note:
DJI (and most consumer drone) exports are frequently variable-frame-rate
(VFR), and `cv2`'s `CAP_PROP_POS_FRAMES` seeking is unreliable on VFR/
long-GOP H.264 files: `cap.set(POS_FRAMES, n); cap.read()` can silently
land on the wrong frame. This version does a SINGLE SEQUENTIAL DECODE
PASS over the whole video instead of seeking — it reads every frame in
order, uses each frame's actual decoded timestamp, and makes sampling +
scene-change decisions inline. This is both more correct and, in
practice, not slower than repeated seeks (H.264 seeking has to decode
back to the last keyframe internally anyway).

Usage:
    python stage2_frame_sampling.py --video drone.mp4 --out_dir work/frames \
        --test_segment_frac 0.15
"""

import argparse
import json
import os
import subprocess

import cv2
import numpy as np


def phash(gray_small: np.ndarray) -> int:
    """Cheap perceptual hash (average hash) for near-duplicate detection."""
    resized = cv2.resize(gray_small, (16, 16), interpolation=cv2.INTER_AREA)
    avg = resized.mean()
    bits = (resized > avg).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def probe_duration_sec(video_path):
    """ffprobe fallback: cv2's CAP_PROP_FRAME_COUNT / CAP_PROP_FPS are
    frequently wrong or inconsistent on VFR drone footage. If ffprobe is
    available, prefer its container-level duration for sanity-checking."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def sample_frames(video_path, out_dir, baseline_interval=2.0, fast_interval=1.0,
                   fast_change_window=3.0, dedup_hamming_thresh=5,
                   scene_hist_thresh=0.3, scene_check_every_sec=1.0):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    reported_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_duration = reported_frame_count / reported_fps if reported_fps > 0 else 0.0

    ffprobe_duration = probe_duration_sec(video_path)
    duration_mismatch = (
        ffprobe_duration is not None and reported_duration > 0
        and abs(ffprobe_duration - reported_duration) / max(ffprobe_duration, 1) > 0.05
    )

    manifest = []
    scene_change_times = []
    last_hash = None
    last_scene_hist = None
    next_scene_check_t = 0.0
    next_sample_t = 0.0
    fast_window_until = -1.0  # timestamp until which we're in "fast" sampling mode
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prefer the decoder's own per-frame timestamp (robust to VFR);
        # fall back to frame_counter / reported_fps if POS_MSEC is unavailable.
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec and pos_msec > 0:
            t = pos_msec / 1000.0
        else:
            t = frame_counter / reported_fps
        frame_counter += 1

        gray = None  # computed lazily, only when needed below

        # --- scene-change check (runs on its own cadence, same pass) ---
        if t >= next_scene_check_t:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if last_scene_hist is not None:
                diff = cv2.compareHist(last_scene_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > scene_hist_thresh:
                    scene_change_times.append(round(t, 2))
                    fast_window_until = t + fast_change_window
            last_scene_hist = hist
            next_scene_check_t = t + scene_check_every_sec

        # --- sampling decision ---
        if t >= next_sample_t:
            in_fast_region = t <= fast_window_until
            interval = fast_interval if in_fast_region else baseline_interval

            if gray is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h = phash(gray)
            is_dup = last_hash is not None and hamming(h, last_hash) <= dedup_hamming_thresh

            if not is_dup:
                last_hash = h
                frame_idx_for_name = frame_counter - 1
                fname = f"frame_{frame_idx_for_name:07d}_t{t:08.2f}.jpg"
                fpath = os.path.join(out_dir, fname)
                cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                manifest.append({
                    "frame_idx": frame_idx_for_name,
                    "timestamp_sec": round(t, 2),
                    "file": fpath,
                    "is_fast_change_region": in_fast_region,
                })
            next_sample_t = t + interval

    cap.release()

    meta = {
        "reported_fps": round(reported_fps, 3),
        "reported_frame_count": reported_frame_count,
        "reported_duration_sec": round(reported_duration, 1),
        "ffprobe_duration_sec": round(ffprobe_duration, 1) if ffprobe_duration else None,
        "duration_mismatch_flagged": duration_mismatch,
        "frames_actually_decoded": frame_counter,
    }
    if duration_mismatch:
        print(f"WARNING: cv2-reported duration ({reported_duration:.1f}s) and ffprobe "
              f"duration ({ffprobe_duration:.1f}s) disagree by >5% — this container's "
              f"metadata may be unreliable (common with VFR drone exports). Sampling "
              f"used per-frame decoded timestamps, not the reported fps, so results "
              f"should still be correct, but double-check frame count below.")
    return manifest, scene_change_times, meta


def split_by_segment(manifest, test_fraction=0.15, segment_len_sec=20.0):
    """Section 3.3: split at the video-segment level, never by shuffling
    individual frames, to prevent identity/temporal leakage."""
    if not manifest:
        return manifest
    max_t = max(m["timestamp_sec"] for m in manifest)
    n_segments = max(int(max_t // segment_len_sec) + 1, 1)
    n_test_segments = max(int(round(n_segments * test_fraction)), 1)
    # Reserve the *last* contiguous segments as held-out test, matching the
    # doc's emphasis on a genuinely held-out, non-interleaved test portion.
    test_segment_ids = set(range(n_segments - n_test_segments, n_segments))

    for m in manifest:
        seg_id = int(m["timestamp_sec"] // segment_len_sec)
        m["segment_id"] = seg_id
        m["split"] = "test" if seg_id in test_segment_ids else "train"
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Section 3.2/3.3 — Frame sampling")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", default="work/frames")
    ap.add_argument("--manifest_out", default="work/frames_manifest.json")
    ap.add_argument("--baseline_interval", type=float, default=2.0)
    ap.add_argument("--fast_interval", type=float, default=1.0)
    ap.add_argument("--test_segment_frac", type=float, default=0.15)
    ap.add_argument("--segment_len_sec", type=float, default=20.0)
    args = ap.parse_args()

    manifest, scene_changes, meta = sample_frames(
        args.video, args.out_dir,
        baseline_interval=args.baseline_interval,
        fast_interval=args.fast_interval,
    )
    manifest = split_by_segment(manifest, args.test_segment_frac, args.segment_len_sec)

    os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)
    with open(args.manifest_out, "w") as f:
        json.dump({"num_frames_sampled": len(manifest),
                    "num_scene_changes_detected": len(scene_changes),
                    "video_meta": meta,
                    "frames": manifest}, f, indent=2)

    n_train = sum(1 for m in manifest if m["split"] == "train")
    n_test = sum(1 for m in manifest if m["split"] == "test")
    print(f"Decoded {meta['frames_actually_decoded']} raw frames sequentially, "
          f"sampled {len(manifest)} ({n_train} train / {n_test} test, split by video "
          f"segment). Scene changes detected: {len(scene_changes)}")
    print(f"Manifest -> {args.manifest_out}")


if __name__ == "__main__":
    main()
