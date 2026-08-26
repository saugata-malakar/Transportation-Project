"""
stage1_video_audit.py
Section 3.1 Video Audit

Probes video resolution, FPS, duration, etc.
Samples ~5 evenly-spaced frames to estimate average person bounding box size.
Computes camera motion metric via optical flow variance.
Assesses lighting conditions (brightness, contrast).
Outputs an audit report with sampling rate recommendation.
"""

import argparse
import json
import os
import subprocess
import cv2
import numpy as np

from config import DETECTION

def get_video_info(video_path):
    info = {
        "resolution": None,
        "fps": None,
        "duration_sec": None,
        "total_frames": None,
        "codec": None,
        "file_size_mb": None,
    }
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return info

    info["file_size_mb"] = os.path.getsize(video_path) / (1024 * 1024)

    # Try ffprobe
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames,codec_name",
            "-of", "json", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            probe = json.loads(result.stdout)
            stream = probe.get("streams", [{}])[0]
            
            info["resolution"] = (stream.get("width"), stream.get("height"))
            info["codec"] = stream.get("codec_name")
            
            fps_str = stream.get("r_frame_rate", "0/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                info["fps"] = float(num) / float(den) if float(den) != 0 else 0
            else:
                info["fps"] = float(fps_str)
                
            info["duration_sec"] = float(stream.get("duration", 0)) if stream.get("duration") else None
            info["total_frames"] = int(stream.get("nb_frames", 0)) if stream.get("nb_frames") else None
    except Exception as e:
        print(f"ffprobe not available or failed: {e}")

    # Fallback or complete with cv2
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        if info["resolution"] == (None, None) or info["resolution"][0] is None:
            info["resolution"] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not info["fps"]:
            info["fps"] = cap.get(cv2.CAP_PROP_FPS)
        if not info["total_frames"]:
            info["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not info["duration_sec"] and info["fps"] and info["fps"] > 0:
            info["duration_sec"] = info["total_frames"] / info["fps"]
        cap.release()

    return info

def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return frames
        
    step = max(1, total_frames // num_frames)
    indices = [min(i * step, total_frames - 1) for i in range(num_frames)]
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def estimate_person_size(frames):
    if not frames:
        return 0, 0
    from stage3_detection import _load_model
    model = _load_model(DETECTION.model_name)
    
    widths = []
    heights = []
    
    for frame in frames:
        results = model.predict(frame, classes=[0], verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            widths.append(x2 - x1)
            heights.append(y2 - y1)
            
    if not widths:
        return 0, 0
        
    return sum(widths) / len(widths), sum(heights) / len(heights)

def analyze_motion_and_lighting(frames):
    if len(frames) < 2:
        return 0.0, 0.0, 0.0
        
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    
    flow_variances = []
    brightnesses = []
    contrasts = []
    
    for i in range(len(gray_frames) - 1):
        prev = gray_frames[i]
        curr = gray_frames[i+1]
        
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_variances.append(np.var(mag))
        
    for gray in gray_frames:
        brightnesses.append(np.mean(gray))
        contrasts.append(np.std(gray))
        
    avg_motion_var = float(sum(flow_variances) / len(flow_variances)) if flow_variances else 0.0
    avg_brightness = float(sum(brightnesses) / len(brightnesses)) if brightnesses else 0.0
    avg_contrast = float(sum(contrasts) / len(contrasts)) if contrasts else 0.0
    
    return avg_motion_var, avg_brightness, avg_contrast

def main():
    parser = argparse.ArgumentParser(description="Section 3.1 - Video Audit")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out_dir", default="work", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Auditing video: {args.video}")
    info = get_video_info(args.video)
    
    frames = extract_frames(args.video, num_frames=5)
    
    avg_w, avg_h = estimate_person_size(frames)
    motion_var, brightness, contrast = analyze_motion_and_lighting(frames)
    
    motion_threshold = 5.0 # Example threshold for motion
    rec_rate = "1 frame/1s" if motion_var > motion_threshold else "1 frame/2s"
    
    report = {
        "video_info": info,
        "metrics": {
            "avg_person_width_px": round(avg_w, 2),
            "avg_person_height_px": round(avg_h, 2),
            "motion_variance": round(motion_var, 2),
            "avg_brightness": round(brightness, 2),
            "avg_contrast": round(contrast, 2),
        },
        "recommendations": {
            "sampling_rate": rec_rate
        }
    }
    
    out_file = os.path.join(args.out_dir, "audit_report.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Audit report saved to {out_file}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
