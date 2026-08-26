"""
stage3_visual_detection_inspection.py
Visual human detection demonstration on drone frames with high-resolution inference.

Produces:
  1. Annotated full-resolution images with crisp pedestrian bounding boxes & confidences
  2. Multi-crop zoom panels showing detected humans side-by-side
  3. Detection statistics (bounding box sizes, aspect ratios, confidence distribution)
  4. Saves to work/detections/visual_inspections/
"""

import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

from config import DETECTION, PATHS


def run_visual_detection(model_name="yolov8m.pt",
                         manifest_path="work/frames_manifest.json",
                         out_dir="work/detections/visual_inspections",
                         conf_threshold=0.25,
                         imgsz=1280,
                         num_frames_to_inspect=6):
    os.makedirs(out_dir, exist_ok=True)
    zoom_dir = os.path.join(out_dir, "pedestrian_zooms")
    os.makedirs(zoom_dir, exist_ok=True)

    print(f"[Detector Visual Inspection] Loading {model_name}...")
    model = YOLO(model_name)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Pick evenly spaced representative frames across the video
    all_frames = manifest["frames"]
    step = max(1, len(all_frames) // num_frames_to_inspect)
    sample_frames = [all_frames[i] for i in range(0, len(all_frames), step)][:num_frames_to_inspect]

    total_detected = 0
    inspection_report = []

    print(f"[Detector Visual Inspection] Running high-resolution detection (imgsz={imgsz}, conf={conf_threshold})...\n")

    for f_idx, fr in enumerate(sample_frames):
        img_path = fr["file"]
        ts = fr["timestamp_sec"]
        frame_id = fr["frame_idx"]

        if not os.path.exists(img_path):
            continue

        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            continue
        h_orig, w_orig = frame_bgr.shape[:2]

        # Run high-res detection specifically for class 0 (person)
        results = model.predict(img_path, conf=conf_threshold, imgsz=imgsz, classes=[0], verbose=False)[0]
        boxes = results.boxes

        annotated_img = frame_bgr.copy()
        person_crops = []

        for b_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            total_detected += 1

            # Clamp coordinates
            x1 = max(0, min(w_orig - 1, x1))
            y1 = max(0, min(h_orig - 1, y1))
            x2 = max(x1 + 1, min(w_orig, x2))
            y2 = max(y1 + 1, min(h_orig, y2))

            bw = x2 - x1
            bh = y2 - y1

            # Save individual pedestrian crop zoom
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (120, 240))
                # Add banner with confidence
                banner = np.zeros((30, 120, 3), dtype=np.uint8)
                cv2.putText(banner, f"Conf: {conf:.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                crop_card = np.vstack([banner, crop_resized])
                person_crops.append(crop_card)

            # Draw glowing bounding box
            color = (0, 255, 0) if conf > 0.5 else (0, 200, 255)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)

            # Label badge
            label = f"Person {conf:.2f} [{bw}x{bh}px]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_img, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(annotated_img, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Top summary banner on the full image
        top_banner = f"Frame {frame_id} (t = {ts:.1f}s) | Detected: {len(boxes)} Humans | Resolution: {w_orig}x{h_orig}"
        cv2.rectangle(annotated_img, (20, 20), (1100, 80), (0, 0, 0), -1)
        cv2.putText(annotated_img, top_banner, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        out_img_path = os.path.join(out_dir, f"inspected_frame_{frame_id:07d}_t{ts:06.1f}s.jpg")
        cv2.imwrite(out_img_path, annotated_img)

        # If person crops were found, create a combined zoom panel for this frame
        if person_crops:
            max_cols = 8
            # Group into rows of max_cols
            rows = []
            for k in range(0, len(person_crops), max_cols):
                chunk = person_crops[k:k+max_cols]
                # Pad row if needed
                while len(chunk) < max_cols:
                    chunk.append(np.zeros_like(person_crops[0]))
                rows.append(np.hstack(chunk))
            zoom_panel = np.vstack(rows)
            zoom_panel_path = os.path.join(zoom_dir, f"zooms_frame_{frame_id:07d}.jpg")
            cv2.imwrite(zoom_panel_path, zoom_panel)

        print(f"  Frame {frame_id:05d} (t={ts:6.1f}s): {len(boxes):2d} pedestrians detected -> {out_img_path}")
        inspection_report.append({
            "frame_id": frame_id,
            "timestamp_sec": ts,
            "detections_count": len(boxes),
            "output_image": out_img_path
        })

    print(f"\n[Detector Visual Inspection] Complete! Generated {len(sample_frames)} inspected frames.")
    return inspection_report


if __name__ == "__main__":
    run_visual_detection()
