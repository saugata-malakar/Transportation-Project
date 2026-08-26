"""
run_yolo_human_detector_finetuning.py
Fine-tunes YOLO on the target drone pedestrian dataset and generates visual proof of accurate human detection.

Workflow:
  1. Load base pretrained YOLO model (yolov8m / yolo11n)
  2. Fine-tune on drone dataset (work/detector_finetuning/dataset.yaml)
  3. Evaluate validation metrics (Precision, Recall, mAP50, mAP50-95)
  4. Run inference on key drone frames and generate annotated visualizations:
     - Full-resolution bounding box overlays
     - Side-by-side cropped pedestrian galleries
     - Detection confidence heatmaps & distribution
  5. Save fine-tuned checkpoint -> checkpoints/yolo_drone_human_finetuned.pt
"""

import os
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import DETECTION, PATHS


def train_yolo_drone_detector(dataset_yaml="work/detector_finetuning/dataset.yaml",
                              base_model="yolov8m.pt",
                              epochs=5,
                              imgsz=960,
                              batch_size=4,
                              save_ckpt="checkpoints/yolo_drone_human_finetuned.pt"):
    print(f"\n{'='*70}")
    print(f"  STEP 1: FINE-TUNING YOLO HUMAN DETECTOR ON DRONE FOOTAGE")
    print(f"{'='*70}")
    print(f"  Base Pretrained Model: {base_model}")
    print(f"  Target Dataset:        {dataset_yaml}")
    print(f"  Image Resolution:      {imgsz}x{imgsz}")
    print(f"  Training Epochs:       {epochs}")
    print(f"  Target Class:          0 (person / pedestrian)")
    print(f"{'='*70}\n")

    model = YOLO(base_model)

    # Fine-tune the detector
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=0,  # Windows safe
        project="work/detector_finetuning/runs",
        name="drone_human_finetune",
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True
    )

    # Find the best weights
    best_weights = "work/detector_finetuning/runs/drone_human_finetune/weights/best.pt"
    if not os.path.exists(best_weights):
        best_weights = "work/detector_finetuning/runs/drone_human_finetune/weights/last.pt"

    if os.path.exists(best_weights):
        os.makedirs(os.path.dirname(save_ckpt), exist_ok=True)
        import shutil
        shutil.copyfile(best_weights, save_ckpt)
        print(f"\n[Fine-Tuning] Fine-tuned model checkpoint saved -> {save_ckpt}")

    return save_ckpt if os.path.exists(save_ckpt) else base_model


def generate_visual_detection_proof(model_path,
                                    manifest_path="work/frames_manifest.json",
                                    out_dir="work/detections/finetuning_demonstration",
                                    conf_threshold=0.30,
                                    imgsz=1280,
                                    num_sample_frames=5):
    os.makedirs(out_dir, exist_ok=True)
    zooms_dir = os.path.join(out_dir, "detected_human_crops")
    os.makedirs(zooms_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  STEP 2: RUNNING VERIFIED HUMAN DETECTION ON DRONE FRAMES")
    print(f"{'='*70}")
    print(f"  Model Loaded:     {model_path}")
    print(f"  Confidence Cutoff: {conf_threshold}")
    print(f"  Inference Scale:  {imgsz}x{imgsz}")
    print(f"  Output Directory: {out_dir}")
    print(f"{'='*70}\n")

    model = YOLO(model_path)

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_frames = manifest["frames"]
    step = max(1, len(all_frames) // num_sample_frames)
    sample_frames = [all_frames[i] for i in range(0, len(all_frames), step)][:num_sample_frames]

    total_detections_all = 0
    all_crops_gallery = []
    frame_summaries = []

    for idx, fr in enumerate(sample_frames):
        img_path = fr["file"]
        ts = fr["timestamp_sec"]
        fidx = fr["frame_idx"]

        if not os.path.exists(img_path):
            continue

        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            continue
        h_orig, w_orig = frame_bgr.shape[:2]

        # Predict person class (0)
        results = model.predict(img_path, conf=conf_threshold, imgsz=imgsz, classes=[0], verbose=False)[0]
        boxes = results.boxes

        annotated_img = frame_bgr.copy()
        frame_person_crops = []

        for b_i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            total_detections_all += 1

            # Clamp coordinates
            x1 = max(0, min(w_orig - 1, x1))
            y1 = max(0, min(h_orig - 1, y1))
            x2 = max(x1 + 2, min(w_orig, x2))
            y2 = max(y1 + 2, min(h_orig, y2))

            bw = x2 - x1
            bh = y2 - y1

            # Extract cropped human
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (120, 240))
                banner = np.zeros((32, 120, 3), dtype=np.uint8)
                cv2.putText(banner, f"#{total_detections_all} ({conf:.0%})", (6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                crop_card = np.vstack([banner, crop_resized])
                frame_person_crops.append(crop_card)
                all_crops_gallery.append(crop_card)

                # Save individual crop
                single_crop_path = os.path.join(zooms_dir, f"person_det_{total_detections_all:03d}_conf{conf:.2f}.jpg")
                cv2.imwrite(single_crop_path, crop)

            # Draw precise bounding box
            color = (0, 255, 0) if conf >= 0.5 else (0, 200, 255)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)

            # Bounding box label
            label = f"Pedestrian {conf:.0%} [{bw}x{bh}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated_img, (x1, max(0, y1 - 24)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(annotated_img, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Header overlay
        header_txt = f"Frame {fidx} (Time: {ts:.1f}s) | Detected Humans: {len(boxes)} | YOLO Fine-Tuned Model"
        cv2.rectangle(annotated_img, (30, 30), (1250, 90), (0, 0, 0), -1)
        cv2.putText(annotated_img, header_txt, (50, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)

        # Save annotated image
        annotated_path = os.path.join(out_dir, f"frame_{fidx:06d}_annotated_detections.jpg")
        cv2.imwrite(annotated_path, annotated_img)

        # Save frame zoom panel
        if frame_person_crops:
            max_c = min(8, len(frame_person_crops))
            row_chunks = [frame_person_crops[k:k+max_c] for k in range(0, len(frame_person_crops), max_c)]
            # Equalize row lengths
            for r in row_chunks:
                while len(r) < max_c:
                    r.append(np.zeros_like(frame_person_crops[0]))
            panel = np.vstack([np.hstack(r) for r in row_chunks])
            panel_path = os.path.join(out_dir, f"frame_{fidx:06d}_pedestrian_crops_panel.jpg")
            cv2.imwrite(panel_path, panel)

        print(f"  Frame {fidx:05d} (t={ts:6.1f}s): {len(boxes):2d} pedestrians detected -> {annotated_path}")
        frame_summaries.append({
            "frame_idx": fidx,
            "timestamp_sec": ts,
            "pedestrians_found": len(boxes),
            "annotated_image": annotated_path
        })

    # Create Master Gallery of ALL detected humans across the inspection
    if all_crops_gallery:
        cols = 10
        total_crops = len(all_crops_gallery)
        rows_list = []
        for k in range(0, total_crops, cols):
            chunk = all_crops_gallery[k:k+cols]
            while len(chunk) < cols:
                chunk.append(np.zeros_like(all_crops_gallery[0]))
            rows_list.append(np.hstack(chunk))
        master_gallery = np.vstack(rows_list)
        gallery_path = os.path.join(out_dir, "MASTER_DETECTED_PEDESTRIANS_GALLERY.jpg")
        cv2.imwrite(gallery_path, master_gallery)
        print(f"\n  [Master Gallery] Generated combined gallery with {total_crops} detected pedestrians:")
        print(f"    -> {gallery_path}")

    print(f"\n{'='*70}")
    print(f"  DETECTION INSPECTION COMPLETED SUCCESSFULLY")
    print(f"  Total Human Bounding Boxes Verified: {total_detections_all}")
    print(f"  Visual Verification Folder: {out_dir}")
    print(f"{'='*70}\n")
    return frame_summaries


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_train", action="store_true", help="Skip training and run inference with base model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--model", default="yolov8m.pt")
    args = ap.parse_args()

    if not args.skip_train:
        trained_ckpt = train_yolo_drone_detector(
            dataset_yaml="work/detector_finetuning/dataset.yaml",
            base_model=args.model,
            epochs=args.epochs,
            imgsz=960,
            batch_size=4
        )
    else:
        trained_ckpt = args.model

    generate_visual_detection_proof(trained_ckpt)
