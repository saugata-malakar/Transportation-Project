"""
finetune_yolo_detector.py
Section 4 & 6 — VisDrone & Target-Domain Detector Fine-Tuning Ladder.
"""

import os
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import shutil
import cv2
from ultralytics import YOLO

from config import DETECTION, PATHS


def prepare_yolo_dataset(detections_path="work/detections/detections.json",
                         manifest_path="work/frames_manifest.json",
                         dataset_dir="work/detector_finetuning"):
    """
    Creates standard YOLO dataset directory:
      dataset_dir/
        images/train, images/val
        labels/train, labels/val
        dataset.yaml
    """
    os.makedirs(os.path.join(dataset_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "val"), exist_ok=True)

    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    abs_dataset_dir = os.path.abspath(dataset_dir).replace("\\", "/")
    yaml_content = f"""path: {abs_dataset_dir}
train: images/train
val: images/val

names:
  0: person
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[Dataset Preparation] Configured YAML: {yaml_path}")
    return yaml_path


def finetune_detector(yaml_path, base_model="yolov8n.pt", epochs=3, imgsz=640, batch_size=8,
                      project_dir="work/detector_runs", save_ckpt="checkpoints/yolo_drone_human_finetuned.pt"):
    print(f"\n[Fine-Tuning] Initializing detector {base_model} on target dataset...")
    model = YOLO(base_model)

    print(f"[Fine-Tuning] Training for {epochs} epochs (imgsz={imgsz}, batch={batch_size})...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=0,
        project=project_dir,
        name="drone_target_adaptation",
        exist_ok=True,
        verbose=True,
        device="cpu"
    )

    # Save to checkpoints
    best_weights = os.path.join(project_dir, "drone_target_adaptation", "weights", "best.pt")
    if not os.path.exists(best_weights):
        best_weights = os.path.join(project_dir, "drone_target_adaptation", "weights", "last.pt")

    if os.path.exists(best_weights):
        os.makedirs(os.path.dirname(save_ckpt), exist_ok=True)
        shutil.copyfile(best_weights, save_ckpt)
        print(f"\n[Fine-Tuning] Model checkpoint saved -> {save_ckpt}")

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fine-tune YOLO detector on drone data")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    yaml_p = prepare_yolo_dataset()
    finetune_detector(yaml_p, base_model=args.model, epochs=args.epochs, imgsz=args.imgsz)
