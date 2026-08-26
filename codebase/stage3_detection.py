"""
stage3_detection.py
Section 4 Human Detection

- Runs RT-DETRv2 (pretrained, COCO) as the D0 zero-adaptation baseline
  on your sampled frames (4, first paragraph).
- Exports predicted boxes at a PERMISSIVE confidence threshold to a
  CVAT-importable format for human review (4.1).
- Provides the training-ladder scaffold (4.3): D0 (COCO pretrained) ->
  D1 (+ optional SkyScenes synthetic pretraining) -> D2 (fine-tuned on your
  reviewed project annotations). Each rung is trained/evaluated on the SAME
  held-out test segment produced by stage2's split.
- Reports Precision / Recall / mAP50 / mAP50-95 / FPS per rung, stratified
  by person size where box-size labels permit.

Requires: pip install ultralytics
(RT-DETRv2 ships as `rtdetr-l.pt` / `rtdetr-x.pt` in ultralytics >= 8.1)

Usage:
    # D0 baseline inference + CVAT export
    python stage3_detection.py detect --frames_dir work/frames \
        --manifest work/frames_manifest.json --out_dir work/detections

    # D2: fine-tune on your CVAT-reviewed annotations (YOLO-format dataset.yaml)
    python stage3_detection.py finetune --data uav_dataset.yaml \
        --weights rtdetr-l.pt --epochs 60 --imgsz 960 --name D2_finetuned

    # Evaluate any rung's weights on the held-out test segment
    python stage3_detection.py eval --weights runs/detect/D2_finetuned/weights/best.pt \
        --data uav_dataset.yaml
"""

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET

from config import DETECTION, PATHS


def _load_model(weights: str):
    if "rtdetr" in str(weights).lower() and os.path.exists(weights):
        from ultralytics import RTDETR
        return RTDETR(weights)
    else:
        from ultralytics import YOLO
        return YOLO(weights)


# COCO class ids relevant to the environment graph (Zhou et al., crossing-intention paper).
# Default stays person-only for the gender/attribute pipeline; the environment-graph
# pipeline (stage9+) calls run_detection with COCO_ENV_CLASSES instead.
COCO_CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                     5: "bus", 7: "truck", 9: "traffic_light"}
COCO_ENV_CLASSES = [0, 1, 2, 3, 5, 7, 9]


def run_detection(frames_dir, manifest_path, out_dir, weights=DETECTION.model_name,
                   conf=DETECTION.confidence_export_threshold, imgsz=DETECTION.imgsz,
                   classes=None):
    """classes: list of COCO class ids to detect. None -> person-only (default,
    used by the gender/attribute pipeline). Pass COCO_ENV_CLASSES for the
    crossing-intention pipeline's environment graph (needs vehicles + traffic
    lights too, per Section III-C of the Zhou et al. framework)."""
    os.makedirs(out_dir, exist_ok=True)
    model = _load_model(weights)
    classes = classes if classes is not None else [0]

    with open(manifest_path) as f:
        manifest = json.load(f)
    frame_records = manifest["frames"]

    all_detections = []
    person_sizes = []
    t0 = time.time()
    for rec in frame_records:
        fpath = rec["file"]
        results = model.predict(fpath, conf=conf, imgsz=imgsz, classes=classes, verbose=False)
        r = results[0]
        frame_dets = []
        for box in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = COCO_CLASS_NAMES.get(cls_id, str(cls_id))
            frame_dets.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": confidence,
                                "class_id": cls_id, "class_name": cls_name})
            if cls_id == 0:
                person_sizes.append((x2 - x1) * (y2 - y1))
        all_detections.append({
            "frame_idx": rec["frame_idx"],
            "timestamp_sec": rec["timestamp_sec"],
            "file": fpath,
            "split": rec.get("split", "train"),
            "detections": frame_dets,
        })
    elapsed = time.time() - t0
    fps = len(frame_records) / elapsed if elapsed > 0 else float("nan")

    det_path = os.path.join(out_dir, "detections.json")
    with open(det_path, "w") as f:
        json.dump({"weights": weights, "conf_threshold": conf, "fps_inference": round(fps, 2),
                    "classes_detected": [COCO_CLASS_NAMES.get(c, str(c)) for c in classes],
                    "num_frames": len(frame_records),
                    "num_detections": sum(len(d["detections"]) for d in all_detections),
                    "median_person_area_px2": (sorted(person_sizes)[len(person_sizes)//2]
                                                if person_sizes else None),
                    "frames": all_detections}, f, indent=2)

    export_to_cvat_xml(all_detections, os.path.join(out_dir, "cvat_preannotations.xml"))

    print(f"Detected {[COCO_CLASS_NAMES.get(c,c) for c in classes]} in {len(frame_records)} "
          f"frames @ {fps:.2f} FPS.")
    print(f"Raw detections -> {det_path}")
    print(f"CVAT pre-annotation XML -> {os.path.join(out_dir, 'cvat_preannotations.xml')}")
    return det_path


def export_to_cvat_xml(all_detections, out_path):
    """Section 4.1 — export predicted boxes to CVAT for human review,
    using a permissive threshold so weak candidates surface for review."""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    for i, frame in enumerate(all_detections):
        image_el = ET.SubElement(root, "image", {
            "id": str(i),
            "name": os.path.basename(frame["file"]),
            "frame_idx": str(frame["frame_idx"]),
        })
        for det in frame["detections"]:
            ET.SubElement(image_el, "box", {
                "label": det.get("class_name", "person"),
                "occluded": "0",
                "source": "rtdetr_auto",
                "xtl": f"{det['x1']:.2f}",
                "ytl": f"{det['y1']:.2f}",
                "xbr": f"{det['x2']:.2f}",
                "ybr": f"{det['y2']:.2f}",
                "z_order": "0",
                "attributes_confidence": f"{det['confidence']:.3f}",
            })
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def finetune(data_yaml, weights=DETECTION.model_name, epochs=60, imgsz=DETECTION.imgsz, name="D_finetuned"):
    """Section 4.3 — one rung of the detector training ladder.
    `data_yaml` must be a YOLO-format dataset config produced after CVAT
    review has been exported back to YOLO txt labels (CVAT supports this
    export format natively: Menu > Export annotations > YOLO 1.1)."""
    model = _load_model(weights)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, name=name, exist_ok=True)
    return model


def evaluate(weights, data_yaml, imgsz=DETECTION.imgsz):
    """Reports Precision, Recall, mAP50, mAP50-95 on the held-out test split
    (Section 4.3). Run once per training-ladder rung with identical data_yaml
    test split to make rungs directly comparable."""
    model = _load_model(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz, split="test")
    report = {
        "weights": weights,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description="Section 4 — Human detection")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect")
    d.add_argument("--frames_dir", default="work/frames")
    d.add_argument("--manifest", default="work/frames_manifest.json")
    d.add_argument("--out_dir", default="work/detections")
    d.add_argument("--weights", default=DETECTION.model_name)
    d.add_argument("--conf", type=float, default=DETECTION.confidence_export_threshold)
    d.add_argument("--env_classes", action="store_true",
                    help="Detect person+vehicles+traffic_light (for the environment graph) "
                         "instead of person-only.")

    ft = sub.add_parser("finetune")
    ft.add_argument("--data", required=True)
    ft.add_argument("--weights", default=DETECTION.model_name)
    ft.add_argument("--epochs", type=int, default=60)
    ft.add_argument("--imgsz", type=int, default=DETECTION.imgsz)
    ft.add_argument("--name", default="D_finetuned")

    ev = sub.add_parser("eval")
    ev.add_argument("--weights", required=True)
    ev.add_argument("--data", required=True)

    args = ap.parse_args()
    if args.cmd == "detect":
        cls = COCO_ENV_CLASSES if args.env_classes else [0]
        run_detection(args.frames_dir, args.manifest, args.out_dir, args.weights, args.conf,
                      classes=cls)
    elif args.cmd == "finetune":
        finetune(args.data, args.weights, args.epochs, args.imgsz, args.name)
    elif args.cmd == "eval":
        evaluate(args.weights, args.data)


if __name__ == "__main__":
    main()
