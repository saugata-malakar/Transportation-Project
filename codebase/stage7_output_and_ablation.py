import os
import csv
import json
import argparse
import numpy as np

# Use lazy imports for heavy libraries as instructed
def get_torch():
    import torch
    return torch

from config import PATHS, FUSION, ATTRIBUTE


def build_output(video_id, tracks_path, attr_preds_path, out_dir):
    """
    Merges detection tracks (JSON from stage4) with attribute predictions
    (JSON from stage6) to produce a per-detection CSV and an identity-level
    summary JSON.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "per_person_attributes.csv")
    out_json = os.path.join(out_dir, "per_person_summary.json")

    # Load identity-level attribute predictions (from stage6)
    if not os.path.exists(attr_preds_path):
        raise FileNotFoundError(f"Attribute predictions not found: {attr_preds_path}")
    with open(attr_preds_path, "r") as f:
        attr_preds = json.load(f)

    # Save the identity-level summary JSON directly
    with open(out_json, "w") as f:
        json.dump(attr_preds, f, indent=2)

    # Load tracks (JSON from stage4)
    if not os.path.exists(tracks_path):
        raise FileNotFoundError(f"Tracks file not found: {tracks_path}")
    with open(tracks_path, "r") as f:
        tracks_data = json.load(f)
    tracks = tracks_data.get("tracks", tracks_data)

    out_fields = [
        "video_id", "frame_id", "person_id", "x1", "y1", "x2", "y2",
        "detector_confidence", "backpack", "hat", "upper_colour",
        "upper_style", "lower_colour", "lower_style", "attribute_confidence"
    ]

    rows = []
    for person_id, detections in tracks.items():
        # Fetch identity-level attributes for this person
        attrs = attr_preds.get(person_id, {})
        # Compute mean attribute confidence
        conf_dict = attrs.get("confidence", {})
        if conf_dict:
            mean_conf = round(sum(conf_dict.values()) / len(conf_dict), 4)
        else:
            mean_conf = ""

        for det in detections:
            out_row = {
                "video_id": video_id,
                "frame_id": det.get("frame_id", ""),
                "person_id": person_id,
                "x1": det.get("x1", ""),
                "y1": det.get("y1", ""),
                "x2": det.get("x2", ""),
                "y2": det.get("y2", ""),
                "detector_confidence": det.get("confidence", ""),
                "backpack": attrs.get("backpack", ""),
                "hat": attrs.get("hat", ""),
                "upper_colour": attrs.get("upper_colour", ""),
                "upper_style": attrs.get("upper_style", ""),
                "lower_colour": attrs.get("lower_colour", ""),
                "lower_style": attrs.get("lower_style", ""),
                "attribute_confidence": mean_conf,
            }
            rows.append(out_row)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully built output files:")
    print(f"  - CSV ({len(rows)} rows): {out_csv}")
    print(f"  - JSON ({len(attr_preds)} persons): {out_json}")


def run_ablation(crop_index_path, labels_path, ckpt_path, out_dir="work/ablation"):
    """
    Implements §7 (Ablation). Evaluates attribute recognition performance when 
    using K=1 to 5 observations per identity.
    """
    import torch
    import cv2
    import torchvision.transforms as transforms
    from stage6_attribute_model import AttributeModel, fuse_k_observations

    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttributeModel(pretrained=False).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found.")
        
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((ATTRIBUTE.crop_size, ATTRIBUTE.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(crop_index_path):
        raise FileNotFoundError(f"Crop index not found: {crop_index_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
    with open(crop_index_path, "r") as f:
        raw_index = json.load(f)
    crop_index = raw_index.get("persons", raw_index)
        
    with open(labels_path, "r") as f:
        labels = json.load(f)
        
    k_values = FUSION.k_ablation_values
    results = {}
    
    print("\nRunning Ablation on K observations (K = 1..5)...")
    with torch.no_grad():
        for k in k_values:
            correct = {attr: 0 for attr in ATTRIBUTE.attributes}
            total = {attr: 0 for attr in ATTRIBUTE.attributes}
            
            for track_id, crop_entries in crop_index.items():
                if track_id not in labels:
                    continue
                    
                track_gt = labels[track_id]
                selected_entries = crop_entries[:k]
                if not selected_entries:
                    continue
                    
                track_results = {attr: [] for attr in ATTRIBUTE.attributes}
                relation_matrices = []
                
                for entry in selected_entries:
                    crop_path = entry.get("crop_file", entry) if isinstance(entry, dict) else entry
                    img = cv2.imread(crop_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    
                    logits_dict, R = model(img_tensor, training=False)
                    relation_matrices.append(R.squeeze(0))
                    
                    for attr in ATTRIBUTE.attributes:
                        probs = torch.softmax(logits_dict[attr], dim=-1).squeeze(0)
                        track_results[attr].append(probs)
                        
                if not relation_matrices:
                    continue
                    
                # Predict attributes
                for attr in ATTRIBUTE.attributes:
                    if attr in track_gt and track_gt[attr] != ATTRIBUTE.unknown_label:
                        attr_probs = torch.stack(track_results[attr], dim=0)  # (K_actual, num_classes)
                        fused_probs = attr_probs.mean(dim=0)
                        pred_class = torch.argmax(fused_probs).item()
                        gt_class = int(track_gt[attr])
                        if pred_class == gt_class:
                            correct[attr] += 1
                        total[attr] += 1
                        
            k_metrics = {}
            for attr in ATTRIBUTE.attributes:
                acc = correct[attr] / total[attr] if total[attr] > 0 else 0.0
                k_metrics[attr] = {"accuracy": round(acc, 4)}
                    
            mean_acc = float(np.mean([m["accuracy"] for m in k_metrics.values()])) if k_metrics else 0.0
            k_metrics["mean_accuracy"] = round(mean_acc, 4)
            results[k] = k_metrics
        
    # Print comparison table
    print("\n" + "="*80)
    print("Ablation Results: K observations vs Accuracy")
    print("="*80)
    
    # Header
    attrs_header = " | ".join([f"{a[:6]:<6}" for a in ATTRIBUTE.attributes])
    print(f"{'K':<5} | {'mA':<8} | {attrs_header}")
    print("-" * 80)
    
    # Rows
    for k in k_values:
        k_res = results.get(k, {})
        ma = k_res.get("mean_accuracy", 0)
        row_str = f"{k:<5} | {ma:.4f}   | "
        
        attr_strs = []
        for attr in ATTRIBUTE.attributes:
            acc = k_res.get(attr, {}).get("accuracy", 0.0)
            attr_strs.append(f"{acc:.4f}")
            
        row_str += " | ".join(attr_strs)
        print(row_str)
        
    print("="*80)
    
    out_results_path = os.path.join(out_dir, "ablation_metrics.json")
    with open(out_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed metrics to {out_results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 7: Output generation and Ablation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Subcommand: build
    build_parser = subparsers.add_parser("build", help="Build final output CSV and JSON")
    build_parser.add_argument("--video_id", required=True, help="Video identifier")
    build_parser.add_argument("--tracks", required=True, help="Path to tracks CSV")
    build_parser.add_argument("--attr_preds", required=True, help="Path to identity-level attribute predictions JSON")
    build_parser.add_argument("--out_dir", default="work/output", help="Output directory")
    
    # Subcommand: ablate
    ablate_parser = subparsers.add_parser("ablate", help="Run ablation study on K observations")
    ablate_parser.add_argument("--crop_index", required=True, help="Path to crop index JSON")
    ablate_parser.add_argument("--labels", required=True, help="Path to ground truth labels JSON")
    ablate_parser.add_argument("--ckpt", required=True, help="Path to attribute model checkpoint")
    ablate_parser.add_argument("--out_dir", default="work/ablation", help="Output directory for ablation results")
    
    args = parser.parse_args()
    
    if args.command == "build":
        build_output(args.video_id, args.tracks, args.attr_preds, args.out_dir)
    elif args.command == "ablate":
        run_ablation(args.crop_index, args.labels, args.ckpt, args.out_dir)
