"""
run_pipeline.py
Orchestrator for stages 1-5 (automated, no labels needed).
Runs: audit -> sample frames -> detect -> track -> select diverse crops.

Usage:
    python run_pipeline.py --video DJI_20251005162440_0129_D.MP4 --video_id run01
"""

import argparse
import os
import json

from config import PATHS, DETECTION, TRACKING, SAMPLING


def main():
    ap = argparse.ArgumentParser(description="UAV Pedestrian Attribute Pipeline (Stages 1-5)")
    ap.add_argument("--video", required=True, help="Path to the input video")
    ap.add_argument("--video_id", required=True, help="Unique identifier for this video run")
    args = ap.parse_args()

    # Create work/ directory structure
    work_dir = PATHS.work_dir
    frames_dir = PATHS.frames_dir
    detections_dir = PATHS.detections_dir
    tracks_dir = PATHS.tracks_dir
    crops_dir = PATHS.crops_dir

    for d in [work_dir, frames_dir, detections_dir, tracks_dir, crops_dir]:
        os.makedirs(d, exist_ok=True)

    manifest_out = os.path.join(work_dir, "frames_manifest.json")

    print("=" * 60)
    print(f"Starting pipeline for video: {args.video} (ID: {args.video_id})")
    print("=" * 60)

    # --- Stage 1: Video Audit ---
    print("\n>>> Stage 1: Video Audit")
    try:
        import stage1_video_audit
        info = stage1_video_audit.get_video_info(args.video)
        frames_for_audit = stage1_video_audit.extract_frames(args.video, num_frames=5)
        avg_w, avg_h = stage1_video_audit.estimate_person_size(frames_for_audit)
        motion_var, brightness, contrast = stage1_video_audit.analyze_motion_and_lighting(frames_for_audit)
        rec_rate = "1 frame/1s" if motion_var > 5.0 else "1 frame/2s"
        report = {
            "video_info": info,
            "metrics": {
                "avg_person_width_px": round(avg_w, 2),
                "avg_person_height_px": round(avg_h, 2),
                "motion_variance": round(motion_var, 2),
                "avg_brightness": round(brightness, 2),
                "avg_contrast": round(contrast, 2),
            },
            "recommendations": {"sampling_rate": rec_rate},
        }
        audit_path = os.path.join(work_dir, "audit_report.json")
        with open(audit_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Audit report saved to {audit_path}")
        print(f"  Recommended sampling: {rec_rate}")
    except Exception as e:
        print(f"  Warning: Stage 1 audit failed ({e}), continuing...")

    # --- Stage 2: Frame Sampling ---
    print("\n>>> Stage 2: Frame Sampling")
    if os.path.exists(manifest_out) and os.path.getsize(manifest_out) > 100:
        with open(manifest_out, "r") as f:
            manifest_data = json.load(f)
        manifest = manifest_data.get("frames", [])
        print(f"  Reusing existing manifest with {len(manifest)} sampled frames -> {manifest_out}")
    else:
        import stage2_frame_sampling
        manifest, scene_changes, meta = stage2_frame_sampling.sample_frames(
            args.video, frames_dir,
            baseline_interval=SAMPLING.baseline_interval_sec,
            fast_interval=SAMPLING.fast_change_interval_sec,
        )
        manifest = stage2_frame_sampling.split_by_segment(manifest)

        with open(manifest_out, "w") as f:
            json.dump(
                {
                    "num_frames_sampled": len(manifest),
                    "num_scene_changes_detected": len(scene_changes),
                    "video_meta": meta,
                    "frames": manifest,
                },
                f,
                indent=2,
            )
        print(f"  Sampled {len(manifest)} frames -> {manifest_out}")

    # --- Stage 3: Detection ---
    print("\n>>> Stage 3: Detection")
    det_json_path = os.path.join(detections_dir, "detections.json")
    if os.path.exists(det_json_path) and os.path.getsize(det_json_path) > 100:
        print(f"  Reusing existing detections -> {det_json_path}")
        det_path = det_json_path
    else:
        import stage3_detection
        det_path = stage3_detection.run_detection(
            frames_dir,
            manifest_out,
            detections_dir,
            weights=DETECTION.model_name,
            conf=DETECTION.confidence_export_threshold,
            imgsz=DETECTION.imgsz,
        )
        print(f"  Detections saved to {det_path}")

    # --- Stage 4: Tracking ---
    print("\n>>> Stage 4: ByteTrack Tracking")
    import stage4_tracking
    det_json_path = os.path.join(detections_dir, "detections.json")
    detections_data = stage4_tracking.load_detections(det_json_path)
    manifest_data = stage4_tracking.load_manifest(manifest_out)
    tracks_result = stage4_tracking.run_tracking(detections_data, manifest_data)
    tracks_out = os.path.join(tracks_dir, "tracks.json")
    with open(tracks_out, "w") as f:
        json.dump(tracks_result, f, indent=2)
    print(f"  Tracks saved to {tracks_out}")

    # --- Stage 5: Crop Selection ---
    print("\n>>> Stage 5: Diverse Crop Selection")
    import stage5_crop_selection

    with open(manifest_out, "r") as f:
        manifest_json = json.load(f)
    # Build frame_id -> file path map
    manifest_map = {}
    for fr in manifest_json["frames"]:
        fid = str(fr.get("frame_idx", fr.get("frame_id", "")))
        manifest_map[fid] = fr["file"]

    out_index = {"persons": {}}
    total_discarded = 0
    total_selected = 0

    for person_id, track_list in tracks_result.get("tracks", {}).items():
        results, discarded = stage5_crop_selection.process_person(
            person_id, track_list, manifest_map, crops_dir,
            k=TRACKING.diverse_crops_per_id,
        )
        total_discarded += discarded
        if results:
            out_index["persons"][person_id] = results
            total_selected += len(results)

    index_path = os.path.join(crops_dir, "crop_index.json")
    with open(index_path, "w") as f:
        json.dump(out_index, f, indent=2)
    print(f"  Selected {total_selected} crops for {len(out_index['persons'])} persons")
    print(f"  Discarded {total_discarded} blurry crops")
    print(f"  Crop index -> {index_path}")

    # Done — print next steps
    print("\n" + "=" * 60)
    print("PIPELINE STAGES 1-5 COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  1. Review crops in {crops_dir}/person_<id>/")
    print("  2. Create labels.json with per-identity attribute annotations:")
    print('     {"<person_id>": {"gender": 0, "hat": 1, "backpack": 0, ...}, ...}')
    print("  3. Train the attribute model:")
    print(f"     python stage6_attribute_model.py train --crop_index {index_path} --labels labels.json")
    print("  4. Run inference + output:")
    print(f"     python stage6_attribute_model.py infer --ckpt checkpoints/attribute_model.pt --crop_index {index_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
