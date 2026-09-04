# UAV Drone Pedestrian Detection, Tracking & Multi-Attribute Recognition System

A complete research and operational computer vision pipeline for **UAV Drone Video Analysis**, implementing:
- High-Resolution Human Detection (`YOLOv8` / `RT-DETRv2`)
- Persistent Multi-Object Tracking (`ByteTrack` / `DeepSORT` with 8D Kalman Filtering + Swin Re-ID)
- Multi-Attribute Recognition (`Swin-Tiny` + `DFSM` Diverse Feature Discovery + `S-ACRM` Spatial Cross-Relation + K-Observation Evidence Fusion)
- Pedestrian Crossing Intention Prediction (`ST-GCNN` + `GCN-AE` + `LSTM Decoder`)
- DataFromSky Physical Kinematics Extraction (Position in meters, Speed in km/h, Acceleration in m/s², Distance to Crosswalks, Behavioral Classification)

---

## 📑 Project Structure

```text
TRANSPORTATION/
├── 1-s2.0-S2590198225000454-main.pdf                              # Reference literature
├── FINAL_UAV_Drone_Project_Methodology_VERIFIED_LINKS.pdf         # Core project methodology (PDF)
├── Pedestrian_Crossing_Intention_Prediction_...pdf               # Crossing intention reference
├── UAV_Drone_Project_Methodology v2.docx                          # Methodology specifications
├── codebase/
│   ├── config.py                                                 # Central pipeline configuration
│   ├── stage1_video_audit.py                                     # Stage 1: Video metadata & altitude audit
│   ├── stage2_frame_sampling.py                                  # Stage 2: Scene-change adaptive sampling
│   ├── stage3_detection.py                                       # Stage 3: Human & environment detection
│   ├── stage3_visual_detection_inspection.py                     # High-res bounding box verification
│   ├── stage4_tracking.py                                        # Stage 4: ByteTrack persistent tracking
│   ├── stage4_deepsort_tracker.py                                # Stage 4: DeepSORT + Swin Re-ID tracker
│   ├── stage5_crop_selection.py                                  # Stage 5: Farthest-point diverse crop selection
│   ├── stage6_attribute_model.py                                 # Stage 6: Swin + DFSM + S-ACRM attribute model
│   ├── stage7_output_and_ablation.py                             # Stage 7: Multi-view K-observation ablation
│   ├── stage8_zebra_crossing.py                                  # Stage 8: Crosswalk polygon extraction
│   ├── stage9_environment_graph.py                               # Stage 9: Spatio-temporal scene graph builder
│   ├── stage10_environment_encoder.py                            # Stage 10: GCN-AE & ST-GCNN encoder
│   ├── stage11_pedestrian_state_encoder.py                       # Stage 11: Trajectory LSTM-AE & Skeleton GCN
│   ├── stage12_intention_decoder.py                              # Stage 12: Intention prediction LSTM decoder
│   ├── stage13_combined_output.py                                # Stage 13: End-to-end unified record merger
│   ├── stage14_train_intention.py                                # Stage 14: Intention training & inference orchestrator
│   ├── stage15_physical_trajectory_analysis.py                   # Stage 15: DataFromSky physical kinematics
│   ├── stage16_pedestrian_comprehensive_features.py              # Stage 16: 8 core features (Gender, Group, Load, Speed, Wait, Attempts, Gestures)
│   ├── stage17_annotate_full_video.py                            # Stage 17: Full-video HUD overlay & crossing annotation
│   ├── stage18_gait_and_crossing_dynamics.py                     # Stage 18: Gait biomechanics & Kerb/Median origin-destination
│   ├── stage19_extract_physical_verification_frames.py           # Stage 19: Physical video frame extraction & verification
│   ├── stage20_comprehensive_physical_verification_atlas.py      # Stage 20: Multi-frame gait & group crossing visual atlas
│   ├── stage21_vivid_standalone_verification_frames.py           # Stage 21: One-image-per-frame vivid verification generator
│   ├── refine_crossing_and_roadway.py                            # Exact roadway & two white boundary line definition
│   ├── generate_mathematical_pdf_report.py                       # 8-page publication-grade PDF report generator
│   ├── finetune_yolo_detector.py                                 # Target-domain detector fine-tuning ladder
│   ├── generate_complete_output.py                               # Human-readable report & CSV generator
│   ├── generate_verification.py                                  # Visual crop grids for gender verification
│   ├── checkpoints/                                              # Trained PyTorch model weights
│   │   ├── attribute_model.pt                                    # Swin-Tiny + DFSM + S-ACRM weights (52 MB)
│   │   ├── yolo11n.pt                                            # YOLO11 detector weights
│   │   └── yolo11n-pose.pt                                       # YOLO11 pose estimation weights
│   └── work/                                                     # Output deliverables
│       ├── PEDESTRIAN_BEHAVIOR_AND_GAIT_MATHEMATICAL_REPORT.pdf   # 8-page comprehensive mathematical PDF report (6.87 MB)
│       ├── crossing_zones/                                       # Perfected roadway & crossing boundaries
│       │   ├── perfect_crossing_zones_annotated.jpg              # High-res zone boundary map
│       │   ├── crossing_zones_perfect.json                       # Exact polygon specifications
│       │   └── annotated_intersection_video.mp4                  # 1080p demonstration video with live HUD (26 MB)
│       ├── features/                                             # 8-feature extraction deliverables
│       │   ├── pedestrian_behavioral_features_per_person.csv     # Per-person 8-feature table
│       │   ├── pedestrian_behavioral_features_per_person.json    # Structured JSON
│       │   ├── pedestrian_behavioral_timeseries.csv              # 759-row trajectory timeseries
│       │   ├── pedestrian_features_dashboard.png                 # 8-panel analytics dashboard
│       │   ├── pedestrian_crossing_trajectories_annotated.jpg    # Full trajectory overlay map
│       │   └── comprehensive_features_report.txt                 # Statistical summary
│       ├── gait_crossing/                                        # Gait & Kerb/Median dynamics
│       │   ├── pedestrian_gait_and_crossing_summary.csv          # Gait metrics & origin-destination table
│       │   ├── pedestrian_gait_and_crossing_summary.json         # Structured JSON
│       │   ├── pedestrian_gait_timeseries.csv                    # 759-row instantaneous gait dynamics
│       │   ├── gait_and_crossing_dashboard.png                   # 6-panel gait & crossing dashboard
│       │   ├── kerb_median_crossing_map.jpg                      # Kerb vs. Median vector map
│       │   └── gait_and_crossing_report.txt                      # Detailed analysis report
│       ├── output/
│       │   ├── FINAL_complete_results.csv                        # Main human-readable table (49 pedestrians)
│       │   ├── FINAL_complete_results.json                       # Main JSON summary
│       │   ├── FINAL_pipeline_report.txt                         # Full printable report (797 lines)
│       │   ├── combined_person_records.csv                       # Stage 13 merged time-window records
│       │   └── per_person_attributes.csv                         # 759 detection-level attribute rows
│       ├── verification/
│       │   ├── ALL_49_persons_grid.jpg                           # Visual grid of all 49 pedestrians
│       │   ├── MALES_33_grid.jpg                                 # Visual grid of 33 males
│       │   ├── FEMALES_16_grid.jpg                               # Visual grid of 16 females
│       │   └── person_<ID>_<Gender>/                             # Multi-angle crops per person
│       ├── trajectories/
│       │   ├── datafromsky_pedestrian_trajectories.csv           # 759 physical trajectory timesteps
│       │   ├── pedestrian_kinematics_summary.json                # Kinematic summary (speeds, distances)
│       │   └── trajectories_birds_eye_overlay.png                # Bird's-eye trajectory overlay plot
│       └── detections/
│           ├── detections.json                                   # 9,810 human detections across 823 frames
│           └── finetuning_demonstration/                         # Visual bounding box verification
```

---

## 📊 Summary of Pipeline Results

- **Processed Drone Video:** 4K UHD Aerial Footage (~28 min, 16.7 GB)
- **Sampled Frames:** 823 frames sampled adaptively based on scene change
- **Total Person Detections:** 9,810 bounding box detections
- **Tracked Pedestrians:** **49 distinct individuals** persistently tracked
- **Gender Classification:** **33 Males** (67.3%), **16 Females** (32.7%) with **90.0% average confidence**
- **Physical Kinematics:** Calibrated Ground Sampling Distance ($GSD = 1.5\text{ cm/pixel}$), speed in km/h, acceleration in m/s², distance to nearest crosswalk in meters
- **Crossing Intention:** Predicted crossing probability and waiting duration at curb

---

## 🚀 Quickstart

### 1. Run Complete Pipeline
```bash
cd codebase
python generate_complete_output.py
```

### 2. Physical Trajectory & Kinematics Analysis
```bash
python stage15_physical_trajectory_analysis.py
```

### 3. Generate Visual Verification Grids
```bash
python generate_verification.py
```

### 4. Inspect High-Resolution YOLO Detections
```bash
python stage3_visual_detection_inspection.py
```
