"""
config.py
Central configuration for the UAV pedestrian-attribute pipeline.
Mirrors the parameters specified in the methodology document
(sections 3, 4, 5, 6, 7, 8).
"""

import os
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    raw_video: str = ""  # set at runtime -> your 5-6 min drone video
    work_dir: str = "work"
    frames_dir: str = "work/frames"
    detections_dir: str = "work/detections"
    cvat_export_dir: str = "work/cvat_export"
    tracks_dir: str = "work/tracks"
    crops_dir: str = "work/crops"
    attribute_model_ckpt: str = "checkpoints/attribute_model.pt"
    detector_weights: str = "checkpoints/rtdetr-l.pt"  # ultralytics RT-DETRv2
    final_output_csv: str = "work/output/per_person_attributes.csv"
    final_output_json: str = "work/output/per_person_attributes.json"


@dataclass
class SamplingConfig:
    # Section 3.2 Frame Sampling
    baseline_interval_sec: float = 2.0       # ~900 frames baseline for a 30-min video
    fast_change_interval_sec: float = 1.0    # ~1800 frames for fast-changing scenes
    scene_change_threshold: float = 30.0     # histogram-diff threshold to flag "fast-changing"
    dedup_hash_threshold: int = 5            # perceptual-hash distance below which a frame is a near-duplicate


@dataclass
class DetectionConfig:
    # Section 4
    model_name: str = "yolov8m.pt"          # Ultralytics pretrained checkpoint
    yolo_assist_model: str = "yolo11n.pt"    # optional annotation-assist model (4.1)
    confidence_export_threshold: float = 0.15  # permissive threshold for CVAT pre-annotation review
    confidence_inference_threshold: float = 0.35  # threshold for the final deployed system
    target_class_name: str = "person"
    imgsz: int = 960


@dataclass
class TrackingConfig:
    # Section 5 — ByteTrack
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 60          # frames to keep a lost track alive
    min_track_len: int = 5          # discard trajectories shorter than this
    diverse_crops_per_id: int = 8   # K observations selected per identity (see 6.1, "5 to 10")


@dataclass
class AttributeConfig:
    # Section 6
    crop_size: int = 224
    backbone: str = "swin_tiny_patch4_window7_224"  # timm model name
    attributes: tuple = (
        "gender",
        "hat",
        "backpack",
        "upper_colour",
        "upper_style",
        "lower_colour",
        "lower_style",
    )
    # class cardinalities per attribute head — adjust to match your UAV-Human label mapping
    attribute_num_classes: dict = field(default_factory=lambda: {
        "gender": 2,          # male / female
        "hat": 2,             # no / yes
        "backpack": 2,        # no / yes
        "upper_colour": 11,
        "upper_style": 4,
        "lower_colour": 11,
        "lower_style": 4,
    })
    dfsm_suppress_topk: int = 3      # PRS: number of top activated spatial cells suppressed
    rrs_mask_ratio: float = 0.15     # RRS: fraction of remaining cells randomly masked in training
    embed_dim: int = 768             # Swin-tiny final stage channel dim
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-5
    unknown_label: int = -1          # "unknown" rather than assumed negative (6.1)


@dataclass
class FusionConfig:
    # Section 6.2 / 7 — evidence fusion across K observations
    k_ablation_values: tuple = (1, 2, 3, 4, 5)  # K = 1..5 ablation (Section 7 / 8)
    fusion_rule: str = "max_confidence"  # "the clearer observation dominates" (6.2)


@dataclass
class IntentionConfig:
    # Crossing-intention module (Zhou et al. framework), Section III / IV.B
    # of the paper. Counts of nearby entities per graph — paper's own
    # experimental setting (Section IV-B): M=5 vehicles, N=3 pedestrians,
    # L=2 traffic signals, Z=2 zebra crossings; pad with zeros if fewer
    # are present nearby.
    M_vehicles: int = 5
    N_pedestrians: int = 3
    L_traffic_lights: int = 2
    Z_zebra_crossings: int = 2
    K_history_frames: int = 8       # frames of history feeding the graphs/LSTM per sample
    appearance_feat_dim: int = 1280  # GhostNet penultimate feature size (paper's Table I choice)
    gcn_ae_hidden_dims: tuple = (512, 256)  # (1280->512->256->512->1280) per Section III-D.1
    gcn_ae_gamma: float = 0.5        # Eq. 8 reconstruction loss weighting
    stgcn_hidden_dim: int = 256
    trajectory_embed_dim: int = 128
    skeleton_embed_dim: int = 256
    decoder_input_dim: int = 256
    decoder_hidden_dim: int = 64
    # Dataset construction (Section IV-A): K1 = clip duration, K2 = reaction
    # time offsets before the crossing starts.
    K1_clip_duration_sec: float = 1.0
    K2_offsets_sec: tuple = (0.0, 1.0, 2.0, 3.0)
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 16


PATHS = PathConfig()
SAMPLING = SamplingConfig()
DETECTION = DetectionConfig()
TRACKING = TrackingConfig()
ATTRIBUTE = AttributeConfig()
FUSION = FusionConfig()
INTENTION = IntentionConfig()
