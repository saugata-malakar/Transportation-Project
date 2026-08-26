import os
import cv2
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import timm
import argparse
from tqdm import tqdm

from config import ATTRIBUTE, PATHS, FUSION

# ==============================================================================
# 1. DFSM — Diverse Feature Discovery Module
# ==============================================================================
class DFSM(nn.Module):
    """
    Diverse Feature Discovery Module (§6.2).
    Includes Peak Region Suppression (PRS) and Random Region Suppression (RRS).
    """
    def __init__(self, suppress_topk, rrs_mask_ratio):
        super(DFSM, self).__init__()
        self.suppress_topk = suppress_topk
        self.rrs_mask_ratio = rrs_mask_ratio

    def forward(self, feature_map, training=False):
        """
        Args:
            feature_map: Spatial features (B, C, H, W)
            training: Boolean indicating if model is in training mode
        Returns:
            suppressed_map: Features with regions suppressed (B, C, H, W)
        """
        B, C, H, W = feature_map.shape

        # Peak Region Suppression (PRS)
        # Channel-wise sum to find highly activated spatial cells
        spatial_sum = feature_map.sum(dim=1)  # (B, H, W)
        spatial_sum_flat = spatial_sum.view(B, -1)  # (B, H*W)

        # Find top-k cells
        topk_val, topk_idx = torch.topk(spatial_sum_flat, self.suppress_topk, dim=-1)

        # Create mask to zero out top-k cells
        mask = torch.ones_like(spatial_sum_flat)
        mask.scatter_(1, topk_idx, 0.0)
        mask = mask.view(B, 1, H, W)

        suppressed_map = feature_map * mask

        # Random Region Suppression (RRS)
        if training and self.rrs_mask_ratio > 0.0:
            # Generate random mask
            rand_tensor = torch.rand(B, 1, H, W, device=feature_map.device)
            # Mask out cells based on rrs_mask_ratio
            drop_mask = (rand_tensor > self.rrs_mask_ratio).float()
            suppressed_map = suppressed_map * drop_mask

        return suppressed_map


# ==============================================================================
# 2. S-ACRM — Spatial-Activation Cross-Relation Module
# ==============================================================================
class SACRM(nn.Module):
    """
    Spatial-Activation Cross-Relation Module (§6.2).
    Discovers correlations between attributes based on spatial activations.
    """
    def __init__(self, attributes):
        super(SACRM, self).__init__()
        self.attributes = attributes

    def forward(self, feature_map, logits_dict, weight_dict):
        """
        Args:
            feature_map: Spatial features (B, C, H, W)
            logits_dict: Dictionary of attribute logits {attr_name: (B, num_classes)}
            weight_dict: Dictionary of attribute head weights {attr_name: (num_classes, C)}
        Returns:
            R: Relation matrix (B, M, M) where M is number of attributes
        """
        B, C, H, W = feature_map.shape
        M = len(self.attributes)

        Vc_list = []
        Ve_list = []

        for attr in self.attributes:
            # Confidence vector (Vc): max confidence for this attribute
            attr_logits = logits_dict[attr]  # (B, num_classes)
            max_logits, _ = torch.max(attr_logits, dim=-1)  # (B,)
            Vc_m = torch.sigmoid(max_logits)
            Vc_list.append(Vc_m)

            # Energy vector (Ve): spatial activation energy
            W_m = weight_dict[attr]  # (num_classes, C)
            W_m_filter = W_m.mean(dim=0).view(1, C, 1, 1)  # (1, C, 1, 1)
            
            # Filter spatial map and sum activations
            filtered = feature_map * W_m_filter
            Ve_m = filtered.sum(dim=(1, 2, 3))  # (B,)
            Ve_list.append(Ve_m)

        Vc = torch.stack(Vc_list, dim=1)  # (B, M)
        Ve = torch.stack(Ve_list, dim=1)  # (B, M)

        # Compute relation matrix R = softmax(Ve @ Vc.T)
        Ve_expanded = Ve.unsqueeze(-1)  # (B, M, 1)
        Vc_expanded = Vc.unsqueeze(1)   # (B, 1, M)
        
        energy_matrix = torch.bmm(Ve_expanded, Vc_expanded)  # (B, M, M)
        R = torch.softmax(energy_matrix, dim=-1)  # (B, M, M)

        return R


# ==============================================================================
# 3. AttributeModel — Full Model
# ==============================================================================
class AttributeModel(nn.Module):
    def __init__(self, pretrained=False):
        super(AttributeModel, self).__init__()
        
        # Load Swin-Tiny backbone
        try:
            self.backbone = timm.create_model(ATTRIBUTE.backbone, pretrained=pretrained)
        except Exception:
            self.backbone = timm.create_model(ATTRIBUTE.backbone, pretrained=False)
        # Remove classifier head
        self.backbone.reset_classifier(0)

        # Initialize modules
        self.dfsm = DFSM(ATTRIBUTE.dfsm_suppress_topk, ATTRIBUTE.rrs_mask_ratio)
        self.sacrm = SACRM(ATTRIBUTE.attributes)

        # Initialize attribute heads
        self.heads = nn.ModuleDict()
        for attr in ATTRIBUTE.attributes:
            num_classes = ATTRIBUTE.attribute_num_classes[attr]
            self.heads[attr] = nn.Linear(ATTRIBUTE.embed_dim, num_classes)

    def forward(self, x, training=False):
        # Extract features (use forward_features to get spatial maps)
        features = self.backbone.forward_features(x)
        
        # Handle Swin output shape depending on timm version
        if features.dim() == 3:  # (B, L, C)
            B, L, C = features.shape
            size = int(L ** 0.5)
            features = features.transpose(1, 2).view(B, C, size, size)
        elif features.dim() == 4 and features.shape[-1] == ATTRIBUTE.embed_dim:  # (B, H, W, C)
            features = features.permute(0, 3, 1, 2)

        # Apply DFSM
        suppressed_features = self.dfsm(features, training=training)

        # Global average pooling
        pooled_features = suppressed_features.mean(dim=(2, 3))

        # Pass through heads
        logits_dict = {}
        weight_dict = {}
        for attr in ATTRIBUTE.attributes:
            logits_dict[attr] = self.heads[attr](pooled_features)
            weight_dict[attr] = self.heads[attr].weight

        # Compute relation matrix via S-ACRM
        R = self.sacrm(suppressed_features, logits_dict, weight_dict)

        return logits_dict, R


# ==============================================================================
# 4. K-Frame Fusion
# ==============================================================================
def fuse_k_observations(relation_matrices):
    """
    Fuse relation matrices from K observations using max-confidence rule.
    Args:
        relation_matrices: list of K tensors, each (M, M) or (B, M, M)
    Returns:
        R_c: element-wise maximum across K matrices
    """
    stacked = torch.stack(relation_matrices, dim=0)
    R_c = torch.max(stacked, dim=0)[0]
    return R_c


# ==============================================================================
# 5. Dataset and Training
# ==============================================================================
class PedestrianCropDataset(Dataset):
    def __init__(self, crop_index_path, labels_path=None, transform=None):
        """
        crop_index_path: JSON from stage5 with format:
            {"persons": {"<track_id>": [{"crop_file": "...", ...}, ...], ...}}
        labels_path: JSON mapping track_id -> dict of attributes
        """
        with open(crop_index_path, 'r') as f:
            raw = json.load(f)

        # Handle both formats: {"persons": {...}} or flat {"<id>": [...]}
        self.crop_index = raw.get("persons", raw)

        self.labels = None
        if labels_path:
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)

        self.transform = transform

        # Flatten structure into a list of (image_path, track_id)
        self.samples = []
        for track_id, crop_entries in self.crop_index.items():
            for entry in crop_entries:
                # entry can be a dict {"crop_file": "...", ...} or a raw path string
                if isinstance(entry, dict):
                    img_path = entry.get("crop_file", entry.get("file", ""))
                else:
                    img_path = entry
                self.samples.append((img_path, track_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, track_id = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for missing image
            img = np.zeros((ATTRIBUTE.crop_size, ATTRIBUTE.crop_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label_dict = {}
        if self.labels and track_id in self.labels:
            track_labels = self.labels[track_id]
            for attr in ATTRIBUTE.attributes:
                label_dict[attr] = track_labels.get(attr, ATTRIBUTE.unknown_label)
        else:
            for attr in ATTRIBUTE.attributes:
                label_dict[attr] = ATTRIBUTE.unknown_label

        return img, label_dict, track_id


def train_attribute_model(crop_index_path, labels_path, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((ATTRIBUTE.crop_size, ATTRIBUTE.crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PedestrianCropDataset(crop_index_path, labels_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=ATTRIBUTE.batch_size, shuffle=True, num_workers=0)
    
    model = AttributeModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=ATTRIBUTE.lr)
    
    # Ignore index for unknown labels
    criterion = nn.CrossEntropyLoss(ignore_index=ATTRIBUTE.unknown_label)
    
    model.train()
    
    for epoch in range(ATTRIBUTE.epochs):
        epoch_loss = 0.0
        
        # We track correct predictions per attribute for basic accuracy logging
        correct = {attr: 0 for attr in ATTRIBUTE.attributes}
        total = {attr: 0 for attr in ATTRIBUTE.attributes}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{ATTRIBUTE.epochs}")
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            optimizer.zero_grad()
            
            logits_dict, _ = model(imgs, training=True)
            
            loss = 0.0
            for attr in ATTRIBUTE.attributes:
                loss += criterion(logits_dict[attr], labels[attr])
                
                # Compute accuracy (only on valid labels)
                preds = torch.argmax(logits_dict[attr], dim=-1)
                valid_mask = labels[attr] != ATTRIBUTE.unknown_label
                if valid_mask.sum() > 0:
                    correct[attr] += (preds[valid_mask] == labels[attr][valid_mask]).sum().item()
                    total[attr] += valid_mask.sum().item()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dataloader):.4f}")
        for attr in ATTRIBUTE.attributes:
            acc = correct[attr] / total[attr] if total[attr] > 0 else 0
            print(f"  {attr} Acc: {acc:.4f}")
            
    # Save checkpoint
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


# ==============================================================================
# 6. Inference
# ==============================================================================
def infer_attributes(ckpt_path, crop_index_path, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((ATTRIBUTE.crop_size, ATTRIBUTE.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model = AttributeModel().to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found. Running with untrained weights.")
        
    model.eval()
    
    with open(crop_index_path, 'r') as f:
        raw = json.load(f)
    crop_index = raw.get("persons", raw)
        
    results = {}
    
    with torch.no_grad():
        for track_id, crop_entries in tqdm(crop_index.items(), desc="Inferring Attributes"):
            
            track_results = {attr: [] for attr in ATTRIBUTE.attributes}
            relation_matrices = []
            
            for entry in crop_entries:
                crop_path = entry.get("crop_file", entry) if isinstance(entry, dict) else entry
                img = cv2.imread(crop_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transform(img).unsqueeze(0).to(device)
                
                logits_dict, R = model(img, training=False)
                relation_matrices.append(R.squeeze(0))  # (M, M)
                
                for attr in ATTRIBUTE.attributes:
                    probs = torch.softmax(logits_dict[attr], dim=-1).squeeze(0)
                    track_results[attr].append(probs)
            
            if not relation_matrices:
                continue
                
            # Fuse K relation matrices
            R_c = fuse_k_observations(relation_matrices)
            
            # Predict attributes based on fused R_c
            # (R_c encodes cross-attribute relations, but for actual predictions,
            # we average the logits/probs as baseline, or use R_c to refine.
            # Following the simplest fusion: average probabilities across K frames)
            
            person_result = {}
            confidences = {}
            
            for attr in ATTRIBUTE.attributes:
                attr_probs = torch.stack(track_results[attr], dim=0)  # (K, num_classes)
                fused_probs = attr_probs.mean(dim=0)
                
                pred_class = torch.argmax(fused_probs).item()
                pred_conf = fused_probs[pred_class].item()
                
                person_result[attr] = pred_class
                confidences[attr] = round(pred_conf, 4)
                
            person_result["confidence"] = confidences
            results[track_id] = person_result
            
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Inference completed. Results saved to {out_path}")


# ==============================================================================
# 7. CLI
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Attribute Model (Stage 6)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--crop_index", required=True, help="Path to JSON mapping track_id -> crops")
    train_parser.add_argument("--labels", required=True, help="Path to ground truth labels JSON")
    train_parser.add_argument("--ckpt", default=PATHS.attribute_model_ckpt, help="Path to save checkpoint")
    
    # Infer command
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--crop_index", required=True, help="Path to JSON mapping track_id -> crops")
    infer_parser.add_argument("--ckpt", default=PATHS.attribute_model_ckpt, help="Path to load checkpoint")
    infer_parser.add_argument("--out", default="work/attribute_predictions.json", help="Path to save predictions")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_attribute_model(args.crop_index, args.labels, args.ckpt)
    elif args.command == "infer":
        infer_attributes(args.ckpt, args.crop_index, args.out)
