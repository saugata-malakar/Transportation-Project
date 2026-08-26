"""
stage8_zebra_crossing.py
Supporting input for Section III-C.2's motion sub-graph (Eq. 6: distance
from each pedestrian to the nearest zebra crossing).

Zebra/pedestrian crossings are NOT a COCO class, so RT-DETR can't detect
them out of the box. Two practical options, both implemented here:

  1. HEURISTIC PROPOSAL (this file, `propose`): classical CV — threshold
     for bright stripe-like blobs, cluster them by proximity/orientation,
     draw candidate polygons on a representative frame so you can review
     them visually without a GUI.
  2. MANUAL OVERRIDE: since your footage is a (mostly) fixed drone hover
     over one intersection, you only need to annotate crosswalk polygons
     ONCE for the whole 30-min video, not per-frame. Open the visualization
     PNG this script produces, then hand-edit `zebra_crossings.json` with
     the correct polygon corner pixel coordinates (eyeballed from the PNG,
     or from any image viewer that shows cursor position).

`zebra_crossings.json` format (consumed by stage9_environment_graph.py):
    [
      {"id": 0, "polygon": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]},
      {"id": 1, "polygon": [...]}
    ]

Usage:
    python stage8_zebra_crossing.py propose --frame work/frames/frame_0000000_t00000.00.jpg \
        --out_dir work/zebra
    # review work/zebra/candidates_visualization.png, then hand-edit
    # work/zebra/zebra_crossings.json (starts as the heuristic proposal)
"""

import argparse
import json
import os

import cv2
import numpy as np


def propose_zebra_crossings(frame_path, out_dir, min_stripe_area=150, max_stripe_area=8000,
                             stripe_aspect_min=2.5, cluster_dist_px=60):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(frame_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Crosswalk stripes are bright relative to their local surroundings —
    # adaptive threshold handles uneven lighting better than a global one.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 51, -15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stripe_rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_stripe_area <= area <= max_stripe_area):
            continue
        rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
        (cx, cy), (w, h), angle = rect
        if min(w, h) < 1:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect < stripe_aspect_min:
            continue  # not elongated enough to be a paint stripe
        stripe_rects.append(rect)

    # Cluster nearby, similarly-oriented stripes into crosswalk groups
    clusters = []  # list of list-of-rect
    used = [False] * len(stripe_rects)
    for i, r in enumerate(stripe_rects):
        if used[i]:
            continue
        cluster = [r]
        used[i] = True
        (cxi, cyi), _, angi = r
        changed = True
        while changed:
            changed = False
            for j, r2 in enumerate(stripe_rects):
                if used[j]:
                    continue
                (cxj, cyj), _, angj = r2
                # near any member of the current cluster + roughly parallel
                for member in cluster:
                    (mcx, mcy), _, mang = member
                    dist = np.hypot(cxj - mcx, cyj - mcy)
                    ang_diff = min(abs(angj - mang), 180 - abs(angj - mang))
                    if dist <= cluster_dist_px and ang_diff <= 20:
                        cluster.append(r2)
                        used[j] = True
                        changed = True
                        break
        if len(cluster) >= 3:  # a real crosswalk has multiple parallel stripes
            clusters.append(cluster)

    zebra_candidates = []
    vis = img.copy()
    for idx, cluster in enumerate(clusters):
        pts = []
        for rect in cluster:
            box = cv2.boxPoints(rect)
            pts.extend(box.tolist())
        pts = np.array(pts, dtype=np.float32)
        hull = cv2.convexHull(pts).reshape(-1, 2)
        polygon = [[round(float(x), 1), round(float(y), 1)] for x, y in hull]
        zebra_candidates.append({"id": idx, "polygon": polygon})

        cv2.polylines(vis, [hull.astype(np.int32)], True, (0, 0, 255), 2)
        cx, cy = hull.mean(axis=0).astype(int)
        cv2.putText(vis, f"zebra_{idx}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    vis_path = os.path.join(out_dir, "candidates_visualization.png")
    cv2.imwrite(vis_path, vis)

    out_path = os.path.join(out_dir, "zebra_crossings.json")
    with open(out_path, "w") as f:
        json.dump(zebra_candidates, f, indent=2)

    print(f"Proposed {len(zebra_candidates)} candidate crosswalk(s).")
    print(f"Review -> {vis_path}")
    print(f"Edit as needed -> {out_path}  (this is what stage9 reads)")
    if len(zebra_candidates) == 0:
        print("No candidates found automatically — this is common with worn paint, "
              "unmarked crossings, or unusual lighting. Skip straight to manually writing "
              "zebra_crossings.json with polygons eyeballed from the frame image.")
    return out_path, vis_path


def main():
    ap = argparse.ArgumentParser(description="Zebra crossing proposal for the motion sub-graph")
    ap.add_argument("--frame", required=True, help="A representative frame from work/frames/")
    ap.add_argument("--out_dir", default="work/zebra")
    args = ap.parse_args()
    propose_zebra_crossings(args.frame, args.out_dir)


if __name__ == "__main__":
    main()
