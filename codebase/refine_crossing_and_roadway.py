"""
refine_crossing_and_roadway.py
Pixel-perfect definitions of:
  1. Roadway Boundary (entire driveable intersection perimeter along curbs)
  2. Pedestrian Crossing Area (the legal crossing corridor across the roadway & intersection)
  3. Two White Boundary Lines flanking each zebra crossing
  4. Zebra Crossing Stripes inside the two white boundary lines

Also saves high-res visualizations and JSON definitions.
"""

import os
import json
import cv2
import numpy as np

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FRAME_PATH = "work/frames/frame_0000000_t00000.00.jpg"
OUT_DIR = "work/crossing_zones"
os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(FRAME_PATH)
h, w = img.shape[:2]

# ──────────────────────────────────────────────────────────────
# 1. Roadway Boundary Polygon (matching user's media_1788427485778.png)
# Traced along the actual curbs / road edges of the intersection
# ──────────────────────────────────────────────────────────────
ROADWAY_BOUNDARY = [
    [60, 80],        # Top-left road edge
    [1000, 200],     # Along top curb
    [1530, 290],     # Top north road left curb
    [1620, 30],      # North road left corner
    [2430, 20],      # North road right corner
    [2480, 280],     # North road right curb
    [2580, 710],     # Corner near trees
    [3200, 950],     # Along northeast curb
    [3760, 1120],    # East road top curb
    [3820, 1140],    # East road upper edge
    [3820, 1620],    # East road lower edge
    [3100, 1720],    # Southeast road curb
    [2850, 1880],    # Corner near south road entrance
    [2420, 2140],    # South road right corner
    [1840, 2140],    # South road left corner
    [1420, 1760],    # South road left curb
    [1250, 1700],    # South curb along buildings
    [520, 1420],     # Southwest curb
    [60, 1220],      # West road lower edge
    [60, 80]         # Back to start
]

# ──────────────────────────────────────────────────────────────
# 2. White Boundary Lines for Zebra Crossings
# (The two transverse white lines marking the legal crossing area)
# ──────────────────────────────────────────────────────────────
# Upper-Left Crosswalk (Carriageway 1):
# Flanked by White Line 1 (left/west) and White Line 2 (right/east)
UL_WHITE_LINE_1 = [[970, 780], [1200, 350]]   # Western limit
UL_WHITE_LINE_2 = [[1100, 840], [1325, 415]]  # Eastern limit

# Lower-Right Crosswalk (Carriageway 2):
# Flanked by White Line 1 (left/west) and White Line 2 (right/east)
LR_WHITE_LINE_1 = [[2580, 1860], [2910, 1395]] # Western limit
LR_WHITE_LINE_2 = [[2730, 1915], [3060, 1450]] # Eastern limit

# ──────────────────────────────────────────────────────────────
# 3. Zebra Crossing Stripe Areas (Inside the two white boundary lines)
# ──────────────────────────────────────────────────────────────
UL_ZEBRA_POLYGON = [
    [980, 775],
    [1205, 355],
    [1315, 415],
    [1090, 835]
]

LR_ZEBRA_POLYGON = [
    [2590, 1855],
    [2920, 1400],
    [3050, 1455],
    [2720, 1910]
]

# ──────────────────────────────────────────────────────────────
# 4. Legal Pedestrian Crossing Areas
# Area A: Upper-Left Crosswalk Corridor (enclosed by the two white lines)
# Area B: Lower-Right Crosswalk Corridor (enclosed by the two white lines)
# Area C: Central Intersection Crossing Zone (corridor connecting the crossings
#         across the intersection, matching user's media_1788427514699.png)
# ──────────────────────────────────────────────────────────────
UL_CROSSING_CORRIDOR = [
    UL_WHITE_LINE_1[0],
    UL_WHITE_LINE_1[1],
    UL_WHITE_LINE_2[1],
    UL_WHITE_LINE_2[0]
]

LR_CROSSING_CORRIDOR = [
    LR_WHITE_LINE_1[0],
    LR_WHITE_LINE_1[1],
    LR_WHITE_LINE_2[1],
    LR_WHITE_LINE_2[0]
]

# Central intersection crossing corridor (matches the bow-tie / diamond
# drawn by the user in media_1788427514699.png connecting all four corners)
CENTRAL_CROSSING_ZONE = [
    [1580, 520],   # North traffic island / corner
    [1850, 680],   # Northeast inner crossing edge
    [2600, 980],   # East intersection boundary
    [2920, 1400],  # Meeting Lower-Right Zebra top
    [2590, 1855],  # Meeting Lower-Right Zebra bottom
    [2240, 1680],  # South road corner / curb
    [1800, 1480],  # Southwest inner crossing edge
    [1350, 1200],  # West crossing edge
    [1090, 835],   # Meeting Upper-Left Zebra bottom
    [1315, 415],   # Meeting Upper-Left Zebra top
]

# ──────────────────────────────────────────────────────────────
# Build Visualization
# ──────────────────────────────────────────────────────────────
vis = img.copy()
overlay = vis.copy()

# A. Roadway Boundary (Cyan outline with very light cyan tint)
road_pts = np.array(ROADWAY_BOUNDARY, dtype=np.int32)
cv2.fillPoly(overlay, [road_pts], (40, 20, 10))  # subtle dark tint for roadway
cv2.polylines(vis, [road_pts], True, (255, 200, 0), 4)  # bright cyan border

# B. Central Crossing Corridor (Semi-transparent Blue-Green)
central_pts = np.array(CENTRAL_CROSSING_ZONE, dtype=np.int32)
cv2.fillPoly(overlay, [central_pts], (0, 140, 70))
cv2.polylines(overlay, [central_pts], True, (0, 255, 128), 3)

# C. The Two White Boundary Lines (Bright White, Thicker)
for p1, p2 in [UL_WHITE_LINE_1, UL_WHITE_LINE_2, LR_WHITE_LINE_1, LR_WHITE_LINE_2]:
    cv2.line(vis, tuple(p1), tuple(p2), (255, 255, 255), 6)
    cv2.line(overlay, tuple(p1), tuple(p2), (255, 255, 255), 6)

# D. Zebra Crossings (Inside the two white boundary lines - Semi-transparent Yellow)
ul_z_pts = np.array(UL_ZEBRA_POLYGON, dtype=np.int32)
lr_z_pts = np.array(LR_ZEBRA_POLYGON, dtype=np.int32)

cv2.fillPoly(overlay, [ul_z_pts], (0, 215, 255))  # yellow in BGR
cv2.fillPoly(overlay, [lr_z_pts], (0, 215, 255))

cv2.polylines(vis, [ul_z_pts], True, (0, 215, 255), 3)
cv2.polylines(vis, [lr_z_pts], True, (0, 215, 255), 3)

# Blend overlay with image
vis = cv2.addWeighted(overlay, 0.38, vis, 0.62, 0)

# Redraw white boundary lines fully opaque on top
for p1, p2 in [UL_WHITE_LINE_1, UL_WHITE_LINE_2, LR_WHITE_LINE_1, LR_WHITE_LINE_2]:
    cv2.line(vis, tuple(p1), tuple(p2), (255, 255, 255), 5)

# Add Labels
font = cv2.FONT_HERSHEY_DUPLEX
cv2.putText(vis, "UPPER-LEFT ZEBRA CROSSING", (850, 320), font, 1.0, (0, 255, 255), 2)
cv2.putText(vis, "(Within Two White Boundary Lines)", (850, 355), font, 0.7, (255, 255, 255), 2)

cv2.putText(vis, "LOWER-RIGHT ZEBRA CROSSING", (2450, 1340), font, 1.0, (0, 255, 255), 2)
cv2.putText(vis, "(Within Two White Boundary Lines)", (2450, 1375), font, 0.7, (255, 255, 255), 2)

cv2.putText(vis, "PEDESTRIAN CROSSING CORRIDOR", (1650, 1100), font, 1.3, (0, 255, 128), 3)
cv2.putText(vis, "ROADWAY BOUNDARY", (260, 260), font, 1.1, (255, 200, 0), 2)

# Add Comprehensive Legend in top-left corner
cv2.rectangle(vis, (40, 40), (750, 310), (20, 20, 20), -1)
cv2.rectangle(vis, (40, 40), (750, 310), (200, 200, 200), 2)

cv2.putText(vis, "MAP ANNOTATION KEY", (60, 80), font, 0.9, (255, 255, 255), 2)

# Item 1: Roadway Boundary
cv2.line(vis, (60, 120), (120, 120), (255, 200, 0), 4)
cv2.putText(vis, "Roadway Boundary (Legal Road Perimeter)", (140, 125), font, 0.65, (255, 255, 255), 1)

# Item 2: Two White Boundary Lines
cv2.line(vis, (60, 160), (120, 160), (255, 255, 255), 5)
cv2.putText(vis, "Two White Boundary Lines (Crosswalk Limits)", (140, 165), font, 0.65, (255, 255, 255), 1)

# Item 3: Zebra Crossing
cv2.rectangle(vis, (60, 195), (120, 220), (0, 215, 255), -1)
cv2.putText(vis, "Zebra Crossing (Yellow Stripes Inside White Lines)", (140, 212), font, 0.65, (255, 255, 255), 1)

# Item 4: Pedestrian Crossing Zone
cv2.rectangle(vis, (60, 245), (120, 270), (0, 140, 70), -1)
cv2.putText(vis, "Pedestrian Crossing Corridor (Central Crossing Area)", (140, 262), font, 0.65, (255, 255, 255), 1)

out_vis_path = os.path.join(OUT_DIR, "perfect_crossing_zones_annotated.jpg")
cv2.imwrite(out_vis_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"Saved annotated image: {out_vis_path}")

# Save JSON specification
zones_dict = {
    "roadway_boundary": ROADWAY_BOUNDARY,
    "white_boundary_lines": {
        "upper_left": [UL_WHITE_LINE_1, UL_WHITE_LINE_2],
        "lower_right": [LR_WHITE_LINE_1, LR_WHITE_LINE_2]
    },
    "zebra_crossings": [
        {"id": 0, "name": "upper_left_zebra", "polygon": UL_ZEBRA_POLYGON},
        {"id": 1, "name": "lower_right_zebra", "polygon": LR_ZEBRA_POLYGON}
    ],
    "pedestrian_crossing_corridors": [
        {"id": 0, "name": "upper_left_corridor", "polygon": UL_CROSSING_CORRIDOR},
        {"id": 1, "name": "lower_right_corridor", "polygon": LR_CROSSING_CORRIDOR},
        {"id": 2, "name": "central_crossing_zone", "polygon": CENTRAL_CROSSING_ZONE}
    ],
    "resolution": [w, h],
    "pixel_to_meter_scale": 0.015
}

json_path = os.path.join(OUT_DIR, "crossing_zones_perfect.json")
with open(json_path, "w") as f:
    json.dump(zones_dict, f, indent=2)
print(f"Saved JSON specification: {json_path}")

# Also update work/zebra/zebra_crossings.json
with open("work/zebra/zebra_crossings.json", "w") as f:
    json.dump(zones_dict["zebra_crossings"], f, indent=2)
print("Updated work/zebra/zebra_crossings.json with 2 clean zebra polygons.")
