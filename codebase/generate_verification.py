"""
Visual verification tool — generates image grids showing every detected person
with their predicted gender, so you can physically verify the results.

Outputs:
  work/verification/all_persons_grid.jpg         — All 49 persons in one image
  work/verification/males_grid.jpg               — All males
  work/verification/females_grid.jpg             — All females
  work/verification/person_<ID>/                  — Individual person folders with all crops
"""

import os
import json
import cv2
import numpy as np

def make_grid(images_with_labels, cols=8, cell_size=150, font_scale=0.45):
    """Create a labelled grid image from list of (image, label_text) tuples."""
    n = len(images_with_labels)
    if n == 0:
        return np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
    
    rows = (n + cols - 1) // cols
    grid_h = rows * (cell_size + 30)  # 30px for label
    grid_w = cols * cell_size
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255  # white background
    
    for idx, (img, label) in enumerate(images_with_labels):
        r = idx // cols
        c = idx % cols
        y_off = r * (cell_size + 30)
        x_off = c * cell_size
        
        # Resize image to fit cell
        if img is not None and img.size > 0:
            h, w = img.shape[:2]
            scale = min(cell_size / w, cell_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            
            # Center in cell
            pad_x = (cell_size - new_w) // 2
            pad_y = (cell_size - new_h) // 2
            grid[y_off + pad_y:y_off + pad_y + new_h, 
                 x_off + pad_x:x_off + pad_x + new_w] = resized
        
        # Draw border
        cv2.rectangle(grid, (x_off, y_off), (x_off + cell_size - 1, y_off + cell_size - 1),
                      (200, 200, 200), 1)
        
        # Draw label below image
        label_y = y_off + cell_size + 18
        # Color code: blue for male, pink for female
        if "Male" in label:
            color = (255, 100, 0)  # blue (BGR)
        elif "Female" in label:
            color = (147, 20, 255)  # pink (BGR)
        else:
            color = (0, 0, 0)
        
        cv2.putText(grid, label, (x_off + 3, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    
    return grid


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)
    
    out_dir = "work/verification"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    with open("work/crops/crop_index.json") as f:
        crop_index = json.load(f)
    with open("work/output/FINAL_complete_results.json") as f:
        results = json.load(f)
    
    # Build lookup
    gender_map = {r["person_id"]: r["gender"] for r in results}
    conf_map = {r["person_id"]: r["attribute_confidence"] for r in results}
    
    all_items = []
    male_items = []
    female_items = []
    
    persons = crop_index.get("persons", {})
    sorted_ids = sorted(persons.keys(), key=lambda x: int(x))
    
    for pid in sorted_ids:
        crops = persons[pid]
        if not crops:
            continue
        
        # Pick the best crop (first one, or highest confidence)
        best_crop_path = crops[0].get("crop_file", "")
        img = cv2.imread(best_crop_path)
        
        if img is None:
            # Try alternative paths
            for c in crops:
                img = cv2.imread(c.get("crop_file", ""))
                if img is not None:
                    break
        
        gender = gender_map.get(pid, "?")
        conf = conf_map.get(pid, 0)
        label = f"#{pid} {gender} {conf:.0%}"
        
        all_items.append((img, label))
        if gender == "Male":
            male_items.append((img, label))
        elif gender == "Female":
            female_items.append((img, label))
        
        # Save individual person folder with all crops
        person_dir = os.path.join(out_dir, f"person_{pid}_{gender}")
        os.makedirs(person_dir, exist_ok=True)
        for i, c in enumerate(crops):
            src = c.get("crop_file", "")
            if os.path.exists(src):
                dst = os.path.join(person_dir, f"crop_{i:02d}.jpg")
                if not os.path.exists(dst):
                    img_c = cv2.imread(src)
                    if img_c is not None:
                        cv2.imwrite(dst, img_c)
    
    # Generate grids
    print(f"Generating verification grids...")
    print(f"  Total persons: {len(all_items)}")
    print(f"  Males: {len(male_items)}")
    print(f"  Females: {len(female_items)}")
    
    # All persons grid
    all_grid = make_grid(all_items, cols=10, cell_size=120)
    all_path = os.path.join(out_dir, "ALL_49_persons_grid.jpg")
    cv2.imwrite(all_path, all_grid)
    print(f"  -> {all_path}")
    
    # Males grid
    male_grid = make_grid(male_items, cols=8, cell_size=140)
    male_path = os.path.join(out_dir, "MALES_33_grid.jpg")
    cv2.imwrite(male_path, male_grid)
    print(f"  -> {male_path}")
    
    # Females grid
    female_grid = make_grid(female_items, cols=8, cell_size=140)
    female_path = os.path.join(out_dir, "FEMALES_16_grid.jpg")
    cv2.imwrite(female_path, female_grid)
    print(f"  -> {female_path}")
    
    print(f"\nVERIFICATION INSTRUCTIONS:")
    print(f"  1. Open {all_path}")
    print(f"     -> Count the people. There should be 49 unique persons.")
    print(f"  2. Open {male_path}")
    print(f"     -> Look at each crop. Verify these look male. (33 persons)")
    print(f"  3. Open {female_path}")
    print(f"     -> Look at each crop. Verify these look female. (16 persons)")
    print(f"  4. Browse work/verification/person_<ID>_<Gender>/ folders")
    print(f"     -> Each folder has 3-8 different angle crops of the SAME person.")
    print(f"     -> Check that each person is truly one individual, not duplicates.")
    print(f"\n  If any gender looks wrong, note the Person # and we can correct it.")


if __name__ == "__main__":
    main()
