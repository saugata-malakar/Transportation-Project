"""
generate_vivid_viewer.py

Creates an offline single-image interactive viewer for all standalone frames.
"""
import os
import json

ROOT = r"c:\Users\Administrator\Downloads\TRANSPORTATION\codebase\work\vivid_standalone_frames"

peds = sorted([f for f in os.listdir(os.path.join(ROOT, "pedestrians")) if f.endswith(".jpg")])
gait = sorted([f for f in os.listdir(os.path.join(ROOT, "gait_steps")) if f.endswith(".jpg")])
groups = sorted([f for f in os.listdir(os.path.join(ROOT, "group_categories")) if f.endswith(".jpg")])
zones = sorted([f for f in os.listdir(os.path.join(ROOT, "spatial_zones")) if f.endswith(".jpg")])

all_items = []
for f in peds:
    all_items.append({"cat": "Pedestrians (All 49)", "name": f.replace("pedestrian_", "").replace(".jpg", ""), "path": f"pedestrians/{f}"})
for f in gait:
    all_items.append({"cat": "Gait Steps (1 Step/Frame)", "name": f.replace(".jpg", ""), "path": f"gait_steps/{f}"})
for f in groups:
    all_items.append({"cat": "Group Categories", "name": f.replace(".jpg", ""), "path": f"group_categories/{f}"})
for f in zones:
    all_items.append({"cat": "Spatial Zones", "name": f.replace(".jpg", ""), "path": f"spatial_zones/{f}"})

items_json = json.dumps(all_items)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Standalone Vivid Frames Physical Verification Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background: #0b0f17; color: #e2e8f0; display: flex; height: 100vh; overflow: hidden; }}
        #sidebar {{ width: 360px; background: #131b26; border-right: 1px solid #1e293b; display: flex; flex-direction: column; }}
        #header {{ padding: 16px; background: #0f172a; border-bottom: 1px solid #1e293b; }}
        #header h1 {{ font-size: 15px; color: #38bdf8; margin-bottom: 4px; }}
        #header p {{ font-size: 11px; color: #94a3b8; }}
        #categories {{ padding: 10px 16px; background: #172033; display: flex; flex-wrap: wrap; gap: 6px; border-bottom: 1px solid #1e293b; }}
        .cat-btn {{ background: #243046; border: none; color: #94a3b8; padding: 5px 10px; border-radius: 4px; font-size: 11px; cursor: pointer; }}
        .cat-btn.active {{ background: #0284c7; color: #fff; font-weight: bold; }}
        #nav-controls {{ padding: 8px 16px; background: #131b26; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; }}
        .nav-btn {{ background: #1e293b; border: 1px solid #334155; color: #cbd5e1; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold; }}
        .nav-btn:hover {{ background: #0284c7; color: #fff; border-color: #38bdf8; }}
        #counter {{ font-size: 11px; color: #94a3b8; }}
        #file-list {{ flex: 1; overflow-y: auto; padding: 8px; }}
        .file-item {{ padding: 8px 12px; border-radius: 4px; font-size: 12px; cursor: pointer; margin-bottom: 4px; background: #1a2333; color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border: 1px solid transparent; }}
        .file-item:hover {{ background: #243046; border-color: #0284c7; }}
        .file-item.selected {{ background: #1e3a5f; color: #38bdf8; border-color: #38bdf8; font-weight: bold; }}
        #viewport {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 16px; background: #080c14; }}
        #main-frame {{ max-width: 100%; max-height: 94vh; object-fit: contain; border-radius: 6px; border: 1px solid #1e293b; box-shadow: 0 10px 30px rgba(0,0,0,0.8); }}
    </style>
</head>
<body>
    <div id="sidebar">
        <div id="header">
            <h1>ONE IMAGE PER FRAME VIEWER</h1>
            <p>Standalone High-Definition Computer Vision Verification</p>
        </div>
        <div id="categories">
            <button class="cat-btn active" onclick="setCategory('All')">All ({len(all_items)})</button>
            <button class="cat-btn" onclick="setCategory('Pedestrians (All 49)')">Pedestrians (49)</button>
            <button class="cat-btn" onclick="setCategory('Gait Steps (1 Step/Frame)')">Gait Steps (8)</button>
            <button class="cat-btn" onclick="setCategory('Group Categories')">Groups (3)</button>
            <button class="cat-btn" onclick="setCategory('Spatial Zones')">Zones (3)</button>
        </div>
        <div id="nav-controls">
            <button class="nav-btn" onclick="prevFrame()">&larr; Previous</button>
            <span id="counter">1 / {len(all_items)}</span>
            <button class="nav-btn" onclick="nextFrame()">Next &rarr;</button>
        </div>
        <div id="file-list"></div>
    </div>
    <div id="viewport">
        <img id="main-frame" src="" alt="Select an image">
    </div>

    <script>
        const items = {items_json};
        let currentCategory = 'All';
        let filteredItems = items;
        let currentIndex = 0;

        function setCategory(cat) {{
            currentCategory = cat;
            document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            filteredItems = (cat === 'All') ? items : items.filter(i => i.cat === cat);
            currentIndex = 0;
            renderList();
            showCurrent();
        }}

        function renderList() {{
            const listEl = document.getElementById('file-list');
            listEl.innerHTML = '';
            filteredItems.forEach((item, idx) => {{
                const div = document.createElement('div');
                div.className = 'file-item' + (idx === currentIndex ? ' selected' : '');
                div.innerText = item.name;
                div.onclick = () => {{
                    currentIndex = idx;
                    showCurrent();
                }};
                listEl.appendChild(div);
            }});
        }}

        function showCurrent() {{
            if (filteredItems.length === 0) return;
            const item = filteredItems[currentIndex];
            document.getElementById('main-frame').src = item.path;
            document.getElementById('counter').innerText = `${{currentIndex + 1}} / ${{filteredItems.length}}`;
            document.querySelectorAll('.file-item').forEach((el, idx) => {{
                el.classList.toggle('selected', idx === currentIndex);
            }});
        }}

        function nextFrame() {{
            if (currentIndex < filteredItems.length - 1) {{
                currentIndex++;
                showCurrent();
            }}
        }}

        function prevFrame() {{
            if (currentIndex > 0) {{
                currentIndex--;
                showCurrent();
            }}
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') nextFrame();
            if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') prevFrame();
        }});

        renderList();
        showCurrent();
    </script>
</body>
</html>
"""

viewer_path = os.path.join(ROOT, "vivid_frames_viewer.html")
with open(viewer_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Generated standalone vivid viewer: {viewer_path}")
