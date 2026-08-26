import os
import urllib.request

os.makedirs("checkpoints", exist_ok=True)

models = {
    "yolov8n.pt": "https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.pt",
    "yolov8s.pt": "https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8s.pt",
    "yolov8m.pt": "https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8m.pt",
    "yolo11n.pt": "https://huggingface.co/ultralytics/yolo11/resolve/main/yolo11n.pt",
    "yolo11n-pose.pt": "https://huggingface.co/ultralytics/yolo11/resolve/main/yolo11n-pose.pt",
    "checkpoints/yolov8n.pt": "https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.pt",
    "checkpoints/yolo11n.pt": "https://huggingface.co/ultralytics/yolo11/resolve/main/yolo11n.pt",
    "checkpoints/yolo11n-pose.pt": "https://huggingface.co/ultralytics/yolo11/resolve/main/yolo11n-pose.pt"
}

for dest, url in models.items():
    if os.path.exists(dest) and os.path.getsize(dest) > 1000000:
        print(f"Already exists: {dest} ({os.path.getsize(dest)/1024/1024:.1f} MB)")
        continue
    print(f"Downloading {dest} from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest, 'wb') as f:
        f.write(resp.read())
    print(f"Saved: {dest} ({os.path.getsize(dest)/1024/1024:.1f} MB)")

print("All requested models downloaded successfully!")
