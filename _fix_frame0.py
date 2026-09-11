"""Regenerate corrupted demo_detections/frame_0000.jpg"""
import json, cv2, os

ROOT = r"h:\trajectory CLIP"
data = json.load(open(os.path.join(ROOT, "output", "results.json"), encoding="utf-8"))
frame = cv2.imread(os.path.join(ROOT, "output", "pipeline_test", "frame_0000.jpg"))
dets = [d for d in data["detections"] if d["frame_id"] == 0]

colors = {"vehicle": (0, 200, 0), "pedestrian": (255, 100, 0), "non_motor_vehicle": (200, 0, 255)}
labels = {"vehicle": "Vehicle", "pedestrian": "Pedestrian", "non_motor_vehicle": "NonMotor"}

for d in dets:
    x1, y1, x2, y2 = int(d["bbox"][0]), int(d["bbox"][1]), int(d["bbox"][2]), int(d["bbox"][3])
    color = colors.get(d["target_type"], (200, 200, 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{labels.get(d['target_type'], d['target_type'])} {d['confidence']:.0%}"
    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

out_path = os.path.join(ROOT, "output", "demo_detections", "frame_0000.jpg")
cv2.imwrite(out_path, frame)
print(f"Regenerated {out_path} with {len(dets)} detections, size={os.path.getsize(out_path)}")
