import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import cv2
from collections import Counter
from ultralytics import YOLO
m = YOLO('yolov8x.pt')
print('Model loaded')

# Test on VisDrone-MOT image
for img_path in ['data/VisDrone-MOT/images/0000001.jpg', 'data/VisDrone2019-DET/images/0000002_00005_d_0000014.jpg']:
    img = cv2.imread(img_path)
    if img is None:
        print(f'Failed to load: {img_path}')
        continue
    print(f'\n=== {img_path} ===')
    print(f'Image shape: {img.shape}')
    for conf_t in [0.05, 0.1, 0.3]:
        r = m.predict(source=img, conf=conf_t, imgsz=[640,640], device='cuda', verbose=False)
        boxes = r[0].boxes
        cls_ids = [int(c) for c in boxes.cls]
        c = Counter(cls_ids)
        names = r[0].names
        mapped = {names[k]: v for k, v in c.items()}
        print(f'  conf={conf_t}: total={len(boxes)}, classes={mapped}')
    # Check vehicle confidences specifically
    r = m.predict(source=img, conf=0.05, imgsz=[640,640], device='cuda', verbose=False)
    boxes = r[0].boxes
    veh_ids = {0,1,2,3,5,7}
    for i in range(len(boxes)):
        cid = int(boxes.cls[i])
        if cid in veh_ids:
            print(f'  Vehicle det: cls={r[0].names[cid]}({cid}), conf={float(boxes.conf[i]):.3f}, box={boxes.xyxy[i].cpu().tolist()}')
