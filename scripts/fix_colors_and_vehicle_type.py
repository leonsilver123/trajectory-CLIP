"""
Fix colors (HSV red detection) and vehicle type classification for cityflow_results.json
"""
import json
import cv2
import numpy as np
import os
import sys
from collections import Counter

# Optimized HSV color ranges
COLOR_RANGES = {
    '白色': ((0, 0, 200), (180, 30, 255)),
    '黑色': ((0, 0, 0), (180, 255, 50)),
    '红色': [  # Red spans two ranges in HSV hue wheel
        ((0, 100, 100), (10, 255, 255)),
        ((170, 100, 100), (180, 255, 255)),
    ],
    '蓝色': ((100, 100, 100), (130, 255, 255)),
    '绿色': ((40, 100, 100), (80, 255, 255)),
    '黄色': ((20, 100, 100), (35, 255, 255)),
    '灰色': ((0, 0, 50), (180, 50, 200)),
    '棕色': ((10, 50, 50), (25, 150, 150)),
}


def detect_color_hsv(image_array):
    """Detect dominant color using HSV color space with multi-range support."""
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    color_scores = {}
    total_pixels = hsv.shape[0] * hsv.shape[1]

    for color_name, ranges in COLOR_RANGES.items():
        if isinstance(ranges, list):
            # Multi-range (e.g. red)
            total_score = 0
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                total_score += np.sum(mask > 0) / total_pixels
            color_scores[color_name] = total_score
        else:
            lower, upper = ranges
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_scores[color_name] = np.sum(mask > 0) / total_pixels

    best_color = max(color_scores, key=color_scores.get)
    if color_scores[best_color] < 0.1:
        return 'unknown'
    return best_color


def classify_vehicle_type(det):
    """Classify vehicle type based on bbox aspect ratio."""
    bbox = det.get('bbox', [0, 0, 0, 0])
    if len(bbox) == 4:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    else:
        return 'unknown'

    if w <= 0 or h <= 0:
        return 'unknown'

    aspect_ratio = w / h

    if aspect_ratio > 2.5:
        return '卡车'
    elif aspect_ratio > 1.8:
        return 'SUV'
    elif aspect_ratio > 1.2:
        return '轿车'
    elif aspect_ratio > 0.8:
        return '面包车'
    else:
        return 'unknown'


def main():
    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(project_root, "output", "cityflow_results.json")

    # Allow limiting records for quick testing
    max_records = None
    if len(sys.argv) > 1:
        try:
            max_records = int(sys.argv[1])
            print(f"[LIMIT] Processing only first {max_records} detections")
        except ValueError:
            pass

    print(f"Loading data from {results_path} ...")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    detections = data.get("detections", [])
    tracks = data.get("tracks", [])
    print(f"Total detections: {len(detections)}, tracks: {len(tracks)}")

    color_counts = Counter()
    type_counts = Counter()
    updated = 0
    color_changed = 0
    no_crop = 0

    limit = len(detections) if max_records is None else min(max_records, len(detections))

    for i in range(limit):
        det = detections[i]
        crop_path = det.get("crop_path", "")

        # 1. Re-detect color with optimized red range
        if crop_path and os.path.exists(crop_path):
            try:
                img = cv2.imread(crop_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    detected_color = detect_color_hsv(img_rgb)
                    old_color = det.get('attributes', {}).get('颜色', 'unknown')
                    if detected_color != 'unknown':
                        if 'attributes' not in det:
                            det['attributes'] = {}
                        det['attributes']['颜色'] = detected_color
                        det['attributes']['color'] = detected_color
                        color_counts[detected_color] += 1
                        if detected_color != old_color:
                            color_changed += 1
                    else:
                        # Keep old color if detection fails
                        old = det.get('attributes', {}).get('颜色', 'unknown')
                        if old != 'unknown':
                            color_counts[old] += 1
                else:
                    no_crop += 1
            except Exception as e:
                no_crop += 1
        else:
            no_crop += 1

        # 2. Vehicle type classification
        vtype = classify_vehicle_type(det)
        if vtype != 'unknown':
            if 'attributes' not in det:
                det['attributes'] = {}
            det['attributes']['车型'] = vtype
            det['attributes']['vehicle_type'] = vtype
            type_counts[vtype] += 1

        updated += 1
        if updated % 5000 == 0:
            print(f"  Processed {updated}/{limit} ... (colors so far: {dict(color_counts)})")

    print(f"\nColor re-detection done. Changed: {color_changed}, No crop: {no_crop}")

    # 3. Update track attributes (voting aggregation)
    print("Aggregating track attributes ...")
    det_to_track_map = data.get("det_to_track_map", {})
    track_to_dets = {}
    for det_idx in range(limit):
        det = detections[det_idx]
        target_id = det.get("target_id", "")
        track_id = det_to_track_map.get(target_id, "")
        if track_id:
            if track_id not in track_to_dets:
                track_to_dets[track_id] = []
            track_to_dets[track_id].append(det)

    track_color_updated = 0
    track_type_updated = 0
    for track in tracks:
        track_id = track.get("track_id", "")
        track_dets = track_to_dets.get(track_id, [])
        if track_dets:
            # Color voting
            colors = [
                d['attributes'].get('颜色', 'unknown')
                for d in track_dets
                if d.get('attributes', {}).get('颜色', 'unknown') != 'unknown'
            ]
            if colors:
                most_common = Counter(colors).most_common(1)[0][0]
                if 'attributes' not in track:
                    track['attributes'] = {}
                track['attributes']['颜色'] = most_common
                track['attributes']['color'] = most_common
                track_color_updated += 1

            # Vehicle type voting
            types = [
                d['attributes'].get('车型', 'unknown')
                for d in track_dets
                if d.get('attributes', {}).get('车型', 'unknown') != 'unknown'
            ]
            if types:
                most_common_type = Counter(types).most_common(1)[0][0]
                if 'attributes' not in track:
                    track['attributes'] = {}
                track['attributes']['车型'] = most_common_type
                track['attributes']['vehicle_type'] = most_common_type
                track_type_updated += 1

    print(f"Track color updated: {track_color_updated}/{len(tracks)}")
    print(f"Track type updated: {track_type_updated}/{len(tracks)}")

    # 4. Save
    if max_records is None:
        print("Saving full results ...")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved to {results_path}")
    else:
        print("[LIMIT] Skipping save for partial run")

    print(f"\n=== Summary ===")
    print(f"Color distribution: {dict(color_counts)}")
    print(f"Type distribution: {dict(type_counts)}")
    print(f"Total updated: {updated}")


if __name__ == "__main__":
    main()
