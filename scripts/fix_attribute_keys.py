"""
fix_attribute_keys.py
Fix English attribute keys to Chinese in cityflow_results.json
and remove redundant fields (color_refined, color_en, color_id).
"""
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "output", "cityflow_results.json")

REDUNDANT_KEYS = {"color_refined", "color_en", "color_id"}
KEY_MAP = {
    "color": "颜色",
    "vehicle_type": "车型",
}


def fix_attributes(attrs: dict) -> dict:
    # Add Chinese key aliases
    for en_key, cn_key in KEY_MAP.items():
        if en_key in attrs:
            attrs[cn_key] = attrs[en_key]
    # Remove redundant fields
    for k in REDUNDANT_KEYS:
        attrs.pop(k, None)
    return attrs


def main():
    print(f"Loading: {JSON_PATH}")
    t0 = time.time()
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded in {time.time() - t0:.1f}s")

    det_count = len(data.get("detections", []))
    trk_count = len(data.get("tracks", []))
    print(f"Detections: {det_count}, Tracks: {trk_count}")

    # Fix detections
    det_fixed = 0
    for det in data.get("detections", []):
        attrs = det.get("attributes", {})
        if attrs:
            before = set(attrs.keys())
            fix_attributes(attrs)
            if set(attrs.keys()) != before:
                det_fixed += 1
    print(f"Detections modified: {det_fixed}")

    # Fix tracks
    trk_fixed = 0
    for trk in data.get("tracks", []):
        attrs = trk.get("attributes", {})
        if attrs:
            before = set(attrs.keys())
            fix_attributes(attrs)
            if set(attrs.keys()) != before:
                trk_fixed += 1
    print(f"Tracks modified: {trk_fixed}")

    # Save
    print("Saving...")
    t0 = time.time()
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved in {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
