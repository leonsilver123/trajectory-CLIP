# -*- coding: utf-8 -*-
"""Focused diagnostic for trajectory data integrity."""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

results_path = os.path.join('output', 'cityflow_results.json')
with open(results_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

tracks = data.get('tracks', [])
dets = data.get('detections', [])
d2t = data.get('det_to_track_map', {})

print(f"=== DATA OVERVIEW ===")
print(f"detections: {len(dets)}")
print(f"tracks: {len(tracks)}")
print(f"det_to_track_map entries: {len(d2t)}")

# Check track structure
if tracks:
    t0 = tracks[0]
    print(f"\n=== TRACK[0] KEYS: {list(t0.keys())}")
    print(f"track_id={t0.get('track_id')}, frame_range={t0.get('frame_range')}, camera_ids={t0.get('camera_ids')}, detection_count={t0.get('detection_count')}")
    has_dets_field = 'detections' in t0
    print(f"has 'detections' field: {has_dets_field}")
    if has_dets_field:
        print(f"  detections count: {len(t0['detections'])}")
        if t0['detections']:
            print(f"  first det keys: {list(t0['detections'][0].keys())}")

    # Show 2 more tracks briefly
    for i in [1, min(5, len(tracks)-1)]:
        t = tracks[i]
        print(f"\ntrack[{i}]: id={t.get('track_id')}, frames={t.get('frame_range')}, cams={t.get('camera_ids')}, det_count={t.get('detection_count')}, has_dets_field={'detections' in t}")

# Check det_to_track_map sample
if d2t:
    sample_keys = list(d2t.keys())[:3]
    print(f"\n=== det_to_track_map sample ===")
    for k in sample_keys:
        print(f"  {k} -> {d2t[k]}")

# Check detection structure
if dets:
    print(f"\n=== DETECTION[0] KEYS: {list(dets[0].keys())}")
    print(f"  sample: id={dets[0].get('id')}, track_id={dets[0].get('track_id')}, camera_id={dets[0].get('camera_id')}, frame_id={dets[0].get('frame_id')}")

# Build reverse map: track_id -> detections
print(f"\n=== BUILDING REVERSE MAP ===")
track_to_dets = {}
for d in dets:
    tid = d.get('id', '')
    track_id = d2t.get(tid, '')
    if track_id:
        track_to_dets.setdefault(track_id, []).append(d)
print(f"track_to_dets entries: {len(track_to_dets)}")
if track_to_dets:
    first_tid = list(track_to_dets.keys())[0]
    print(f"  track '{first_tid}' has {len(track_to_dets[first_tid])} detections")

print("\n=== DONE ===")
