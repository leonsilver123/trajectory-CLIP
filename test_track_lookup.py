# -*- coding: utf-8 -*-
"""Quick check: why track lookup fails."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('output/cityflow_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tracks = data.get('tracks', [])
d2t = data.get('det_to_track_map', {})
dets = data.get('detections', [])

# Build set of all track_ids
track_ids = set(t.get('track_id', '') for t in tracks)
print(f"Total tracks: {len(tracks)}")
print(f"Unique track_ids in tracks: {len(track_ids)}")
print(f"Sample track_ids: {list(track_ids)[:5]}")

# Check what track_ids det_to_track_map references
mapped_track_ids = set(d2t.values())
print(f"\nUnique track_ids in det_to_track_map values: {len(mapped_track_ids)}")
print(f"Sample mapped track_ids: {list(mapped_track_ids)[:5]}")

# Check overlap
overlap = track_ids & mapped_track_ids
print(f"\nOverlap: {len(overlap)}")
missing = mapped_track_ids - track_ids
print(f"Track IDs in map but NOT in tracks: {len(missing)}")
if missing:
    print(f"  Examples: {list(missing)[:5]}")

# Check target_id format in detections
print(f"\nSample target_ids in detections:")
for d in dets[:3]:
    tid = d.get('target_id', '')
    mapped = d2t.get(tid, 'NOT IN MAP')
    print(f"  target_id={tid}, mapped_track={mapped}")
    # Check if this track_id is in tracks set
    print(f"    in tracks set: {mapped in track_ids}")

# Check track vehicle_id format
print(f"\nSample track vehicle_ids:")
for t in tracks[:3]:
    print(f"  track_id={t.get('track_id')}, vehicle_id={t.get('vehicle_id')}, source={t.get('source')}")
