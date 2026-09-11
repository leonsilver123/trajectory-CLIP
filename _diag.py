import json
from collections import defaultdict

path = "output/cityflow_results.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dets = data.get("detections", [])
tracks = data.get("tracks", [])
print(f"detections: {len(dets)}")
print(f"tracks: {len(tracks)}")

# Group by vehicle_id
vd = defaultdict(set)
for d in dets:
    tid = d.get("target_id", "")
    parts = tid.split("_")
    vid = ""
    for p in parts:
        if p.startswith("V") and p[1:].isdigit():
            vid = p
            break
    if vid:
        vd[vid].add(d.get("camera_id", ""))

print(f"unique vehicles: {len(vd)}")
multi = [v for v, c in vd.items() if len(c) >= 2]
print(f"vehicles in >=2 cameras: {len(multi)}")
for v in list(multi)[:5]:
    print(f"  {v}: {sorted(vd[v])}")
single = [v for v, c in vd.items() if len(c) == 1]
print(f"vehicles in 1 camera: {len(single)}")

# check track structure
if tracks:
    t0 = tracks[0]
    print(f"\nTrack keys: {list(t0.keys())}")
    print(f"Track sample: {json.dumps(t0, ensure_ascii=False)[:500]}")

# check what instance_id looks like in search results
print("\nSample detection target_ids:")
for d in dets[:5]:
    print(f"  {d['target_id']}")
