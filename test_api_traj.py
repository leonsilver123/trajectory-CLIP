# -*- coding: utf-8 -*-
"""Test API search results for trajectory fields."""
import json, sys, requests
sys.stdout.reconfigure(encoding='utf-8')

try:
    r = requests.post('http://localhost:8000/api/v1/search/query', 
                      json={'query_text': 'white car', 'top_k': 3}, timeout=10)
    data = r.json()
    results = data.get('candidates', data.get('results', data.get('data', {}).get('results', [])))
    print(f"API status: {r.status_code}, results count: {len(results)}")
    if results:
        item = results[0]
        print(f"\nResult[0] keys: {list(item.keys())}")
        print(f"  track_id: {item.get('track_id')}")
        print(f"  has_trajectory: {item.get('has_trajectory')}")
        print(f"  trajectory_frame_range: {item.get('trajectory_frame_range')}")
        print(f"  trajectory_camera_ids: {item.get('trajectory_camera_ids')}")
        print(f"  trajectory_detection_count: {item.get('trajectory_detection_count')}")
        print(f"  track_frames: {item.get('track_frames', 'NOT PRESENT')}")
        print(f"  track_cameras: {item.get('track_cameras', 'NOT PRESENT')}")
        print(f"  track_frame_count: {item.get('track_frame_count', 'NOT PRESENT')}")
    else:
        print("No results returned")
        print(f"Full response: {json.dumps(data, ensure_ascii=False)[:500]}")
except Exception as e:
    print(f"API call failed: {e}")
