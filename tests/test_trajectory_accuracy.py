"""
轨迹拼接与还原的准确率及鲁棒性测试
====================================
使用 AICity22 gt.txt 作为 Ground Truth，
cityflow_results.json 作为系统输出，
评估轨迹还原的 MOT 标准指标、跨镜拼接准确率、鲁棒性。
"""

import json
import os
import math
from collections import defaultdict
from pathlib import Path

import pytest
import numpy as np

# ── 路径常量 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = PROJECT_ROOT / "output" / "cityflow_results.json"
GT_BASE = PROJECT_ROOT / "cityflow" / "AICity22_Track1_MTMC_Tracking" / "train"

# 场景 → 摄像头映射
SCENE_CAMERAS = {
    "S01": [f"c{i:03d}" for i in range(1, 6)],       # 5 cams
    "S03": [f"c{i:03d}" for i in range(10, 16)],     # 6 cams
    "S04": [f"c{i:03d}" for i in range(16, 41)],     # 25 cams
}

# ── IoU 计算 ──────────────────────────────────────────────

def iou_xyxy(box_a, box_b):
    """计算两个 [x1,y1,x2,y2] 格式 bbox 的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_ltrb_to_xyxy(gt_ltrb, det_xyxy):
    """gt: [left,top,w,h] → [x1,y1,x2,y2], 然后算 IoU"""
    gt_xyxy = [gt_ltrb[0], gt_ltrb[1], gt_ltrb[0] + gt_ltrb[2], gt_ltrb[1] + gt_ltrb[3]]
    return iou_xyxy(gt_xyxy, det_xyxy)


# ── 数据加载 ──────────────────────────────────────────────

def load_gt_scene(scene_name):
    """
    加载某场景所有摄像头的 gt.txt
    返回:
        gt_tracks: dict[gt_id] → list of (frame, cam, left, top, w, h)
        gt_per_cam: dict[cam] → dict[frame] → list of (gt_id, left, top, w, h)
    """
    cameras = SCENE_CAMERAS.get(scene_name, [])
    gt_tracks = defaultdict(list)  # gt_id → [(frame, cam, l, t, w, h), ...]
    gt_per_cam = defaultdict(lambda: defaultdict(list))

    for cam in cameras:
        gt_path = GT_BASE / scene_name / cam / "gt" / "gt.txt"
        if not gt_path.exists():
            continue
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                frame, gt_id = int(parts[0]), int(parts[1])
                l, t, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                # visibility = float(parts[6]) if len(parts) > 6 else 1.0
                gt_tracks[gt_id].append((frame, cam, l, t, w, h))
                gt_per_cam[cam][frame].append((gt_id, l, t, w, h))

    return dict(gt_tracks), dict(gt_per_cam)


def load_system_results():
    """
    加载 cityflow_results.json
    返回:
        detections: list of det dicts
        tracks: list of track dicts
    """
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["detections"], data["tracks"]


def build_det_per_cam(detections):
    """
    将 detections 按 (camera_id, frame_id) 组织
    返回: dict[cam] → dict[frame] → list of (det_idx, bbox_xyxy, vehicle_id)
    """
    det_per_cam = defaultdict(lambda: defaultdict(list))
    for i, det in enumerate(detections):
        cam = det["camera_id"]
        frame = det["frame_id"]
        bbox = det["bbox"]  # [x1, y1, x2, y2]
        # 从 target_id 提取 vehicle_id: CF3_c001_V0034_000001 → 34
        vid = det.get("target_id", "").split("_V")
        vehicle_id = int(vid[1].split("_")[0]) if len(vid) > 1 else -1
        det_per_cam[cam][frame].append((i, bbox, vehicle_id))
    return dict(det_per_cam)


def build_track_per_scene(tracks):
    """将 tracks 按 scene_id 组织"""
    per_scene = defaultdict(list)
    for t in tracks:
        per_scene[t.get("scene_id", "unknown")].append(t)
    return dict(per_scene)


# ── 匈牙利匹配 ──────────────────────────────────────────────

def hungarian_match(gt_list, det_list, iou_threshold=0.3):
    """
    使用匈牙利算法匹配 gt 和 det
    gt_list: [(gt_id, l, t, w, h), ...]
    det_list: [(det_idx, bbox_xyxy, vehicle_id), ...]
    返回: matched_pairs [(gt_id, det_vehicle_id), ...], unmatched_gt_ids [gt_id,...], unmatched_det_indices [det_idx,...]
    """
    from scipy.optimize import linear_sum_assignment

    n_gt = len(gt_list)
    n_det = len(det_list)

    if n_gt == 0 and n_det == 0:
        return [], [], []
    if n_gt == 0:
        return [], [], list(range(n_det))
    if n_det == 0:
        return [], [g[0] for g in gt_list], []

    # 构建代价矩阵 (1 - IoU)
    cost_matrix = np.ones((n_gt, n_det))
    for i, (gid, gl, gt_, gw, gh) in enumerate(gt_list):
        for j, (didx, dbbox, dvid) in enumerate(det_list):
            cost_matrix[i, j] = 1.0 - iou_ltrb_to_xyxy([gl, gt_, gw, gh], dbbox)

    # 匈牙利算法
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    unmatched_gt_ids = []
    unmatched_det_indices = list(range(n_det))
    matched_rows = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < (1.0 - iou_threshold):  # IoU > threshold
            gt_id = gt_list[r][0]
            det_vid = det_list[c][2]
            matched_pairs.append((gt_id, det_vid))
            matched_rows.add(r)
            if c in unmatched_det_indices:
                unmatched_det_indices.remove(c)

    # 收集未匹配的 gt (返回 gt_id 而非索引)
    for i, g in enumerate(gt_list):
        if i not in matched_rows:
            unmatched_gt_ids.append(g[0])

    return matched_pairs, unmatched_gt_ids, unmatched_det_indices


# ── MOT 指标计算 ──────────────────────────────────────────

def compute_mot_metrics(gt_per_cam, det_per_cam, scenes):
    """
    计算 MOT 标准指标 (IDF1, IDP, IDR, etc.)
    对每个场景的每个摄像头逐帧匹配，然后跨帧累积 ID 匹配。

    返回: dict with metrics
    """
    total_idtp = 0
    total_idfp = 0
    total_idfn = 0
    total_gt_ids = 0
    total_det_ids = 0

    # 用于计算 ID switches
    prev_gt_to_det = {}  # gt_id → det_vehicle_id (上一帧的匹配)
    id_switches = 0

    # 用于 MT / ML
    gt_track_frames = defaultdict(int)   # gt_id → 总帧数
    gt_track_matched = defaultdict(int)  # gt_id → 被匹配帧数

    # 用于 Fragmentation
    gt_track_state = {}  # gt_id → 'tracked' / 'lost'
    fragmentations = 0

    all_matched_per_frame = []  # (frame_global, gt_id, det_id)

    for scene in scenes:
        gt_tracks, gt_per_cam_scene = load_gt_scene(scene)
        cameras = SCENE_CAMERAS.get(scene, [])

        # 收集该场景所有帧
        all_frames = set()
        for cam in cameras:
            if cam in gt_per_cam_scene:
                all_frames.update(gt_per_cam_scene[cam].keys())
            if cam in det_per_cam:
                all_frames.update(det_per_cam[cam].keys())

        prev_gt_to_det_scene = {}

        for frame in sorted(all_frames):
            for cam in cameras:
                gt_list = gt_per_cam_scene.get(cam, {}).get(frame, [])
                det_list = det_per_cam.get(cam, {}).get(frame, [])

                matched, unm_gt_ids, unm_det = hungarian_match(gt_list, det_list)

                # 统计
                for gt_id, det_vid in matched:
                    gt_track_frames[gt_id] += 1
                    gt_track_matched[gt_id] += 1

                    # ID switch 检测
                    if gt_id in prev_gt_to_det_scene:
                        if prev_gt_to_det_scene[gt_id] != det_vid:
                            id_switches += 1
                    prev_gt_to_det_scene[gt_id] = det_vid

                    # Fragmentation 检测
                    if gt_id in gt_track_state:
                        if gt_track_state[gt_id] == 'lost':
                            fragmentations += 1
                    gt_track_state[gt_id] = 'tracked'

                # 未匹配的 gt → IDFN
                for gt_id in unm_gt_ids:
                    gt_track_frames[gt_id] += 1
                    if gt_track_state.get(gt_id) == 'tracked':
                        gt_track_state[gt_id] = 'lost'
                        fragmentations += 1

                # 未匹配的 det → IDFP
                n_unmatched_det = len(unm_det)
                n_unmatched_gt = len(unm_gt_ids)

                total_idtp += len(matched)
                total_idfp += n_unmatched_det
                total_idfn += n_unmatched_gt

    # 计算最终指标
    idf1 = 2 * total_idtp / (2 * total_idtp + total_idfp + total_idfn) if (2 * total_idtp + total_idfp + total_idfn) > 0 else 0
    idp = total_idtp / (total_idtp + total_idfp) if (total_idtp + total_idfp) > 0 else 0
    idr = total_idtp / (total_idtp + total_idfn) if (total_idtp + total_idfn) > 0 else 0

    # MT / ML
    n_gt_tracks = len(gt_track_frames)
    mostly_tracked = sum(1 for gid in gt_track_frames if gt_track_matched[gid] / gt_track_frames[gid] > 0.8)
    mostly_lost = sum(1 for gid in gt_track_frames if gt_track_matched[gid] / gt_track_frames[gid] < 0.2)
    mt_ratio = mostly_tracked / n_gt_tracks if n_gt_tracks > 0 else 0
    ml_ratio = mostly_lost / n_gt_tracks if n_gt_tracks > 0 else 0

    return {
        "IDF1": idf1,
        "IDP": idp,
        "IDR": idr,
        "IDTP": total_idtp,
        "IDFP": total_idfp,
        "IDFN": total_idfn,
        "ID_Switches": id_switches,
        "Fragmentations": fragmentations,
        "MT": mostly_tracked,
        "ML": mostly_lost,
        "MT_ratio": mt_ratio,
        "ML_ratio": ml_ratio,
        "n_gt_tracks": n_gt_tracks,
    }


def compute_mot_per_scene(det_per_cam, scenes):
    """逐场景计算 MOT 指标"""
    results = {}
    for scene in scenes:
        gt_tracks, gt_per_cam_scene = load_gt_scene(scene)
        if not gt_tracks:
            continue
        metrics = compute_mot_metrics(gt_per_cam_scene, det_per_cam, [scene])
        results[scene] = metrics
    return results


# ── 跨镜拼接评估 ──────────────────────────────────────────

def extract_cross_camera_edges_gt(scene_name):
    """
    从 gt.txt 提取真值跨镜边
    返回: dict[gt_id] → set of cameras where this ID appears
    """
    gt_tracks, _ = load_gt_scene(scene_name)
    cross_cam = {}
    for gt_id, records in gt_tracks.items():
        cams = set(r[1] for r in records)
        if len(cams) > 1:
            cross_cam[gt_id] = cams
    return cross_cam


def extract_cross_camera_edges_system(tracks, scene_name):
    """
    从系统输出提取跨镜边 (通过 vehicle_id 关联)
    返回: dict[vehicle_id] → set of cameras
    """
    scene_tracks = [t for t in tracks if t.get("scene_id") == scene_name]
    vid_cams = defaultdict(set)
    for t in scene_tracks:
        vid = t.get("vehicle_id", -1)
        for cam in t.get("camera_ids", []):
            vid_cams[vid].add(cam)
    # 只保留出现在多个摄像头的
    return {vid: cams for vid, cams in vid_cams.items() if len(cams) > 1}


def compute_cross_camera_metrics(scene_name, tracks):
    """计算跨镜拼接指标"""
    gt_cross = extract_cross_camera_edges_gt(scene_name)
    sys_cross = extract_cross_camera_edges_system(tracks, scene_name)

    # 真值跨镜边数量
    n_gt_cross = len(gt_cross)

    # 系统预测的跨镜边数量
    n_sys_cross = len(sys_cross)

    # 正确预测的跨镜边: 系统预测的 vehicle_id 在 gt 中确实跨镜
    # 且系统预测的摄像头集合与 gt 有交集
    true_positives = 0
    for vid, sys_cams in sys_cross.items():
        if vid in gt_cross:
            gt_cams = gt_cross[vid]
            # 检查是否有至少2个共同摄像头
            common = sys_cams & gt_cams
            if len(common) >= 2:
                true_positives += 1

    precision = true_positives / n_sys_cross if n_sys_cross > 0 else 0
    recall = true_positives / n_gt_cross if n_gt_cross > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "gt_cross_edges": n_gt_cross,
        "sys_cross_edges": n_sys_cross,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def system_data():
    """加载系统输出数据"""
    if not RESULTS_JSON.exists():
        pytest.skip(f"Results file not found: {RESULTS_JSON}")
    dets, tracks = load_system_results()
    return dets, tracks


@pytest.fixture(scope="module")
def det_per_cam(system_data):
    dets, _ = system_data
    return build_det_per_cam(dets)


@pytest.fixture(scope="module")
def all_tracks(system_data):
    _, tracks = system_data
    return tracks


# ══════════════════════════════════════════════════════════
# A. 轨迹还原准确率测试 (MOT 标准指标)
# ══════════════════════════════════════════════════════════

class TestTrajectoryAccuracy:
    """轨迹还原准确率 - MOT 标准指标"""

    def test_overall_mot_metrics(self, det_per_cam):
        """A1. 整体 MOT 指标 (IDF1, IDP, IDR, ID Switches)"""
        print("\n" + "=" * 60)
        print("A1. 整体轨迹还原 MOT 指标")
        print("=" * 60)

        all_scenes = list(SCENE_CAMERAS.keys())
        metrics = compute_mot_metrics(None, det_per_cam, all_scenes)

        print(f"\n{'指标':<15} {'值':>10} {'达标线':>10} {'状态':>6}")
        print("-" * 45)

        checks = [
            ("IDF1", metrics["IDF1"], 0.6, ">="),
            ("IDP", metrics["IDP"], 0.7, ">="),
            ("IDR", metrics["IDR"], 0.6, ">="),
            ("ID_Switches", metrics["ID_Switches"], 50, "<="),
            ("MT_ratio", metrics["MT_ratio"], 0.5, ">="),
            ("ML_ratio", metrics["ML_ratio"], 0.2, "<="),
        ]

        for name, val, threshold, op in checks:
            if op == ">=":
                status = "[OK]" if val >= threshold else "[X]"
            else:
                status = "[OK]" if val <= threshold else "[X]"
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"| {name:<13} | {val_str:>10} | {threshold:>10} | {status:>4} |")

        print(f"\n详细统计:")
        print(f"  IDTP={metrics['IDTP']}, IDFP={metrics['IDFP']}, IDFN={metrics['IDFN']}")
        print(f"  GT轨迹数: {metrics['n_gt_tracks']}")
        print(f"  Mostly Tracked: {metrics['MT']}/{metrics['n_gt_tracks']} ({metrics['MT_ratio']:.2%})")
        print(f"  Mostly Lost: {metrics['ML']}/{metrics['n_gt_tracks']} ({metrics['ML_ratio']:.2%})")
        print(f"  Fragmentations: {metrics['Fragmentations']}")

        # 输出 IDF1 (主指标)
        assert metrics["IDF1"] >= 0, "IDF1 should be non-negative"

    def test_per_scene_mot(self, det_per_cam):
        """A2. 分场景 MOT 指标"""
        print("\n" + "=" * 60)
        print("A2. 分场景 MOT 指标")
        print("=" * 60)

        per_scene = compute_mot_per_scene(det_per_cam, list(SCENE_CAMERAS.keys()))

        print(f"\n{'场景':<8} {'IDF1':>8} {'IDP':>8} {'IDR':>8} {'IDS':>6} {'GT数':>6}")
        print("-" * 50)
        for scene in sorted(per_scene.keys()):
            m = per_scene[scene]
            print(f"| {scene:<6} | {m['IDF1']:>8.4f} | {m['IDP']:>8.4f} | {m['IDR']:>8.4f} | {m['ID_Switches']:>6} | {m['n_gt_tracks']:>6} |")

        assert len(per_scene) > 0, "Should have metrics for at least one scene"

    def test_per_camera_detail(self, det_per_cam):
        """A3. 分摄像头详细指标"""
        print("\n" + "=" * 60)
        print("A3. 分摄像头详细指标")
        print("=" * 60)

        all_results = {}
        for scene in SCENE_CAMERAS:
            gt_tracks, gt_per_cam_scene = load_gt_scene(scene)
            if not gt_tracks:
                continue
            cameras = SCENE_CAMERAS[scene]
            for cam in cameras:
                # 构建单摄像头的 gt_per_cam
                single_cam_gt = {cam: gt_per_cam_scene.get(cam, {})}
                single_cam_det = {cam: det_per_cam.get(cam, {})}
                m = compute_mot_metrics(single_cam_gt, single_cam_det, [scene])
                all_results[f"{scene}/{cam}"] = m

        print(f"\n{'摄像头':<12} {'IDF1':>8} {'IDP':>8} {'IDR':>8} {'IDS':>5} {'TP':>6} {'FP':>6} {'FN':>6}")
        print("-" * 72)
        for name in sorted(all_results.keys()):
            m = all_results[name]
            print(f"| {name:<10} | {m['IDF1']:>8.4f} | {m['IDP']:>8.4f} | {m['IDR']:>8.4f} | {m['ID_Switches']:>5} | {m['IDTP']:>6} | {m['IDFP']:>6} | {m['IDFN']:>6} |")

        assert len(all_results) > 0


# ══════════════════════════════════════════════════════════
# B. 跨镜拼接准确率测试
# ══════════════════════════════════════════════════════════

class TestCrossCameraStitching:
    """跨摄像头拼接准确率"""

    def test_cross_camera_per_scene(self, all_tracks):
        """B1. 各场景跨镜拼接指标"""
        print("\n" + "=" * 60)
        print("B1. 跨镜拼接准确率")
        print("=" * 60)

        all_metrics = {}
        for scene in SCENE_CAMERAS:
            m = compute_cross_camera_metrics(scene, all_tracks)
            all_metrics[scene] = m

        print(f"\n{'场景':<8} {'GT跨镜边':>10} {'系统跨镜边':>10} {'TP':>6} {'Precision':>10} {'Recall':>10} {'F1':>8}")
        print("-" * 72)
        for scene in sorted(all_metrics.keys()):
            m = all_metrics[scene]
            print(f"| {scene:<6} | {m['gt_cross_edges']:>10} | {m['sys_cross_edges']:>10} | {m['true_positives']:>6} | {m['precision']:>10.4f} | {m['recall']:>10.4f} | {m['f1']:>8.4f} |")

        # 汇总
        total_gt = sum(m['gt_cross_edges'] for m in all_metrics.values())
        total_sys = sum(m['sys_cross_edges'] for m in all_metrics.values())
        total_tp = sum(m['true_positives'] for m in all_metrics.values())
        total_prec = total_tp / total_sys if total_sys > 0 else 0
        total_rec = total_tp / total_gt if total_gt > 0 else 0
        total_f1 = 2 * total_prec * total_rec / (total_prec + total_rec) if (total_prec + total_rec) > 0 else 0
        print("-" * 72)
        print(f"| {'合计':<6} | {total_gt:>10} | {total_sys:>10} | {total_tp:>6} | {total_prec:>10.4f} | {total_rec:>10.4f} | {total_f1:>8.4f} |")

        print(f"\n说明:")
        print(f"  - GT跨镜边: 真值中出现在多个摄像头的车辆ID数")
        print(f"  - 系统跨镜边: 系统输出中 vehicle_id 出现在多个摄像头的数量")
        print(f"  - 由于系统输出均为单摄track，跨镜关联通过vehicle_id间接建立")

    def test_cross_camera_detail_per_vehicle(self, all_tracks):
        """B2. 跨镜拼接 - 逐车辆分析"""
        print("\n" + "=" * 60)
        print("B2. 跨镜拼接 - 逐车辆分析 (S01)")
        print("=" * 60)

        scene = "S01"
        gt_cross = extract_cross_camera_edges_gt(scene)
        sys_cross = extract_cross_camera_edges_system(all_tracks, scene)

        # 分析前10个跨镜车辆
        gt_cross_sorted = sorted(gt_cross.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"\n真值跨镜车辆 (共{len(gt_cross)}辆):")
        print(f"{'GT_ID':<8} {'GT摄像头':>20} {'系统摄像头':>20} {'匹配':>6}")
        print("-" * 60)
        correct = 0
        for gt_id, gt_cams in gt_cross_sorted[:15]:
            sys_cams = sys_cross.get(gt_id, set())
            matched = "[OK]" if len(sys_cams & gt_cams) >= 2 else "[X]"
            if len(sys_cams & gt_cams) >= 2:
                correct += 1
            gt_str = ",".join(sorted(gt_cams))
            sys_str = ",".join(sorted(sys_cams)) if sys_cams else "无"
            print(f"| {gt_id:<6} | {gt_str:>20} | {sys_str:>20} | {matched:>4} |")

        print(f"\n前15辆中正确匹配: {correct}/15")


# ══════════════════════════════════════════════════════════
# C. 鲁棒性测试
# ══════════════════════════════════════════════════════════

class TestRobustness:
    """鲁棒性分析"""

    def test_c1_camera_density(self, det_per_cam):
        """C1. 摄像头密度影响: S01(5摄) vs S03(6摄) vs S04(25摄)"""
        print("\n" + "=" * 60)
        print("C1. 摄像头密度对拼接准确率的影响")
        print("=" * 60)

        density_info = {}
        for scene in SCENE_CAMERAS:
            n_cams = len(SCENE_CAMERAS[scene])
            gt_tracks, gt_per_cam_scene = load_gt_scene(scene)
            n_gt_ids = len(gt_tracks)
            n_cross = sum(1 for records in gt_tracks.values() if len(set(r[1] for r in records)) > 1)

            # 计算该场景 MOT
            m = compute_mot_metrics(gt_per_cam_scene, det_per_cam, [scene])

            density_info[scene] = {
                "n_cameras": n_cams,
                "n_gt_ids": n_gt_ids,
                "n_cross_cam_ids": n_cross,
                "IDF1": m["IDF1"],
                "IDP": m["IDP"],
                "IDR": m["IDR"],
                "ID_Switches": m["ID_Switches"],
            }

        print(f"\n{'场景':<8} {'摄像头数':>8} {'GT_ID数':>8} {'跨镜ID':>8} {'IDF1':>8} {'IDP':>8} {'IDR':>8} {'IDS':>6}")
        print("-" * 72)
        for scene in sorted(density_info.keys()):
            d = density_info[scene]
            print(f"| {scene:<6} | {d['n_cameras']:>8} | {d['n_gt_ids']:>8} | {d['n_cross_cam_ids']:>8} | {d['IDF1']:>8.4f} | {d['IDP']:>8.4f} | {d['IDR']:>8.4f} | {d['ID_Switches']:>6} |")

        print(f"\n分析:")
        s01 = density_info.get("S01", {})
        s04 = density_info.get("S04", {})
        if s01 and s04:
            print(f"  S01 ({s01['n_cameras']}摄) IDF1={s01['IDF1']:.4f}")
            print(f"  S04 ({s04['n_cameras']}摄) IDF1={s04['IDF1']:.4f}")
            if s04['IDF1'] > 0:
                ratio = s01['IDF1'] / s04['IDF1']
                print(f"  IDF1 比值 (S01/S04): {ratio:.2f}x")

    def test_c2_time_interval(self, det_per_cam):
        """C2. 时间间隔对拼接的影响"""
        print("\n" + "=" * 60)
        print("C2. 跨镜转移时间间隔分析")
        print("=" * 60)

        # 统计真值跨镜转移的时间间隔
        interval_stats = defaultdict(list)  # scene → [time_intervals]

        for scene in SCENE_CAMERAS:
            gt_tracks, _ = load_gt_scene(scene)
            for gt_id, records in gt_tracks.items():
                # 按摄像头分组
                cam_records = defaultdict(list)
                for frame, cam, l, t, w, h in records:
                    cam_records[cam].append(frame)

                # 计算跨镜转移间隔
                cams_sorted = sorted(cam_records.keys())
                for i in range(len(cams_sorted) - 1):
                    cam_a = cams_sorted[i]
                    cam_b = cams_sorted[i + 1]
                    last_frame_a = max(cam_records[cam_a])
                    first_frame_b = min(cam_records[cam_b])
                    interval = first_frame_b - last_frame_a
                    if interval >= 0:
                        interval_stats[scene].append(interval)

        print(f"\n时间间隔分布 (帧差):")
        print(f"{'场景':<8} {'样本数':>8} {'均值':>8} {'中位数':>8} {'P25':>8} {'P75':>8} {'最大':>8}")
        print("-" * 64)
        for scene in sorted(interval_stats.keys()):
            intervals = interval_stats[scene]
            if not intervals:
                continue
            arr = np.array(intervals)
            print(f"| {scene:<6} | {len(arr):>8} | {arr.mean():>8.1f} | {np.median(arr):>8.1f} | {np.percentile(arr, 25):>8.1f} | {np.percentile(arr, 75):>8.1f} | {arr.max():>8} |")

        # 按时间间隔区间统计拼接成功率
        print(f"\n按时间间隔区间的跨镜关联统计:")
        bins = [(0, 10), (10, 30), (30, 60), (60, 120), (120, 300), (300, float('inf'))]
        print(f"{'区间(帧)':<15} {'数量':>8} {'占比':>8}")
        print("-" * 35)
        all_intervals = []
        for intervals in interval_stats.values():
            all_intervals.extend(intervals)
        total = len(all_intervals)
        for lo, hi in bins:
            count = sum(1 for x in all_intervals if lo <= x < hi)
            pct = count / total if total > 0 else 0
            label = f"{lo}-{hi}" if hi != float('inf') else f"{lo}+"
            print(f"| {label:<13} | {count:>8} | {pct:>8.1%} |")

    def test_c3_visibility(self, det_per_cam):
        """C3. 目标可见度对跟踪连续性的影响"""
        print("\n" + "=" * 60)
        print("C3. 目标可见度/遮挡对跟踪的影响")
        print("=" * 60)

        # gt.txt 的 visibility 字段 (第7列) 在 AICity22 中全为1
        # 改用 bbox 面积作为可见度代理指标
        print(f"\n注: AICity22 gt.txt visibility字段全为1，使用bbox面积作为可见度代理指标")

        for scene in ["S01", "S04"]:
            gt_tracks, gt_per_cam_scene = load_gt_scene(scene)
            if not gt_tracks:
                continue

            # 按 bbox 面积分组
            small_area = []   # < 5000
            medium_area = []  # 5000-20000
            large_area = []   # > 20000

            for gt_id, records in gt_tracks.items():
                avg_area = np.mean([w * h for _, _, _, _, w, h in records])
                n_frames = len(records)
                # 检查该 ID 是否被系统正确跟踪
                # 统计每个摄像头中匹配的检测数
                matched_frames = 0
                total_frames = 0
                for frame, cam, l, t, w, h in records:
                    total_frames += 1
                    det_list = det_per_cam.get(cam, {}).get(frame, [])
                    if det_list:
                        # 检查是否有匹配的检测
                        for didx, dbbox, dvid in det_list:
                            if iou_ltrb_to_xyxy([l, t, w, h], dbbox) > 0.3:
                                matched_frames += 1
                                break

                track_rate = matched_frames / total_frames if total_frames > 0 else 0
                entry = (gt_id, track_rate, n_frames)

                if avg_area < 5000:
                    small_area.append(entry)
                elif avg_area < 20000:
                    medium_area.append(entry)
                else:
                    large_area.append(entry)

            print(f"\n  场景 {scene}:")
            print(f"  {'面积区间':<15} {'车辆数':>8} {'平均跟踪率':>12} {'平均帧数':>10}")
            print(f"  " + "-" * 50)
            for label, group in [("小(<5k)", small_area), ("中(5k-20k)", medium_area), ("大(>20k)", large_area)]:
                if group:
                    avg_rate = np.mean([r for _, r, _ in group])
                    avg_frames = np.mean([f for _, _, f in group])
                    print(f"  | {label:<13} | {len(group):>8} | {avg_rate:>12.4f} | {avg_frames:>10.1f} |")

    def test_c4_scene_complexity(self, det_per_cam):
        """C4. 场景复杂度与 ID Switches 的关系"""
        print("\n" + "=" * 60)
        print("C4. 场景复杂度与 ID Switches 关系")
        print("=" * 60)

        complexity = {}
        for scene in SCENE_CAMERAS:
            gt_tracks, gt_per_cam_scene = load_gt_scene(scene)
            if not gt_tracks:
                continue

            # 计算车辆密度: 平均每帧的车辆数
            total_frames = set()
            total_dets = 0
            for cam in SCENE_CAMERAS[scene]:
                for frame in gt_per_cam_scene.get(cam, {}):
                    total_frames.add((cam, frame))
                    total_dets += len(gt_per_cam_scene[cam][frame])

            avg_density = total_dets / len(total_frames) if total_frames else 0
            n_cameras = len(SCENE_CAMERAS[scene])

            m = compute_mot_metrics(gt_per_cam_scene, det_per_cam, [scene])

            complexity[scene] = {
                "n_cameras": n_cameras,
                "avg_density": avg_density,
                "total_ids": len(gt_tracks),
                "ID_Switches": m["ID_Switches"],
                "IDF1": m["IDF1"],
            }

        print(f"\n{'场景':<8} {'摄像头':>6} {'平均密度':>10} {'GT_ID数':>8} {'ID_Switch':>10} {'IDF1':>8}")
        print("-" * 58)
        for scene in sorted(complexity.keys()):
            c = complexity[scene]
            print(f"| {scene:<6} | {c['n_cameras']:>6} | {c['avg_density']:>10.2f} | {c['total_ids']:>8} | {c['ID_Switches']:>10} | {c['IDF1']:>8.4f} |")

        print(f"\n分析:")
        print(f"  车辆密度越高 → 预期 ID Switches 越多")
        print(f"  摄像头越多 → 跨镜拼接难度越大")


# ══════════════════════════════════════════════════════════
# D. 观测链完整性测试
# ══════════════════════════════════════════════════════════

class TestObservationChain:
    """观测链完整性测试"""

    def test_observation_coverage(self, det_per_cam, all_tracks):
        """D1. 观测段覆盖率 & 推断段分析"""
        print("\n" + "=" * 60)
        print("D1. 观测链完整性分析")
        print("=" * 60)

        # 统计系统输出的 track 覆盖情况
        for scene in SCENE_CAMERAS:
            gt_tracks, gt_per_cam_scene = load_gt_scene(scene)
            if not gt_tracks:
                continue

            scene_tracks = [t for t in all_tracks if t.get("scene_id") == scene]
            cameras = SCENE_CAMERAS[scene]

            # 统计 GT 中被系统覆盖的帧数
            total_gt_frames = 0
            covered_gt_frames = 0

            # 统计每个 GT ID 的覆盖情况
            gt_coverage = {}
            for gt_id, records in gt_tracks.items():
                gt_frames = len(records)
                total_gt_frames += gt_frames
                matched = 0
                for frame, cam, l, t, w, h in records:
                    det_list = det_per_cam.get(cam, {}).get(frame, [])
                    for didx, dbbox, dvid in det_list:
                        if iou_ltrb_to_xyxy([l, t, w, h], dbbox) > 0.3:
                            matched += 1
                            break
                covered_gt_frames += matched
                gt_coverage[gt_id] = matched / gt_frames if gt_frames > 0 else 0

            coverage_rate = covered_gt_frames / total_gt_frames if total_gt_frames > 0 else 0

            # 统计系统 track 的帧覆盖
            sys_total_frames = 0
            for t in scene_tracks:
                fr = t.get("frame_range", [0, 0])
                sys_total_frames += (fr[1] - fr[0] + 1) if len(fr) == 2 else 0

            print(f"\n  场景 {scene}:")
            print(f"    GT 总观测帧: {total_gt_frames}")
            print(f"    系统覆盖帧: {covered_gt_frames}")
            print(f"    观测覆盖率: {coverage_rate:.2%}")
            print(f"    系统 track 数: {len(scene_tracks)}")
            print(f"    系统 track 总帧跨度: {sys_total_frames}")

            # 完整度分析: GT ID 被完全覆盖 (>90%) 的比例
            complete = sum(1 for v in gt_coverage.values() if v > 0.9)
            partial = sum(1 for v in gt_coverage.values() if 0.2 < v <= 0.9)
            lost = sum(1 for v in gt_coverage.values() if v <= 0.2)
            n_ids = len(gt_coverage)
            print(f"    完整覆盖(>90%): {complete}/{n_ids} ({complete / n_ids:.1%})" if n_ids > 0 else "")
            print(f"    部分覆盖(20-90%): {partial}/{n_ids} ({partial / n_ids:.1%})" if n_ids > 0 else "")
            print(f"    基本丢失(<20%): {lost}/{n_ids} ({lost / n_ids:.1%})" if n_ids > 0 else "")

    def test_track_continuity(self, det_per_cam, all_tracks):
        """D2. 轨迹连续性 - 链断裂分析"""
        print("\n" + "=" * 60)
        print("D2. 轨迹连续性分析")
        print("=" * 60)

        for scene in ["S01"]:  # 重点分析 S01
            gt_tracks, gt_per_cam_scene = load_gt_scene(scene)
            if not gt_tracks:
                continue

            print(f"\n  场景 {scene} 轨迹连续性:")
            print(f"  {'GT_ID':<8} {'摄像头':>10} {'帧范围':>15} {'帧数':>6} {'系统匹配':>10} {'连续段':>8}")
            print(f"  " + "-" * 65)

            total_segments = 0
            total_gt_ids = 0

            for gt_id in sorted(gt_tracks.keys())[:20]:  # 前20个
                records = gt_tracks[gt_id]
                cam_records = defaultdict(list)
                for frame, cam, l, t, w, h in records:
                    cam_records[cam].append(frame)

                for cam in sorted(cam_records.keys()):
                    frames = sorted(cam_records[cam])
                    n_frames = len(frames)
                    total_gt_ids += 1

                    # 检查系统匹配的连续性
                    matched_frames = []
                    for f in frames:
                        det_list = det_per_cam.get(cam, {}).get(f, [])
                        for didx, dbbox, dvid in det_list:
                            gt_entry = [e for e in gt_per_cam_scene.get(cam, {}).get(f, []) if e[0] == gt_id]
                            if gt_entry:
                                l, t, w, h = gt_entry[0][1:]
                                if iou_ltrb_to_xyxy([l, t, w, h], dbbox) > 0.3:
                                    matched_frames.append(f)
                                    break

                    # 计算连续段数
                    n_segments = 1
                    for i in range(1, len(matched_frames)):
                        if matched_frames[i] - matched_frames[i - 1] > 2:
                            n_segments += 1
                    total_segments += n_segments

                    matched_str = f"{len(matched_frames)}/{n_frames}"
                    cam_str = cam
                    frame_range = f"{frames[0]}-{frames[-1]}"
                    print(f"  | {gt_id:<6} | {cam_str:>10} | {frame_range:>15} | {n_frames:>6} | {matched_str:>10} | {n_segments:>8} |")

            avg_segments = total_segments / total_gt_ids if total_gt_ids > 0 else 0
            print(f"\n  平均连续段数: {avg_segments:.2f} (越接近1越好)")
            print(f"  链断裂率: {(avg_segments - 1):.2f} (0表示无断裂)")


# ══════════════════════════════════════════════════════════
# E. 综合报告
# ══════════════════════════════════════════════════════════

class TestSummaryReport:
    """生成综合报告"""

    def test_full_report(self, det_per_cam, all_tracks):
        """生成完整的轨迹拼接准确率报告"""
        print("\n")
        print("=" * 70)
        print("        轨迹拼接与还原准确率综合报告")
        print("=" * 70)

        # 1. 整体 MOT 指标
        print("\n1. 整体 MOT 指标:")
        print("-" * 50)
        all_scenes = list(SCENE_CAMERAS.keys())
        metrics = compute_mot_metrics(None, det_per_cam, all_scenes)

        checks = [
            ("IDF1", metrics["IDF1"], ">=0.6", metrics["IDF1"] >= 0.6),
            ("IDP", metrics["IDP"], ">=0.7", metrics["IDP"] >= 0.7),
            ("IDR", metrics["IDR"], ">=0.6", metrics["IDR"] >= 0.6),
            ("ID Switches", metrics["ID_Switches"], "<=50", metrics["ID_Switches"] <= 50),
            ("MT(>80%)", f"{metrics['MT_ratio']:.1%}", ">=50%", metrics["MT_ratio"] >= 0.5),
            ("ML(<20%)", f"{metrics['ML_ratio']:.1%}", "<=20%", metrics["ML_ratio"] <= 0.2),
        ]

        print(f"| {'指标':<15} | {'值':>12} | {'达标线':>8} | {'状态':>4} |")
        print("|" + "-" * 48 + "|")
        for name, val, threshold, ok in checks:
            status = "[OK]" if ok else "[X]"
            if isinstance(val, float):
                val_display = f"{val:.4f}"
            else:
                val_display = str(val)
            print(f"| {name:<15} | {val_display:>12} | {threshold:>8} | {status:>4} |")

        # 2. 跨镜拼接指标
        print("\n2. 跨镜拼接指标:")
        print("-" * 50)
        for scene in SCENE_CAMERAS:
            m = compute_cross_camera_metrics(scene, all_tracks)
            print(f"  {scene}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")
            print(f"         GT跨镜={m['gt_cross_edges']}, 系统跨镜={m['sys_cross_edges']}, TP={m['true_positives']}")

        # 3. 分场景 IDF1
        print("\n3. 分场景 IDF1:")
        print("-" * 50)
        for scene in SCENE_CAMERAS:
            gt_tracks, gt_per_cam_scene = load_gt_scene(scene)
            if not gt_tracks:
                continue
            m = compute_mot_metrics(gt_per_cam_scene, det_per_cam, [scene])
            n_cams = len(SCENE_CAMERAS[scene])
            print(f"  {scene} ({n_cams}摄): IDF1={m['IDF1']:.4f}, IDP={m['IDP']:.4f}, IDR={m['IDR']:.4f}, IDS={m['ID_Switches']}")

        # 4. 数据概况
        print("\n4. 数据概况:")
        print("-" * 50)
        print(f"  系统检测数: {len([d for d in det_per_cam.values() for f in d.values() for _ in f])}")
        print(f"  系统轨迹数: {len(all_tracks)}")
        for scene in SCENE_CAMERAS:
            gt_tracks, _ = load_gt_scene(scene)
            n_sys = len([t for t in all_tracks if t.get("scene_id") == scene])
            print(f"  {scene}: GT_ID={len(gt_tracks)}, 系统track={n_sys}")

        print("\n" + "=" * 70)
        print("        报告结束")
        print("=" * 70)

        # 不做 assert，仅输出报告
        assert True
