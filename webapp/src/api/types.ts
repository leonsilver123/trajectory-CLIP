/**
 * 后端 API 类型定义
 *
 * 字段全部来自对真实后端的实测响应（api/routes/*.py），不是猜测。
 * 硬性约定：后端大量数值字段可能为 null（见 PLAN2 红线 3「不编造数据」），
 * 因此这里把可能缺失/为空的字段一律标注为 `| null` 或可选，
 * 由 utils/format.ts 的格式化函数显式渲染成「--」/「未知」，绝不用假数字替代。
 */

/** 依据类型：强身份匹配 vs 概率推断 */
export type Basis = 'strong_identity' | 'probabilistic_inference' | (string & {});

/** 检索候选（POST /api/v1/search/query 的 candidates[] 元素）
 *
 * 注意：`/search/plate` 返回的是**精简结构**（无 track_id / camera_name /
 * clip_score / final_score 等），因此这里除 instance_id 外的字段全部可选。
 */
export interface Candidate {
  instance_id: string;              // 唯一标识（= target_id），回溯的入参
  track_id?: string | null;
  camera_id?: string | null;
  camera_name?: string | null;
  timestamp?: string | null;
  target_type?: string | null;
  attributes?: Record<string, string | number | null> | null;
  plate_number?: string | null;
  quality_score?: number | null;
  clip_score?: number | null;
  attribute_score?: number | null;
  text_score?: number | null;
  attribute_match_score?: number | null;
  combined_score?: number | null;
  final_score?: number | null;      // 排序主分（/plate 不返回）
  rank?: number | null;
  keyframe_path?: string | null;    // 相对 output/ 的路径，可能带 output\ 前缀与反斜杠
  detection_bbox?: number[] | null;
  has_trajectory?: boolean | null;
  trajectory_frame_range?: unknown;
  trajectory_camera_ids?: string[] | null;
  trajectory_detection_count?: number | null;
  track_frame_count?: number | null;
  track_cameras?: string[] | null;
  track_frames?: TrackFrame[] | null;
}

/** 候选所属轨迹内的单帧检测 */
export interface TrackFrame {
  frame_id?: number | null;
  camera_id?: string | null;
  camera_name?: string | null;
  crop_path?: string | null;
  timestamp?: string | null;
  bbox?: number[] | null;
  confidence?: number | null;
}

/** 检索响应 */
export interface SearchResponse {
  query_id: string;
  candidates: Candidate[];
  total_count: number;
}

/** 确认目标响应 */
export interface ConfirmResponse {
  query_id: string;
  instance_id: string;
  status: string;
  message: string;
}

/** 观测节点（真实检测到的镜头，非推断） */
export interface ObservationNode {
  camera_id: string;
  camera_name?: string | null;
  tracklet_id?: string | null;
  timestamp?: string | null;        // 该摄像头本地时间
  global_timestamp?: string | null; // 跨镜全局对齐后的时间（T5），实测存在
  time_offset_seconds?: number | null; // 本地时间相对全局的偏移，实测存在
  latitude?: number | null;
  longitude?: number | null;
  keyframe_path?: string | null;    // 实测恒为 null：回溯节点不带关键帧，候选才带
  confidence?: number | null;       // 可能为 null
  basis?: Basis | null;
  basis_text?: string | null;
}

/** 观测段（目标在某摄像头视野内的真实区间） */
export interface ObservationSegment {
  tracklet_id?: string | null;
  camera_id: string;
  camera_name?: string | null;
  start_time?: string | null;       // "HH:MM:SS"（注意：不含日期）
  end_time?: string | null;         // "HH:MM:SS"
  direction?: string | null;
  // 实测为空串 ""（非 null）：表示后端未做出入口描述，UI 按缺失渲染
  entry_description?: string | null;
  exit_description?: string | null;
  basis?: Basis | null;
  basis_text?: string | null;
}

/** 推断段（摄像头之间，未观测到，概率推断） */
export interface InferenceSegment {
  source_camera_id: string;
  target_camera_id: string;
  source_camera_name?: string | null;
  target_camera_name?: string | null;
  source_latitude?: number | null;
  source_longitude?: number | null;
  target_latitude?: number | null;
  target_longitude?: number | null;
  confidence?: number | null;
  estimated_travel_time?: number | null;   // 由路网距离推算，实测有值（如 5.5）
  actual_travel_time?: number | null;      // 实测为 null
  // 以下为实测存在但前端暂未展示的字段：
  source_tracklet_id?: string | null;
  target_tracklet_id?: string | null;
  local_travel_time?: number | null;       // 未做全局对齐的本地时间差，实测可能为负
  time_aligned?: boolean | null;           // 两侧观测是否已完成全局时间对齐
  source_time_offset_seconds?: number | null;
  target_time_offset_seconds?: number | null;
  route_description?: string | null;
  basis?: Basis | null;
  basis_text?: string | null;
  score_detail?: Record<string, number | null> | null;
}

/** 候选路径 */
export interface CandidatePath {
  path_id?: string | null;
  road_segments?: string[] | null;
  confidence?: number | null;
  distance_meters?: number | null;
  estimated_time?: number | null;
  description?: string | null;
  basis?: Basis | null;
  basis_text?: string | null;
}

/** 身份依据 */
export interface IdentityInfo {
  mode?: string | null;
  basis?: Basis | null;
  basis_text?: string | null;
  vehicle_id?: string | null;
  target_id?: string | null;
}

/** 证据面板（各维度分数，实测可能为 0.0 / null） */
export interface Evidence {
  plate_consistency?: number | null;
  appearance_similarity?: number | null;
  attribute_consistency?: number | null;
  temporal_feasibility?: number | null;
  spatial_feasibility?: number | null;
  direction_consistency?: number | null;
  source?: string | null;
  link_count?: number | null;
  target_id?: string | null;
  vehicle_id?: string | null;
  detection_count?: number | null;
  camera_count?: number | null;
  identity_basis?: Basis | null;
  identity_certainty?: number | null;
  linkage_confidence?: number | null;
}

/** 回溯目标信息 */
export interface TargetInstance {
  instance_id?: string | null;
  target_id?: string | null;
  camera_id?: string | null;
  timestamp?: string | null;
  target_type?: string | null;
  attributes?: Record<string, string | number | null> | null;
  plate_number?: string | null;
  quality_score?: number | null;
  keyframe_path?: string | null;
}

/** 回溯响应（POST /api/v1/backtrack/trace） */
export interface BacktrackResponse {
  query_id: string;
  camera_sequence: string[];
  observation_nodes: ObservationNode[];
  observation_segments: ObservationSegment[];
  inference_segments: InferenceSegment[];
  candidate_paths: CandidatePath[];
  overall_confidence: number | null;   // 可能为 null
  identity?: IdentityInfo | null;
  evidence?: Evidence | null;
  target_instance?: TargetInstance | null;
}

/** 回溯请求 */
export interface BacktrackRequest {
  instance_id?: string;
  query_id?: string;
  max_upstream?: number;
  max_downstream?: number;
  /** auto 自动选路 / strong 强制强身份 / stitch 强制概率拼接 */
  mode?: 'auto' | 'strong' | 'stitch';
}

/** 单摄像头在完整轨迹中的段（POST /backtrack/trajectory） */
export interface TrajectoryCameraSegment {
  camera_id: string;
  camera_name?: string | null;
  arrival_time?: string | null;
  departure_time?: string | null;
  duration_seconds?: number | null;
  detection_count?: number | null;
  direction?: string | null;
  frames?: TrackFrame[] | null;
}

/** 完整跨镜轨迹 */
export interface Trajectory {
  track_id?: string | null;
  vehicle_id?: string | null;
  first_appearance?: string | null;
  last_appearance?: string | null;
  total_duration_seconds?: number | null;
  total_cameras?: number | null;
  total_detections?: number | null;
  camera_sequence?: TrajectoryCameraSegment[] | null;
  attributes?: Record<string, string | number | null> | null;
}

/** POST /backtrack/trajectory 响应 */
export interface TrajectoryResponse {
  success: boolean;
  trajectory: Trajectory | null;
}

/** GET /dashboard/stats */
export interface DashboardStats {
  camera_count?: number | null;
  camera_online?: number | null;
  instance_count?: number | null;
  tracklet_count?: number | null;
  edge_count?: number | null;
  today_search_count?: number | null;
  today_backtrack_count?: number | null;
  search_by_type?: Record<string, number> | null;
  search_frequency?: unknown[] | null;
  avg_observation_chain_length?: number | null;
  confidence_distribution?: { range: string; count: number }[] | null;
}

/** 摄像头元数据（GET /dashboard/cameras） */
export interface CameraMeta {
  camera_id: string;
  name?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  /** 场景近似中心坐标（真实 GPS，场景级精度；逐摄像头 GPS 数据集中不存在） */
  scene_gps_center?: [number, number] | null;
  scene?: string | null;
  status?: string | null;
  direction?: number | null;
  today_detections?: number | null;
}

/** GET /dashboard/health */
export interface HealthStatus {
  status?: string | null;
  version?: string | null;
  device?: string | null;
  results_loaded?: boolean | null;
  data_source?: string | null;
}
