/**
 * 后端接口封装
 *
 * 全部为对真实端点的薄封装，不做任何数据补全/兜底伪造。
 * 端点来源：api/routes/{search,confirm,backtrack,dashboard}.py
 */

import { api } from './client';
import type {
  BacktrackRequest,
  BacktrackResponse,
  CameraMeta,
  ConfirmResponse,
  DashboardStats,
  HealthStatus,
  SearchResponse,
  TrajectoryResponse,
} from './types';

/** POST /api/v1/search/query —— 文本检索（属性粗筛 + CLIP 精排） */
export async function searchByText(
  queryText: string,
  topK = 20,
  targetType?: string,
): Promise<SearchResponse> {
  const { data } = await api.post<SearchResponse>('/search/query', {
    query_text: queryText,
    top_k: topK,
    target_type: targetType ?? null,
  });
  return data;
}

/** POST /api/v1/search/plate —— 车牌精确查询 */
export async function searchByPlate(plateNumber: string): Promise<SearchResponse> {
  const { data } = await api.post<SearchResponse>('/search/plate', {
    plate_number: plateNumber,
  });
  return data;
}

/** POST /api/v1/confirm/target —— 用户确认目标，后端写回会话状态 */
export async function confirmTarget(
  queryId: string,
  instanceId: string,
): Promise<ConfirmResponse> {
  const { data } = await api.post<ConfirmResponse>('/confirm/target', {
    query_id: queryId,
    instance_id: instanceId,
  });
  return data;
}

/** POST /api/v1/backtrack/trace —— 以确认目标为锚点构建跨镜观测链 */
export async function backtrackTrace(req: BacktrackRequest): Promise<BacktrackResponse> {
  const { data } = await api.post<BacktrackResponse>('/backtrack/trace', {
    instance_id: req.instance_id ?? null,
    query_id: req.query_id ?? null,
    max_upstream: req.max_upstream ?? 10,
    max_downstream: req.max_downstream ?? 10,
    mode: req.mode ?? 'auto',
  });
  return data;
}

/** POST /api/v1/backtrack/trajectory —— 完整跨镜轨迹（按 vehicle_id 聚合） */
export async function fetchTrajectory(params: {
  track_id?: string;
  instance_id?: string;
}): Promise<TrajectoryResponse> {
  const { data } = await api.post<TrajectoryResponse>('/backtrack/trajectory', {
    track_id: params.track_id ?? null,
    instance_id: params.instance_id ?? null,
  });
  return data;
}

/** GET /api/v1/dashboard/stats —— 统计信息 */
export async function fetchStats(): Promise<DashboardStats> {
  const { data } = await api.get<DashboardStats>('/dashboard/stats');
  return data;
}

/** GET /api/v1/dashboard/cameras —— 摄像头列表 */
export async function fetchCameras(): Promise<CameraMeta[]> {
  const { data } = await api.get<{ cameras: CameraMeta[] }>('/dashboard/cameras');
  return data.cameras ?? [];
}

/** GET /api/v1/dashboard/health —— 服务健康状态 */
export async function fetchHealth(): Promise<HealthStatus> {
  const { data } = await api.get<HealthStatus>('/dashboard/health');
  return data;
}
