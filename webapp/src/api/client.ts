/**
 * axios 实例与静态资源地址解析
 *
 * VITE_API_BASE:
 *   - 留空（默认）：走同源相对路径。开发时由 vite.config.ts 的 proxy 转发到
 *     http://127.0.0.1:8000；生产时由 FastAPI 同时托管 dist/ 与 /api，也是同源。
 *   - 设为 http://host:8000：前端独立部署时直连后端（后端已开 CORS）。
 */

import axios, { AxiosError } from 'axios';

/** 后端根地址，去掉尾部斜杠 */
export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? '').replace(/\/+$/, '');

export const api = axios.create({
  baseURL: `${API_BASE}/api/v1`,
  timeout: 180_000, // 首次检索要加载 CLIP 模型，实测约 6s，留足余量
  headers: { 'Content-Type': 'application/json' },
});

/** 把后端返回的错误整理成可读中文信息（含 FastAPI 的 detail 字段） */
export function describeError(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const ax = err as AxiosError<{ detail?: unknown }>;
    const detail = ax.response?.data?.detail;
    if (typeof detail === 'string' && detail) return detail;
    if (Array.isArray(detail)) return '请求参数不合法';
    if (ax.code === 'ECONNABORTED') return '请求超时，后端可能仍在加载模型，请重试';
    if (!ax.response) return `无法连接后端（${API_BASE || '同源'}），请确认服务已启动`;
    return `请求失败：HTTP ${ax.response.status}`;
  }
  return err instanceof Error ? err.message : '未知错误';
}
