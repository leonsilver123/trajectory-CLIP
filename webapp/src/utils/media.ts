/**
 * 图片地址解析
 *
 * 后端返回的 `keyframe_path` / `crop_path` 是**相对 output/ 的路径**，且实测为
 * Windows 风格并带 `output\` 前缀，例如：
 *   "output\\aicity22_crops\\c005\\c005_f00677_idx028579.jpg"
 * 后端把 output/ 挂载在 /static（见 api/main.py），因此需要归一化成：
 *   "<API_BASE>/static/aicity22_crops/c005/c005_f00677_idx028579.jpg"
 */

import { API_BASE } from '../api/client';

/** 把后端返回的图片路径归一化成可访问的 URL；无法解析时返回 null（由 UI 显示占位） */
export function staticUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  // 统一分隔符，去掉开头的 ./ 与 output 前缀
  let rel = path.replace(/\\/g, '/').trim();
  rel = rel.replace(/^\.?\//, '');
  rel = rel.replace(/^output\//i, '');
  if (!rel) return null;
  // 已经是完整 URL 就直接用
  if (/^https?:\/\//i.test(rel)) return rel;
  return `${API_BASE}/static/${rel}`;
}

/** 取文件名，用于无图时展示可读标识 */
export function basename(path: string | null | undefined): string | null {
  if (!path) return null;
  const parts = path.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1] || null;
}
