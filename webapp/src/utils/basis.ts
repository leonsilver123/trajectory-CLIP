/**
 * 依据（basis）可视化映射
 *
 * 后端用 `basis` 区分观测/推断的来源，这是本项目「离散观测链」的核心语义：
 *   - strong_identity         强身份匹配（有真实 vehicle_id / 车牌等硬标识）
 *   - probabilistic_inference 概率推断（无硬标识，靠外观+时空评分拼接）
 * 前端必须把二者在颜色与线型上明确区分开，不能让推断看起来像观测。
 */

import type { Basis } from '../api/types';
import { DASH, isStr } from './format';

export interface BasisStyle {
  key: string;
  label: string;       // 中文短标签
  color: string;       // 主色
  description: string; // 悬停解释
}

const STRONG: BasisStyle = {
  key: 'strong_identity',
  label: '强身份',
  color: '#52c41a',
  description: '强身份匹配：依据真实硬标识（vehicle_id / 车牌）直接关联，非推断',
};

const INFERENCE: BasisStyle = {
  key: 'probabilistic_inference',
  label: '概率推断',
  color: '#faad14',
  description: '概率推断：无硬标识，基于外观与时空可行性评分拼接，存在不确定性',
};

const UNKNOWN_BASIS: BasisStyle = {
  key: 'unknown',
  label: '依据未知',
  color: '#8c8c8c',
  description: '后端未提供 basis 字段，无法判断该结论是观测还是推断',
};

/** 把后端的 basis 字符串映射为可视化样式；缺失时返回「依据未知」而不是猜一个 */
export function basisStyle(basis: Basis | null | undefined): BasisStyle {
  if (!isStr(basis)) return UNKNOWN_BASIS;
  if (basis === 'strong_identity') return STRONG;
  if (basis === 'probabilistic_inference') return INFERENCE;
  return { key: basis, label: basis, color: '#8c8c8c', description: `未知依据类型：${basis}` };
}

/** 是否为推断类依据（用于决定虚线/实线） */
export function isInference(basis: Basis | null | undefined): boolean {
  return isStr(basis) && basis === 'probabilistic_inference';
}

/** 依据文本；缺失时给出 DASH 而不是编造解释 */
export function basisText(t: string | null | undefined): string {
  return isStr(t) ? t : DASH;
}
