/**
 * Value / ScoreText —— 空值安全的数值展示组件
 *
 * 这是「不编造数据」红线在 UI 层的统一出口：页面不应该直接渲染裸数字，
 * 而应该使用本组件，由它决定「有真实值」还是「未知」。
 *
 * 两种缺失语义刻意区分：
 *   - `--`   数据本身没有这个字段（例如 /plate 接口不返回 clip_score）
 *   - `未知` 字段存在但值为 null（例如 actual_travel_time 因时间戳未对齐而为 null）
 */

import { Tooltip, Typography } from 'antd';
import type { ReactNode } from 'react';
import { DASH, fmtNum, isNum } from '../utils/format';

const { Text } = Typography;

interface ValueProps {
  /** 真实数值，null/undefined/NaN 都会被判定为缺失 */
  value: number | null | undefined;
  digits?: number;
  suffix?: ReactNode;
  /** 缺失时的文案：DASH（字段缺失）或 UNKNOWN（值为 null 的推断字段） */
  fallback?: string;
  /** 悬停解释缺失原因，避免用户误以为系统出错 */
  hint?: string;
  strong?: boolean;
  style?: React.CSSProperties;
}

/** 渲染一个可能为空的数值；缺失时显示占位符，绝不补零 */
export function Value({
  value,
  digits = 2,
  suffix,
  fallback = DASH,
  hint,
  strong,
  style,
}: ValueProps) {
  const missing = !isNum(value);
  const text = missing ? fallback : `${fmtNum(value, digits)}${suffix ?? ''}`;

  const node = (
    <Text
      type={missing ? 'secondary' : undefined}
      strong={strong && !missing}
      style={{ fontVariantNumeric: 'tabular-nums', ...style }}
    >
      {text}
    </Text>
  );

  if (missing && hint) {
    return <Tooltip title={hint}>{node}</Tooltip>;
  }
  return node;
}

interface NullableProps {
  /** 任意可能为空的值，为空时渲染 fallback */
  value: string | number | null | undefined;
  fallback?: string;
  hint?: string;
  strong?: boolean;
  style?: React.CSSProperties;
}

/** 渲染可能为空的字符串/数字；空串也视为缺失 */
export function Nullable({ value, fallback = DASH, hint, strong, style }: NullableProps) {
  const missing = value === null || value === undefined || value === '';
  const text = missing ? fallback : String(value);
  const node = (
    <Text
      type={missing ? 'secondary' : undefined}
      strong={strong && !missing}
      style={style}
    >
      {text}
    </Text>
  );
  if (missing && hint) return <Tooltip title={hint}>{node}</Tooltip>;
  return node;
}

/** 键值对行：左标签右值，值走空值安全渲染 */
export function Field({
  label,
  value,
  fallback = DASH,
  hint,
  digits = 2,
}: {
  label: string;
  value: string | number | null | undefined;
  fallback?: string;
  hint?: string;
  digits?: number;
}) {
  const isNumber = typeof value === 'number';
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        gap: 12,
        padding: '3px 0',
      }}
    >
      <Text type="secondary" style={{ fontSize: 12, whiteSpace: 'nowrap' }}>
        {label}
      </Text>
      {isNumber ? (
        <Value value={value} digits={digits} fallback={fallback} hint={hint} />
      ) : (
        <Nullable value={value} fallback={fallback} hint={hint} />
      )}
    </div>
  );
}
