/**
 * ScoreBar —— 空值安全的分数条
 *
 * antd 的 Progress 在 percent 为 NaN/undefined 时会渲染成 0%，
 * 那等于把「未知」画成「0 分」，违反不编造数据红线。
 * 所以这里先判断 null：为空就直接显示「未知」并禁用进度条。
 */

import { Progress, Tooltip, Typography } from 'antd';
import { isNum, toPercent } from '../utils/format';

const { Text } = Typography;

interface Props {
  label: string;
  value: number | null | undefined;
  /** 悬停解释该指标含义 */
  tip?: string;
  /** 为空时的文案：'--' 表示字段缺失，'未知' 表示值为 null */
  fallback?: string;
  color?: string;
  /** 数值展示的小数位 */
  digits?: number;
}

export default function ScoreBar({
  label,
  value,
  tip,
  fallback = '未知',
  color,
  digits = 3,
}: Props) {
  const percent = toPercent(value);
  const missing = percent === null;

  const body = (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          {label}
        </Text>
        <Text
          type={missing ? 'secondary' : undefined}
          style={{ fontSize: 12, fontVariantNumeric: 'tabular-nums' }}
        >
          {missing
            ? fallback
            : `${(value as number).toFixed(digits)}（${percent.toFixed(1)}%）`}
        </Text>
      </div>
      {missing ? (
        // 用一条虚线占位，明确表达「没有数据」而不是「0 分」
        <div
          style={{
            height: 6,
            marginTop: 4,
            borderTop: '1px dashed rgba(255,255,255,0.25)',
          }}
        />
      ) : (
        <Progress
          percent={percent}
          showInfo={false}
          size="small"
          strokeColor={color}
          trailColor="rgba(255,255,255,0.12)"
          style={{ marginBottom: 0 }}
        />
      )}
    </div>
  );

  if (tip) {
    const explain = isNum(value) ? `${tip}` : `${tip}（后端未提供有效数值）`;
    return <Tooltip title={explain}>{body}</Tooltip>;
  }
  return body;
}
