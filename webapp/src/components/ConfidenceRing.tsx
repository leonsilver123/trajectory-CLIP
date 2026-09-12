/**
 * ConfidenceRing —— 总置信度进度环
 *
 * overall_confidence 实测可能为 null（例如时间戳未对齐、评分退化时）。
 * 此处**不显示 0%**，而是画出灰色虚线圈 + 「未知」文字，避免把未知读成低分。
 */

import { Progress, Tooltip, Typography } from 'antd';
import { UNKNOWN, toPercent } from '../utils/format';
import { palette } from '../theme';

const { Text } = Typography;

interface Props {
  value: number | null | undefined;
  label?: string;
  size?: number;
  hint?: string;
}

/** 依据分值给环上色：低置信度用暖色警示，避免「看起来都很可信」 */
function ringColor(v: number): string {
  if (v >= 0.75) return palette.strong;
  if (v >= 0.5) return '#1677ff';
  if (v >= 0.3) return palette.inference;
  return palette.danger;
}

export default function ConfidenceRing({
  value,
  label = '总体置信度',
  size = 150,
  hint,
}: Props) {
  const percent = toPercent(value);
  const missing = percent === null;

  const ring = (
    <div style={{ textAlign: 'center' }}>
      {missing ? (
        <div
          style={{
            width: size,
            height: size,
            margin: '0 auto',
            borderRadius: '50%',
            border: `2px dashed ${palette.border}`,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Text type="secondary" style={{ fontSize: 20 }}>
            {UNKNOWN}
          </Text>
          <Text type="secondary" style={{ fontSize: 11 }}>
            后端未给出数值
          </Text>
        </div>
      ) : (
        <Progress
          type="dashboard"
          percent={percent}
          size={size}
          strokeColor={ringColor(value as number)}
          trailColor="rgba(255,255,255,0.12)"
          format={(p) => (
            <span>
              <div
                style={{
                  fontSize: 26,
                  fontWeight: 600,
                  fontVariantNumeric: 'tabular-nums',
                  lineHeight: 1.2,
                }}
              >
                {(value as number).toFixed(4)}
              </div>
              <div style={{ fontSize: 12, opacity: 0.65 }}>{(p ?? 0).toFixed(1)}%</div>
            </span>
          )}
        />
      )}
      <div style={{ marginTop: 4 }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          {label}
        </Text>
      </div>
    </div>
  );

  if (hint) return <Tooltip title={hint}>{ring}</Tooltip>;
  return ring;
}

/** 供别处复用的置信度取色 */
export { ringColor };
