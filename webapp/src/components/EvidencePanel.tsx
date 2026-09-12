/**
 * EvidencePanel —— 回溯证据面板
 *
 * 展示 evidence 的各维度分数（外观/属性/车牌/时间/空间/方向）+ 统计信息。
 * 实测这些分数可能是 0.0 或 null：0.0 是真实评分结果（要正常显示），
 * null 则显示「未知」，二者语义不同，不能混为一谈。
 * 证据整体缺失（evidence 为 null）时明确提示，而不是显示一排 0。
 */

import { Alert, Card, Descriptions, Divider, Space, Typography } from 'antd';
import type { Evidence } from '../api/types';
import { Nullable, Value } from './Value';
import ScoreBar from './ScoreBar';
import BasisTag from './BasisTag';
import { DASH, UNKNOWN, isNum } from '../utils/format';
import { palette } from '../theme';

const { Text } = Typography;

const DIMENSIONS: { key: keyof Evidence; label: string; tip: string }[] = [
  { key: 'appearance_similarity', label: '外观相似度', tip: '跨镜目标外观特征（含 ReID）的相似程度' },
  { key: 'attribute_consistency', label: '属性一致度', tip: '颜色/车型等结构化属性在不同摄像头间的一致程度' },
  { key: 'plate_consistency', label: '车牌一致度', tip: '车牌识别结果的一致性；无车牌时为中性值' },
  { key: 'temporal_feasibility', label: '时间可行性', tip: '两摄像头间转移所需时间是否物理上可行' },
  { key: 'spatial_feasibility', label: '空间可行性', tip: '两摄像头间的路网距离是否支持该转移' },
  { key: 'direction_consistency', label: '方向一致度', tip: '目标在各摄像头中的行驶方向是否连贯' },
];

interface Props {
  evidence: Evidence | null | undefined;
}

export default function EvidencePanel({ evidence }: Props) {
  if (!evidence) {
    return (
      <Alert
        type="warning"
        showIcon
        message="后端未返回证据数据"
        description="evidence 字段为 null，无法展示各维度评分（不以 0 分代替）。"
      />
    );
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Card size="small" title="证据维度评分" styles={{ body: { paddingTop: 12 } }}>
        {DIMENSIONS.map((d) => (
          <ScoreBar
            key={String(d.key)}
            label={d.label}
            value={evidence[d.key] as number | null | undefined}
            tip={d.tip}
          />
        ))}
        <Divider style={{ margin: '10px 0' }} />
        <Text type="secondary" style={{ fontSize: 12 }}>
          进度条为空值（未知）时以虚线占位；0% 表示真实评分为 0，两者含义不同。
        </Text>
      </Card>

      <Card size="small" title="证据来源与统计">
        <Descriptions
          size="small"
          column={1}
          labelStyle={{ width: 120, fontSize: 12 }}
          contentStyle={{ fontSize: 12 }}
        >
          <Descriptions.Item label="数据来源">
            <Nullable value={evidence.source} />
          </Descriptions.Item>
          <Descriptions.Item label="身份依据">
            {evidence.identity_basis ? (
              <BasisTag basis={evidence.identity_basis} />
            ) : (
              <Text type="secondary">依据未知</Text>
            )}
          </Descriptions.Item>
          <Descriptions.Item label="身份确定性">
            <Value
              value={evidence.identity_certainty}
              digits={3}
              fallback={UNKNOWN}
              hint="强身份匹配时为 1.0；概率推断时取决于拼接评分"
            />
          </Descriptions.Item>
          <Descriptions.Item label="链路置信度">
            <Value value={evidence.linkage_confidence} digits={4} fallback={UNKNOWN} />
          </Descriptions.Item>
          <Descriptions.Item label="拼接链路数">
            <Value value={evidence.link_count} digits={0} />
          </Descriptions.Item>
          <Descriptions.Item label="车辆 ID">
            <Text style={{ fontFamily: 'monospace' }}>
              {evidence.vehicle_id ? evidence.vehicle_id : DASH}
            </Text>
          </Descriptions.Item>
          <Descriptions.Item label="检测总数">
            <Value value={evidence.detection_count} digits={0} />
          </Descriptions.Item>
          <Descriptions.Item label="涉及摄像头">
            <Value value={evidence.camera_count} digits={0} />
          </Descriptions.Item>
        </Descriptions>

        {!isNum(evidence.temporal_feasibility) && (
          <Alert
            style={{ marginTop: 10 }}
            type="info"
            showIcon
            message="时间可行性缺失"
            description="后端时间戳尚未做全局对齐（见 PLAN2 的 T5），该维度暂不可用。"
          />
        )}
      </Card>

      <Text type="secondary" style={{ fontSize: 12, color: palette.textDim }}>
        提示：各维度评分是拼接算法对「这是同一个目标」的支持程度，不是目标本身的属性。
      </Text>
    </Space>
  );
}
