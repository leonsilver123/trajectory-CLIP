/**
 * CandidateCard —— 单个检索候选卡片
 *
 * 面向非技术用户：只保留一个易懂的「匹配度」，把 final_score / clip_score /
 * track_id 等技术字段收起来，不堆在卡片上。
 * 诚实性：分数与字段为 null 时显示「--」，不补 0。
 */

import { Card, Image, Space, Tag, Tooltip, Typography } from 'antd';
import { CheckCircleFilled } from '@ant-design/icons';
import type { Candidate } from '../api/types';
import { Nullable } from './Value';
import { attrEntries, fmtTargetType, toPercent } from '../utils/format';
import { staticUrl } from '../utils/media';
import { palette } from '../theme';

const { Text } = Typography;

interface Props {
  candidate: Candidate;
  selected: boolean;
  onSelect: (c: Candidate) => void;
}

export default function CandidateCard({ candidate, selected, onSelect }: Props) {
  const img = staticUrl(candidate.keyframe_path);
  const attrs = attrEntries(candidate.attributes);
  const percent = toPercent(candidate.final_score);

  return (
    <Card
      hoverable
      onClick={() => onSelect(candidate)}
      styles={{ body: { padding: 10 } }}
      style={{
        cursor: 'pointer',
        borderColor: selected ? palette.accent : undefined,
        boxShadow: selected ? `0 0 0 2px ${palette.accent}55` : undefined,
        position: 'relative',
      }}
      cover={
        <div
          style={{
            height: 150,
            background: '#000',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden',
          }}
        >
          {img ? (
            <Image
              src={img}
              alt={candidate.instance_id}
              height={150}
              style={{ objectFit: 'contain', width: '100%' }}
              fallback="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNTAiIGhlaWdodD0iMTUwIj48cmVjdCB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgZmlsbD0iIzFhMWEyNSIvPjx0ZXh0IHg9Ijc1IiB5PSI3OCIgZmlsbD0iIzg4OCIgZm9udC1zaXplPSIxMyIgdGV4dC1hbmNob3I9Im1pZGRsZSI+5Zu-54mH5LiK6L29PC90ZXh0Pjwvc3ZnPg=="
            />
          ) : (
            <Text type="secondary" style={{ fontSize: 12 }}>
              无候选图
            </Text>
          )}
        </div>
      }
    >
      {selected && (
        <CheckCircleFilled
          style={{
            position: 'absolute',
            top: 8,
            right: 8,
            fontSize: 20,
            color: palette.accent,
            background: '#000',
            borderRadius: '50%',
          }}
        />
      )}

      {/* 排名 + 类型 + 是否有轨迹 */}
      <Space size={6} wrap style={{ marginBottom: 6 }}>
        <Tag color="geekblue" style={{ marginInlineEnd: 0 }}>
          #{candidate.rank ?? '--'}
        </Tag>
        <Tag style={{ marginInlineEnd: 0 }}>{fmtTargetType(candidate.target_type)}</Tag>
        {candidate.has_trajectory ? (
          <Tag color="cyan" style={{ marginInlineEnd: 0 }}>
            有轨迹
          </Tag>
        ) : (
          <Tag style={{ marginInlineEnd: 0 }}>无轨迹</Tag>
        )}
      </Space>

      {/* 匹配度：单一易懂指标，技术分收进 tooltip */}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
        <Text type="secondary">匹配度</Text>
        <Tooltip title="由外观相似度与属性匹配综合计算，值越高越可能是目标">
          {percent === null ? (
            <Text type="secondary">--</Text>
          ) : (
            <Text strong style={{ fontVariantNumeric: 'tabular-nums' }}>
              {percent}%
            </Text>
          )}
        </Tooltip>
      </div>

      {/* 摄像头 + 属性（业务信息，技术 ID 收进 title 悬停） */}
      <div
        style={{
          marginTop: 6,
          paddingTop: 6,
          borderTop: `1px solid ${palette.border}`,
          fontSize: 12,
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
          <Text type="secondary">摄像头</Text>
          <Text ellipsis style={{ maxWidth: 140 }} title={candidate.camera_id ?? undefined}>
            <Nullable value={candidate.camera_name ?? candidate.camera_id} />
          </Text>
        </div>
        {attrs.length > 0 && (
          <div style={{ marginTop: 4 }}>
            <Space size={4} wrap>
              {attrs.map(([k, v]) => (
                <Tag key={k} style={{ marginInlineEnd: 0, fontSize: 11 }}>
                  {k}: {v}
                </Tag>
              ))}
            </Space>
          </div>
        )}
      </div>
    </Card>
  );
}
