/**
 * TrajectoryPage —— 完整跨镜轨迹（页面 4）
 *
 * 输入 track_id 或 instance_id → POST /backtrack/trajectory
 * 展示 vehicle_id、首末次出现、总时长、经过摄像头数、检测数，
 * 以及每个摄像头的到达/离开时间与帧缩略图。
 *
 * 与 /backtrack 的区别：本页是按 vehicle_id 聚合的**完整轨迹视图**，
 * 不做强/弱身份区分；/backtrack 页是带依据标注的观测链。
 */

import { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Alert,
  Button,
  Card,
  Col,
  Collapse,
  Descriptions,
  Empty,
  Image,
  Input,
  Row,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography,
  App as AntApp,
} from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { Trajectory, TrajectoryCameraSegment } from '../api/types';
import { fetchTrajectory } from '../api/endpoints';
import { describeError } from '../api/client';
import { Nullable, Value } from '../components/Value';
import { attrEntries, fmtDuration, isStr } from '../utils/format';
import { staticUrl } from '../utils/media';
import { palette } from '../theme';

const { Title, Text, Paragraph } = Typography;

export default function TrajectoryPage() {
  const [params, setParams] = useSearchParams();
  const { message } = AntApp.useApp();

  const [trackId, setTrackId] = useState(params.get('track_id') ?? '');
  const [instanceId, setInstanceId] = useState(params.get('instance_id') ?? '');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [traj, setTraj] = useState<Trajectory | null>(null);

  async function run() {
    if (!trackId.trim() && !instanceId.trim()) {
      message.warning('请填写 track_id 或 instance_id 至少一项');
      return;
    }
    setLoading(true);
    setError(null);
    setTraj(null);
    // 同步到 URL，便于刷新与分享
    const next = new URLSearchParams();
    if (trackId.trim()) next.set('track_id', trackId.trim());
    if (instanceId.trim()) next.set('instance_id', instanceId.trim());
    setParams(next, { replace: true });

    try {
      const res = await fetchTrajectory({
        track_id: trackId.trim() || undefined,
        instance_id: instanceId.trim() || undefined,
      });
      if (!res.success || !res.trajectory) {
        setError('后端返回 success=false 或 trajectory 为空');
        return;
      }
      setTraj(res.trajectory);
    } catch (e) {
      setError(describeError(e));
    } finally {
      setLoading(false);
    }
  }

  const segColumns: ColumnsType<TrajectoryCameraSegment> = [
    {
      title: '次序',
      key: 'idx',
      width: 70,
      render: (_: unknown, __: TrajectoryCameraSegment, i: number) => (
        <Tag color="geekblue">{i + 1}</Tag>
      ),
    },
    {
      title: '摄像头',
      dataIndex: 'camera_name',
      render: (_: unknown, r) => (
        <Space size={4}>
          <Text>{r.camera_name ?? r.camera_id}</Text>
          <Tag style={{ marginInlineEnd: 0, fontSize: 11 }}>{r.camera_id}</Tag>
        </Space>
      ),
    },
    {
      title: '到达时间',
      dataIndex: 'arrival_time',
      width: 190,
      render: (v: string | null | undefined) => <Nullable value={v} fallback="未知" />,
    },
    {
      title: '离开时间',
      dataIndex: 'departure_time',
      width: 190,
      render: (v: string | null | undefined) => <Nullable value={v} fallback="未知" />,
    },
    {
      title: '停留时长',
      dataIndex: 'duration_seconds',
      width: 120,
      sorter: (a, b) => (a.duration_seconds ?? -1) - (b.duration_seconds ?? -1),
      render: (v: number | null | undefined) =>
        isNumOrNull(v) ? <Text>{fmtDuration(v)}</Text> : <Text type="secondary">未知</Text>,
    },
    {
      title: '检测数',
      dataIndex: 'detection_count',
      width: 100,
      sorter: (a, b) => (a.detection_count ?? -1) - (b.detection_count ?? -1),
      render: (v: number | null | undefined) => <Value value={v} digits={0} fallback="未知" />,
    },
    {
      title: '方向',
      dataIndex: 'direction',
      width: 110,
      render: (v: string | null | undefined) => <Nullable value={v} fallback="未知" />,
    },
  ];

  const seq = traj?.camera_sequence ?? [];
  const allFrames = seq.flatMap((s) => s.frames ?? []);

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Card>
        <Title level={4} style={{ marginTop: 0, marginBottom: 4 }}>
          完整轨迹
        </Title>
        <Paragraph type="secondary" style={{ marginBottom: 12, fontSize: 13 }}>
          查看目标车辆在所有摄像头中的完整出现记录。通常由「查找目标」或「跨镜回溯」跳转进入。
        </Paragraph>

        <Row gutter={[12, 12]}>
          <Col xs={24} md={10}>
            <Input
              placeholder="轨迹编号（从查找或回溯页带入）"
              value={trackId}
              onChange={(e) => setTrackId(e.target.value)}
              onPressEnter={() => void run()}
              allowClear
            />
          </Col>
          <Col xs={24} md={10}>
            <Input
              placeholder="检测编号（选填）"
              value={instanceId}
              onChange={(e) => setInstanceId(e.target.value)}
              onPressEnter={() => void run()}
              allowClear
            />
          </Col>
          <Col xs={24} md={4}>
            <Button
              type="primary"
              block
              icon={<SearchOutlined />}
              loading={loading}
              onClick={() => void run()}
            >
              查询轨迹
            </Button>
          </Col>
        </Row>
      </Card>

      {error && (
        <Alert
          type="error"
          showIcon
          message="轨迹查询失败"
          description={
            <Space direction="vertical">
              <Text>{error}</Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                常见原因：track_id / instance_id 在后端数据中不存在（404）。
              </Text>
            </Space>
          }
        />
      )}

      {loading && (
        <Card>
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Spin size="large" />
            <div style={{ marginTop: 12 }}>
              <Text type="secondary">正在查询轨迹…</Text>
            </div>
          </div>
        </Card>
      )}

      {!loading && traj && (
        <>
          <Row gutter={[16, 16]}>
            <Col xs={12} md={4}>
              <Card size="small">
                <Statistic
                  title="车辆编号"
                  valueRender={() => (
                    <Text style={{ fontSize: 18, fontFamily: 'monospace' }}>
                      {isStr(traj.vehicle_id) ? traj.vehicle_id : '--'}
                    </Text>
                  )}
                />
              </Card>
            </Col>
            <Col xs={12} md={4}>
              <Card size="small">
                <Statistic
                  title="经过摄像头"
                  valueRender={() => (
                    <Value value={traj.total_cameras} digits={0} style={{ fontSize: 22 }} />
                  )}
                />
              </Card>
            </Col>
            <Col xs={12} md={4}>
              <Card size="small">
                <Statistic
                  title="检测总数"
                  valueRender={() => (
                    <Value value={traj.total_detections} digits={0} style={{ fontSize: 22 }} />
                  )}
                />
              </Card>
            </Col>
            <Col xs={12} md={6}>
              <Card size="small">
                <Statistic
                  title="总时长"
                  valueRender={() => (
                    <Text style={{ fontSize: 18 }}>
                      {isNumOrNull(traj.total_duration_seconds)
                        ? fmtDuration(traj.total_duration_seconds)
                        : '未知'}
                    </Text>
                  )}
                />
              </Card>
            </Col>
            <Col xs={24} md={6}>
              <Card size="small">
                <Text type="secondary" style={{ fontSize: 12 }}>
                  首次出现
                </Text>
                <div>
                  <Nullable value={traj.first_appearance} fallback="未知" />
                </div>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  末次出现
                </Text>
                <div>
                  <Nullable value={traj.last_appearance} fallback="未知" />
                </div>
              </Card>
            </Col>
          </Row>

          <Card size="small" title="轨迹概要">
            <Descriptions size="small" column={{ xs: 1, sm: 2, lg: 3 }}>
              <Descriptions.Item label="轨迹编号">
                <Text style={{ fontFamily: 'monospace', fontSize: 12 }}>
                  {isStr(traj.track_id) ? traj.track_id : '--'}
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="车辆编号">
                <Text style={{ fontFamily: 'monospace', fontSize: 12 }}>
                  {isStr(traj.vehicle_id) ? traj.vehicle_id : '--'}
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="摄像头数">
                <Value value={traj.total_cameras} digits={0} />
              </Descriptions.Item>
            </Descriptions>
            {attrEntries(traj.attributes).length > 0 && (
              <div style={{ marginTop: 6 }}>
                <Text type="secondary" style={{ fontSize: 12, marginRight: 8 }}>
                  属性：
                </Text>
                {attrEntries(traj.attributes).map(([k, v]) => (
                  <Tag key={k}>
                    {k}: {v}
                  </Tag>
                ))}
              </div>
            )}
          </Card>

          <Card title="各摄像头到达 / 离开" size="small">
            {seq.length === 0 ? (
              <Empty description="后端返回空 camera_sequence" />
            ) : (
              <Table<TrajectoryCameraSegment>
                size="small"
                rowKey={(r, i) => `${r.camera_id}-${i ?? 0}`}
                columns={segColumns}
                dataSource={seq}
                pagination={false}
                scroll={{ x: 900 }}
              />
            )}
          </Card>

          <Card title={`帧缩略图（共 ${allFrames.length} 帧）`} size="small">
            {seq.length === 0 ? (
              <Empty description="无帧数据" />
            ) : (
              <Collapse
                defaultActiveKey={seq[0] ? [seq[0].camera_id] : []}
                items={seq.map((s) => {
                  const frames = (s.frames ?? []).filter((f) => staticUrl(f.crop_path));
                  return {
                    key: s.camera_id,
                    label: (
                      <Space>
                        <Text strong>{s.camera_name ?? s.camera_id}</Text>
                        <Tag>{s.frames?.length ?? 0} 帧</Tag>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {s.arrival_time ?? '未知'} ~ {s.departure_time ?? '未知'}
                        </Text>
                      </Space>
                    ),
                    children:
                      frames.length === 0 ? (
                        <Text type="secondary">
                          该摄像头下没有可用的帧图（crop_path 为空或缺失）
                        </Text>
                      ) : (
                        <Image.PreviewGroup>
                          <Space wrap size={8}>
                            {frames.slice(0, 40).map((f) => (
                              <div key={`${f.frame_id}-${f.crop_path}`} style={{ width: 96 }}>
                                <Image
                                  src={staticUrl(f.crop_path) ?? ''}
                                  width={96}
                                  height={72}
                                  style={{ objectFit: 'cover', borderRadius: 4 }}
                                />
                                <div
                                  style={{
                                    fontSize: 11,
                                    color: palette.textDim,
                                    textAlign: 'center',
                                  }}
                                >
                                  #{f.frame_id ?? '--'}
                                </div>
                              </div>
                            ))}
                          </Space>
                          {(s.frames?.length ?? 0) > 40 && (
                            <div style={{ marginTop: 8, fontSize: 12, color: palette.textDim }}>
                              仅展示前 40 帧（该摄像头共 {s.frames?.length} 帧）
                            </div>
                          )}
                        </Image.PreviewGroup>
                      ),
                  };
                })}
              />
            )}
          </Card>
        </>
      )}

      {!loading && !traj && !error && (
        <Card>
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="填写 track_id 或 instance_id 后查询"
          />
        </Card>
      )}
    </Space>
  );
}

/** 局部判空：避免在渲染里直接对 null 调 toFixed */
function isNumOrNull(v: number | null | undefined): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}
