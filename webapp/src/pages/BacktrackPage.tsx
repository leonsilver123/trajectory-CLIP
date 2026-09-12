/**
 * BacktrackPage —— 跨镜回溯详情（页面 2）
 *
 * 入参：URL 查询参数 instance_id（必填）/ query_id（可选，用于回写会话）
 * 数据：POST /backtrack/trace
 *
 * 四个视图：
 *   1. 摄像头序列 —— antd Steps 展示 camera_sequence，点击联动下方
 *   2. 时间轴     —— ECharts 甘特图（观测段实心 / 推断段虚线）
 *   3. 地图       —— Leaflet 点位 + 实线路径 + 虚线推断段
 *   4. 证据面板   —— 各维度评分 + 总体置信度环
 *
 * 所有可能为 null 的字段一律显式显示「未知」，不编造数值。
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Radio,
  Row,
  Space,
  Spin,
  Steps,
  Table,
  Tag,
  Typography,
} from 'antd';
import { ReloadOutlined, AimOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type {
  BacktrackResponse,
  CandidatePath,
  InferenceSegment,
  ObservationSegment,
} from '../api/types';
import { backtrackTrace } from '../api/endpoints';
import { describeError } from '../api/client';
import BasisTag from '../components/BasisTag';
import CameraMap from '../components/CameraMap';
import ConfidenceRing from '../components/ConfidenceRing';
import EvidencePanel from '../components/EvidencePanel';
import TimelineGantt from '../components/TimelineGantt';
import { Nullable, Value } from '../components/Value';
import { attrEntries, fmtClock, UNKNOWN, isStr } from '../utils/format';
import { staticUrl } from '../utils/media';
import { palette } from '../theme';

const { Title, Text, Paragraph } = Typography;

type Mode = 'auto' | 'strong' | 'stitch';

export default function BacktrackPage() {
  const [params] = useSearchParams();

  const instanceId = params.get('instance_id') ?? '';
  const queryId = params.get('query_id') ?? undefined;

  const [mode, setMode] = useState<Mode>('auto');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<BacktrackResponse | null>(null);
  const [activeStep, setActiveStep] = useState(0);

  const load = useCallback(async () => {
    if (!instanceId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await backtrackTrace({
        instance_id: instanceId,
        query_id: queryId,
        mode,
      });
      setData(res);
      setActiveStep(0);
    } catch (e) {
      setError(describeError(e));
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [instanceId, queryId, mode]);

  useEffect(() => {
    void load();
  }, [load]);

  /** 推断段表格列：null 一律走「未知」 */
  const inferenceColumns = useMemo<ColumnsType<InferenceSegment>>(
    () => [
      {
        title: '来源摄像头',
        dataIndex: 'source_camera_name',
        render: (_: unknown, r) => (
          <Space size={4}>
            <Text>{r.source_camera_name ?? r.source_camera_id}</Text>
            <Tag style={{ marginInlineEnd: 0, fontSize: 11 }}>{r.source_camera_id}</Tag>
          </Space>
        ),
      },
      {
        title: '目标摄像头',
        dataIndex: 'target_camera_name',
        render: (_: unknown, r) => (
          <Space size={4}>
            <Text>{r.target_camera_name ?? r.target_camera_id}</Text>
            <Tag style={{ marginInlineEnd: 0, fontSize: 11 }}>{r.target_camera_id}</Tag>
          </Space>
        ),
      },
      {
        title: '置信度',
        dataIndex: 'confidence',
        width: 110,
        sorter: (a, b) => (a.confidence ?? -1) - (b.confidence ?? -1),
        render: (v: number | null | undefined) => (
          <Value value={v} digits={3} fallback={UNKNOWN} strong />
        ),
      },
      {
        title: '估计行程时间',
        dataIndex: 'estimated_travel_time',
        width: 130,
        render: (v: number | null | undefined) => (
          <Value
            value={v}
            digits={1}
            fallback={UNKNOWN}
            hint="后端 estimated_travel_time 为 null（时间戳未全局对齐，见 T5）"
          />
        ),
      },
      {
        title: '实际行程时间',
        dataIndex: 'actual_travel_time',
        width: 130,
        render: (v: number | null | undefined) => (
          <Value
            value={v}
            digits={1}
            fallback={UNKNOWN}
            hint="后端 actual_travel_time 为 null，说明该段没有被真实观测时间约束"
          />
        ),
      },
      {
        title: '依据',
        dataIndex: 'basis',
        width: 110,
        render: (_: unknown, r) => <BasisTag basis={r.basis} size="small" />,
      },
      {
        title: '说明',
        dataIndex: 'route_description',
        render: (v: string | null | undefined) => (
          <Text style={{ fontSize: 12 }}>
            <Nullable value={v} />
          </Text>
        ),
      },
    ],
    [],
  );

  /** 观测段表格列 */
  const obsColumns = useMemo<ColumnsType<ObservationSegment>>(
    () => [
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
        title: '进入',
        dataIndex: 'start_time',
        width: 110,
        render: (v: string | null | undefined) => (
          <Text className="tabular">{fmtClock(v)}</Text>
        ),
      },
      {
        title: '离开',
        dataIndex: 'end_time',
        width: 110,
        render: (v: string | null | undefined) => (
          <Text className="tabular">{fmtClock(v)}</Text>
        ),
      },
      {
        title: '方向',
        dataIndex: 'direction',
        width: 110,
        render: (v: string | null | undefined) => <Nullable value={v} fallback={UNKNOWN} />,
      },
      {
        title: '依据',
        dataIndex: 'basis',
        width: 110,
        render: (_: unknown, r) => <BasisTag basis={r.basis} size="small" />,
      },
    ],
    [],
  );

  /** 候选路径表格列 */
  const pathColumns = useMemo<ColumnsType<CandidatePath>>(
    () => [
      { title: '路径 ID', dataIndex: 'path_id', width: 130 },
      {
        title: '路径描述',
        dataIndex: 'description',
        render: (v: string | null | undefined) => (
          <Text style={{ fontSize: 12 }}>
            <Nullable value={v} />
          </Text>
        ),
      },
      {
        title: '置信度',
        dataIndex: 'confidence',
        width: 110,
        render: (v: number | null | undefined) => (
          <Value value={v} digits={4} fallback={UNKNOWN} />
        ),
      },
      {
        title: '距离',
        dataIndex: 'distance_meters',
        width: 110,
        render: (v: number | null | undefined) => (
          <Value value={v} digits={1} fallback={UNKNOWN} />
        ),
      },
      {
        title: '估计时间',
        dataIndex: 'estimated_time',
        width: 110,
        render: (v: number | null | undefined) => (
          <Value value={v} digits={1} fallback={UNKNOWN} />
        ),
      },
      {
        title: '依据',
        dataIndex: 'basis',
        width: 110,
        render: (_: unknown, r) => <BasisTag basis={r.basis} size="small" />,
      },
    ],
    [],
  );

  // 未传 instance_id：给出明确引导，而不是空白页
  if (!instanceId) {
    return (
      <Card>
        <Empty
          description={
            <Space direction="vertical" size={4}>
              <Text>未指定要回溯的目标（URL 缺少 instance_id）</Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                请先在「目标检索」页选择并确认一个候选目标
              </Text>
            </Space>
          }
        >
          <Link to="/">
            <Button type="primary" icon={<AimOutlined />}>
              前往目标检索
            </Button>
          </Link>
        </Empty>
      </Card>
    );
  }

  const nodes = data?.observation_nodes ?? [];
  const obsSegments = data?.observation_segments ?? [];
  const inferences = data?.inference_segments ?? [];
  const cameraSequence = data?.camera_sequence ?? [];

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      {/* 头部：目标信息 + 模式切换 */}
      <Card>
        <Row justify="space-between" align="middle" gutter={[12, 12]}>
          <Col>
            <Title level={4} style={{ margin: 0 }}>
              跨镜回溯
            </Title>
            <Text type="secondary" style={{ fontSize: 12 }}>
              目标在多摄像头之间的轨迹
            </Text>
          </Col>
          <Col>
            <Space wrap>
              <Text type="secondary" style={{ fontSize: 12 }}>
                匹配方式：
              </Text>
              <Radio.Group
                size="small"
                value={mode}
                onChange={(e) => setMode(e.target.value as Mode)}
                optionType="button"
                buttonStyle="solid"
                options={[
                  { label: '自动', value: 'auto' },
                  { label: '精确匹配', value: 'strong' },
                  { label: '概率推断', value: 'stitch' },
                ]}
              />
              <Button size="small" icon={<ReloadOutlined />} onClick={() => void load()}>
                重新回溯
              </Button>
            </Space>
          </Col>
        </Row>

        <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}>
          自动：优先用硬标识（车牌/车辆编号）精确匹配；概率推断：无硬标识时按外观与时空可行性推断，置信度较低。
        </Paragraph>
      </Card>

      {error && (
        <Alert
          type="error"
          showIcon
          message="回溯失败"
          description={
            <Space direction="vertical">
              <Text>{error}</Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                常见原因：instance_id 在后端数据中不存在（404），或数据文件不可用（500）。
              </Text>
            </Space>
          }
          action={
            <Button size="small" onClick={() => void load()}>
              重试
            </Button>
          }
        />
      )}

      {loading && (
        <Card>
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Spin size="large" />
            <div style={{ marginTop: 12 }}>
              <Text type="secondary">正在构建观测链…</Text>
            </div>
          </div>
        </Card>
      )}

      {!loading && data && (
        <>
          {/* 总览：置信度环 + 关键统计 */}
          <Row gutter={[16, 16]}>
            <Col xs={24} md={8} lg={6}>
              <Card style={{ textAlign: 'center' }}>
                <ConfidenceRing
                  value={data.overall_confidence}
                  hint="链路整体置信度：由各观测段与推断段的拼接评分聚合得到"
                />
                <div style={{ marginTop: 8 }}>
                  <BasisTag basis={data.identity?.basis} />
                </div>
              </Card>
            </Col>
            <Col xs={24} md={16} lg={18}>
              <Card size="small" title="回溯摘要">
                <Row gutter={[16, 16]}>
                  <Col xs={12} sm={8} lg={5}>
                    <div>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        经过摄像头
                      </Text>
                      <div style={{ fontSize: 22, fontWeight: 600 }}>
                        {cameraSequence.length}
                      </div>
                    </div>
                  </Col>
                  <Col xs={12} sm={8} lg={5}>
                    <div>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        观测节点
                      </Text>
                      <div style={{ fontSize: 22, fontWeight: 600 }}>{nodes.length}</div>
                    </div>
                  </Col>
                  <Col xs={12} sm={8} lg={5}>
                    <div>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        观测段
                      </Text>
                      <div style={{ fontSize: 22, fontWeight: 600 }}>
                        {obsSegments.length}
                      </div>
                    </div>
                  </Col>
                  <Col xs={12} sm={8} lg={5}>
                    <div>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        推断段
                      </Text>
                      <div style={{ fontSize: 22, fontWeight: 600 }}>
                        {inferences.length}
                      </div>
                    </div>
                  </Col>
                  <Col xs={12} sm={8} lg={4}>
                    <div>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        候选路径
                      </Text>
                      <div style={{ fontSize: 22, fontWeight: 600 }}>
                        {data.candidate_paths?.length ?? 0}
                      </div>
                    </div>
                  </Col>
                </Row>

                <Descriptions
                  size="small"
                  column={{ xs: 1, sm: 2, lg: 3 }}
                  style={{ marginTop: 12 }}
                  labelStyle={{ fontSize: 12 }}
                  contentStyle={{ fontSize: 12 }}
                >
                  <Descriptions.Item label="匹配方式">
                    <Nullable
                      value={
                        data.identity?.mode === 'strong'
                          ? '精确匹配'
                          : data.identity?.mode === 'weak'
                            ? '概率推断'
                            : undefined
                      }
                      fallback={UNKNOWN}
                    />
                  </Descriptions.Item>
                  <Descriptions.Item label="车辆编号">
                    <Text style={{ fontFamily: 'monospace' }}>
                      {isStr(data.identity?.vehicle_id) ? data.identity?.vehicle_id : '--'}
                    </Text>
                  </Descriptions.Item>
                  <Descriptions.Item label="目标类型">
                    <Nullable value={data.target_instance?.target_type} />
                  </Descriptions.Item>
                  <Descriptions.Item label="目标摄像头">
                    <Nullable value={data.target_instance?.camera_id} />
                  </Descriptions.Item>
                  <Descriptions.Item label="目标时间">
                    <Nullable value={data.target_instance?.timestamp} />
                  </Descriptions.Item>
                  <Descriptions.Item label="车牌">
                    <Nullable
                      value={data.target_instance?.plate_number}
                      fallback={UNKNOWN}
                      hint="后端 plate_number 为 null：该目标没有识别到车牌"
                    />
                  </Descriptions.Item>
                  <Descriptions.Item label="依据说明">
                    <Text style={{ fontSize: 12 }}>
                      <Nullable value={data.identity?.basis_text} fallback={UNKNOWN} />
                    </Text>
                  </Descriptions.Item>
                </Descriptions>

                {attrEntries(data.target_instance?.attributes).length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <Text type="secondary" style={{ fontSize: 12, marginRight: 8 }}>
                      目标属性：
                    </Text>
                    {attrEntries(data.target_instance?.attributes).map(([k, v]) => (
                      <Tag key={k}>{k}: {v}</Tag>
                    ))}
                  </div>
                )}
              </Card>
            </Col>
          </Row>

          {/* 摄像头序列 Steps */}
          <Card title="摄像头序列（观测链）">
            {cameraSequence.length === 0 ? (
              <Empty description="后端返回空 camera_sequence" />
            ) : (
              <>
                <Steps
                  size="small"
                  current={activeStep}
                  onChange={setActiveStep}
                  direction="horizontal"
                  style={{ overflowX: 'auto', paddingBottom: 8 }}
                  items={cameraSequence.map((cam) => {
                    const node = nodes.find((n) => n.camera_id === cam);
                    const seg = obsSegments.find((s) => s.camera_id === cam);
                    return {
                      title: node?.camera_name ?? cam,
                      description: (
                        <Space direction="vertical" size={0}>
                          <Text type="secondary" style={{ fontSize: 11 }}>
                            {cam}
                          </Text>
                          <Text type="secondary" style={{ fontSize: 11 }}>
                            {fmtClock(seg?.start_time)} ~ {fmtClock(seg?.end_time)}
                          </Text>
                          <BasisTag basis={node?.basis} size="small" />
                        </Space>
                      ),
                      icon: node ? undefined : undefined,
                    };
                  })}
                />

                {/* 当前节点详情 */}
                {nodes[activeStep] && (
                  <Card size="small" style={{ marginTop: 12, background: 'rgba(255,255,255,0.03)' }}>
                    <Row gutter={[16, 12]} align="middle">
                      <Col flex="160px">
                        {staticUrl(nodes[activeStep].keyframe_path) ? (
                          <img
                            src={staticUrl(nodes[activeStep].keyframe_path) ?? ''}
                            alt="keyframe"
                            style={{ width: '100%', borderRadius: 6 }}
                          />
                        ) : (
                          <div
                            style={{
                              height: 90,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              border: `1px dashed ${palette.border}`,
                              borderRadius: 6,
                              color: palette.textDim,
                              fontSize: 12,
                              textAlign: 'center',
                            }}
                          >
                            该节点无关键帧
                            <br />
                            （后端返回 null）
                          </div>
                        )}
                      </Col>
                      <Col flex="auto">
                        <Descriptions size="small" column={{ xs: 1, sm: 2, lg: 3 }}>
                          <Descriptions.Item label="摄像头">
                            {nodes[activeStep].camera_name ?? nodes[activeStep].camera_id}
                          </Descriptions.Item>
                          <Descriptions.Item label="时间">
                            <Nullable value={nodes[activeStep].timestamp} fallback={UNKNOWN} />
                          </Descriptions.Item>
                          <Descriptions.Item label="置信度">
                            <Value
                              value={nodes[activeStep].confidence}
                              digits={3}
                              fallback={UNKNOWN}
                            />
                          </Descriptions.Item>
                          <Descriptions.Item label="依据">
                            <BasisTag basis={nodes[activeStep].basis} />
                          </Descriptions.Item>
                        </Descriptions>
                        {isStr(nodes[activeStep].basis_text) && (
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            {nodes[activeStep].basis_text}
                          </Text>
                        )}
                      </Col>
                    </Row>
                  </Card>
                )}
              </>
            )}
          </Card>

          {/* 时间轴甘特 */}
          <Card
            title="时间轴（观测段实心 · 推断段虚线）"
            extra={
              <Space size={8}>
                <Tag color={palette.strong}>强身份</Tag>
                <Tag color={palette.inference}>概率推断</Tag>
              </Space>
            }
          >
            <TimelineGantt
              nodes={nodes}
              segments={obsSegments}
              inferences={inferences}
              cameraSequence={cameraSequence}
            />
          </Card>

          {/* 地图 */}
          <Card title="路径地图（观测实线 · 推断虚线）">
            <CameraMap
              nodes={nodes}
              segments={obsSegments}
              inferences={inferences}
              cameraSequence={cameraSequence}
            />
          </Card>

          {/* 证据面板 */}
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={10}>
              <EvidencePanel evidence={data.evidence} />
            </Col>
            <Col xs={24} lg={14}>
              <Space direction="vertical" size={12} style={{ width: '100%' }}>
                <Card title="推断段明细" size="small">
                  {inferences.length === 0 ? (
                    <Empty
                      image={Empty.PRESENTED_IMAGE_SIMPLE}
                      description="没有推断段（全部为观测到的镜头）"
                    />
                  ) : (
                    <Table<InferenceSegment>
                      size="small"
                      rowKey={(r) => `${r.source_camera_id}->${r.target_camera_id}`}
                      columns={inferenceColumns}
                      dataSource={inferences}
                      pagination={false}
                      scroll={{ x: 900 }}
                    />
                  )}
                </Card>

                <Card title="观测段明细" size="small">
                  {obsSegments.length === 0 ? (
                    <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有观测段" />
                  ) : (
                    <Table<ObservationSegment>
                      size="small"
                      rowKey={(r) => r.tracklet_id ?? r.camera_id}
                      columns={obsColumns}
                      dataSource={obsSegments}
                      pagination={false}
                      scroll={{ x: 800 }}
                    />
                  )}
                </Card>

                <Card title="候选路径" size="small">
                  {(data.candidate_paths ?? []).length === 0 ? (
                    <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有候选路径" />
                  ) : (
                    <Table<CandidatePath>
                      size="small"
                      rowKey={(r, i) => r.path_id ?? `path-${i ?? 0}`}
                      columns={pathColumns}
                      dataSource={data.candidate_paths ?? []}
                      pagination={false}
                      scroll={{ x: 900 }}
                    />
                  )}
                </Card>
              </Space>
            </Col>
          </Row>

          <Text type="secondary" style={{ fontSize: 12, color: palette.textDim }}>
            说明：观测段是目标真实出现在该摄像头视野内的区间；推断段是摄像头之间未被
            观测到的部分，由算法按外观与时空可行性评分推断，两者以颜色与线型区分。
            任何显示为「未知」的字段都是后端返回 null，前端不做补值。
          </Text>
        </>
      )}
    </Space>
  );
}
