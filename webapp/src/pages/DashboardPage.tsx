/**
 * DashboardPage —— 统计大屏（页面 3）
 *
 * 数据：GET /dashboard/stats、/dashboard/cameras、/dashboard/health
 * 图表：ECharts（置信度分布柱状图、目标类型饼图、摄像头检测量 TOP N 柱状图）
 *
 * 诚实性约定：
 *   - 后端 `search_frequency` / `avg_observation_chain_length` / `edge_count` /
 *     `today_*` 目前实测为空数组或 0，**不画成曲线来暗示有时间序列数据**，
 *     而是明确标注「后端当前未提供」。
 *   - KPI 卡缺失值显示「--」。
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Col,
  Row,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { EChartsOption } from 'echarts';
import type { CameraMeta, DashboardStats, HealthStatus } from '../api/types';
import { fetchCameras, fetchHealth, fetchStats } from '../api/endpoints';
import { describeError } from '../api/client';
import EChart from '../components/EChart';
import { Value } from '../components/Value';
import { palette } from '../theme';

const { Title, Text } = Typography;

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [cameras, setCameras] = useState<CameraMeta[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // 三个接口互不依赖，并发拉取；任一失败不影响其他数据展示
      const [s, c, h] = await Promise.allSettled([
        fetchStats(),
        fetchCameras(),
        fetchHealth(),
      ]);
      if (s.status === 'fulfilled') setStats(s.value);
      if (c.status === 'fulfilled') setCameras(c.value);
      if (h.status === 'fulfilled') setHealth(h.value);

      const failed = [s, c, h].filter((r) => r.status === 'rejected');
      if (failed.length === 3) {
        setError(describeError((s as PromiseRejectedResult).reason));
      } else if (failed.length > 0) {
        setError(`${failed.length} 个统计接口调用失败，已展示可用部分`);
      }
    } catch (e) {
      setError(describeError(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  /** 置信度分布柱状图 */
  const confOption = useMemo<EChartsOption>(() => {
    const dist = stats?.confidence_distribution ?? [];
    return {
      backgroundColor: 'transparent',
      grid: { left: 50, right: 20, top: 24, bottom: 36 },
      tooltip: { trigger: 'axis' },
      xAxis: {
        type: 'category',
        data: dist.map((d) => d.range),
        axisLabel: { color: 'rgba(255,255,255,0.65)' },
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: 'rgba(255,255,255,0.65)' },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } },
      },
      series: [
        {
          type: 'bar',
          name: '检测数量',
          data: dist.map((d) => d.count),
          itemStyle: { color: palette.accent, borderRadius: [4, 4, 0, 0] },
          barMaxWidth: 48,
        },
      ],
    };
  }, [stats]);

  /** 目标类型饼图 */
  const typeOption = useMemo<EChartsOption>(() => {
    const byType = stats?.search_by_type ?? {};
    const data = Object.entries(byType).map(([name, value]) => ({ name, value }));
    return {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: {
        bottom: 0,
        textStyle: { color: 'rgba(255,255,255,0.7)' },
      },
      series: [
        {
          type: 'pie',
          radius: ['45%', '68%'],
          center: ['50%', '45%'],
          data,
          label: { color: 'rgba(255,255,255,0.8)' },
          itemStyle: { borderColor: '#141c2b', borderWidth: 2 },
        },
      ],
    };
  }, [stats]);

  /** 摄像头检测量 TOP 15 —— 数据来自 /dashboard/cameras 的 today_detections */
  const camBarOption = useMemo<EChartsOption>(() => {
    const top = [...cameras]
      .sort((a, b) => (b.today_detections ?? 0) - (a.today_detections ?? 0))
      .slice(0, 15)
      .reverse();
    return {
      backgroundColor: 'transparent',
      grid: { left: 110, right: 30, top: 16, bottom: 30 },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      xAxis: {
        type: 'value',
        axisLabel: { color: 'rgba(255,255,255,0.65)' },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } },
      },
      yAxis: {
        type: 'category',
        data: top.map((c) => c.name ?? c.camera_id),
        axisLabel: { color: 'rgba(255,255,255,0.65)', fontSize: 11 },
      },
      series: [
        {
          type: 'bar',
          name: '检测数量',
          data: top.map((c) => c.today_detections ?? 0),
          itemStyle: { color: '#13c2c2', borderRadius: [0, 4, 4, 0] },
        },
      ],
    };
  }, [cameras]);

  const cameraColumns: ColumnsType<CameraMeta> = useMemo(
    () => [
      {
        title: '编号',
        dataIndex: 'camera_id',
        width: 120,
        render: (v: string) => <Text style={{ fontFamily: 'monospace' }}>{v}</Text>,
      },
      { title: '名称', dataIndex: 'name', render: (v) => v ?? '--' },
      {
        title: '场景',
        dataIndex: 'scene',
        width: 90,
        render: (v: string | null | undefined) => v ?? '--',
      },
      {
        title: '检测数',
        dataIndex: 'today_detections',
        width: 110,
        sorter: (a, b) => (a.today_detections ?? 0) - (b.today_detections ?? 0),
        render: (v: number | null | undefined) => <Value value={v} digits={0} />,
      },
    ],
    [],
  );

  const dist = stats?.confidence_distribution ?? [];
  const byType = stats?.search_by_type ?? {};
  const topCams = [...cameras]
    .sort((a, b) => (b.today_detections ?? 0) - (a.today_detections ?? 0))
    .slice(0, 15);
  const hasCamDetection = topCams.some((c) => (c.today_detections ?? 0) > 0);

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Card>
        <Row justify="space-between" align="middle">
          <Col>
            <Title level={4} style={{ margin: 0 }}>
              统计概览
            </Title>
            <Text type="secondary" style={{ fontSize: 12 }}>
              系统当前数据概况
            </Text>
          </Col>
          <Col>
            <Space>
              {health && (
                <Tag color={health.status === 'running' ? 'green' : 'default'}>
                  {health.status === 'running' ? '服务正常' : '服务异常'}
                </Tag>
              )}
              <Button icon={<ReloadOutlined />} loading={loading} onClick={() => void load()}>
                刷新
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {error && <Alert type="warning" showIcon message={error} />}

      {/* KPI 卡：缺失显示 --，不填 0 */}
      <Row gutter={[16, 16]}>
        <Col xs={12} md={8}>
          <Card>
            <Statistic
              title="摄像头总数"
              valueRender={() => (
                <Value value={stats?.camera_count} digits={0} style={{ fontSize: 26 }} />
              )}
            />
          </Card>
        </Col>
        <Col xs={12} md={8}>
          <Card>
            <Statistic
              title="检测记录数"
              valueRender={() => (
                <Value value={stats?.instance_count} digits={0} style={{ fontSize: 26 }} />
              )}
            />
          </Card>
        </Col>
        <Col xs={12} md={8}>
          <Card>
            <Statistic
              title="轨迹数"
              valueRender={() => (
                <Value value={stats?.tracklet_count} digits={0} style={{ fontSize: 26 }} />
              )}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="检测置信度分布" size="small">
            {dist.length === 0 || dist.every((d) => d.count === 0) ? (
              <EChart option={{}} empty height={280} emptyText="后端未返回置信度分布" />
            ) : (
              <EChart option={confOption} height={280} />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="目标类型构成" size="small">
            {Object.keys(byType).length === 0 ? (
              <EChart option={{}} empty height={280} emptyText="后端未返回目标类型统计" />
            ) : (
              <EChart option={typeOption} height={280} />
            )}
          </Card>
        </Col>
      </Row>

      <Card title="摄像头检测量 TOP 15" size="small">
        {!hasCamDetection ? (
          <Alert type="info" showIcon message="暂无摄像头检测量数据" />
        ) : (
          <EChart option={camBarOption} height={340} />
        )}
      </Card>

      <Card
        title={
          <Space>
            <span>摄像头列表</span>
            <Tag>{cameras.length} 个</Tag>
          </Space>
        }
        size="small"
      >
        <Table<CameraMeta>
          size="small"
          rowKey="camera_id"
          columns={cameraColumns}
          dataSource={cameras}
          loading={loading}
          pagination={{ pageSize: 12, showSizeChanger: true, showTotal: (t) => `共 ${t} 个` }}
          scroll={{ x: 800 }}
        />
      </Card>
    </Space>
  );
}
