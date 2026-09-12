/**
 * SearchPage —— 目标检索（页面 1，核心流程）
 *
 * 流程：输入自然语言描述 / 车牌 → 调 /search/query 或 /search/plate
 *       → 图片网格展示候选（Image.PreviewGroup 可预览）
 *       → 选中一个 → 调 /confirm/target 确认 → 跳转 /backtrack 回溯
 *
 * 诚实性约定：候选数值全部走 Value/Nullable 空值安全渲染。
 * /search/plate 返回的是**精简结构**（无 clip_score / final_score / camera_name），
 * 这类字段在该 tab 下显示「--」，不填 0。
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Alert,
  Button,
  Card,
  Col,
  Empty,
  Image,
  Input,
  Row,
  Segmented,
  Space,
  Spin,
  Tag,
  Typography,
  App as AntApp,
} from 'antd';
import { SearchOutlined, AimOutlined } from '@ant-design/icons';
import type { Candidate } from '../api/types';
import { confirmTarget, searchByPlate, searchByText } from '../api/endpoints';
import { describeError } from '../api/client';
import CandidateCard from '../components/CandidateCard';
import { staticUrl } from '../utils/media';

const { Title, Paragraph, Text } = Typography;

type Mode = 'text' | 'plate';

const EXAMPLES = ['黑色轿车', '白色汽车', '蓝色卡车', '银色面包车'];

export default function SearchPage() {
  const navigate = useNavigate();
  const { message } = AntApp.useApp();

  const [mode, setMode] = useState<Mode>('text');
  const [text, setText] = useState('');
  const [plate, setPlate] = useState('');
  const [topK, setTopK] = useState(20);

  const [loading, setLoading] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [queryId, setQueryId] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [searched, setSearched] = useState(false);

  async function runSearch(q: string) {
    const keyword = q.trim();
    if (!keyword) {
      message.warning(mode === 'text' ? '请输入目标描述' : '请输入车牌号');
      return;
    }
    setLoading(true);
    setError(null);
    setSelected(null);
    try {
      const res =
        mode === 'text'
          ? await searchByText(keyword, topK)
          : await searchByPlate(keyword);
      setQueryId(res.query_id);
      setCandidates(res.candidates ?? []);
      setSearched(true);
      if ((res.candidates ?? []).length === 0) {
        message.info('后端返回 0 个候选，请尝试更换描述或放宽条件');
      }
    } catch (e) {
      setError(describeError(e));
      setCandidates([]);
      setQueryId(null);
      setSearched(true);
    } finally {
      setLoading(false);
    }
  }

  async function onConfirm() {
    if (!selected) {
      message.warning('请先选择一个候选目标');
      return;
    }
    if (!queryId) {
      message.error('缺少 query_id，请重新检索');
      return;
    }
    setConfirming(true);
    try {
      const res = await confirmTarget(queryId, selected.instance_id);
      message.success(res.message || '已确认目标，正在进入回溯');
      // 用 URL 承载会话信息，刷新/分享都不丢
      const params = new URLSearchParams({
        instance_id: selected.instance_id,
        query_id: queryId,
      });
      if (selected.track_id) params.set('track_id', selected.track_id);
      navigate(`/backtrack?${params.toString()}`);
    } catch (e) {
      message.error(describeError(e));
    } finally {
      setConfirming(false);
    }
  }

  const previewItems = candidates
    .map((c) => staticUrl(c.keyframe_path))
    .filter((u): u is string => Boolean(u));

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Card>
        <Title level={4} style={{ marginTop: 0, marginBottom: 4 }}>
          查找目标
        </Title>
        <Paragraph type="secondary" style={{ marginBottom: 12, fontSize: 13 }}>
          用一句话描述目标（例如「黑色轿车」），或按车牌精确查找。
          系统会返回候选目标，需人工确认后才会进入跨镜回溯。
        </Paragraph>

        <Segmented
          value={mode}
          onChange={(v) => {
            setMode(v as Mode);
            setCandidates([]);
            setSelected(null);
            setQueryId(null);
            setSearched(false);
            setError(null);
          }}
          options={[
            { label: '按描述查找', value: 'text' },
            { label: '按车牌查找', value: 'plate' },
          ]}
          style={{ marginBottom: 12 }}
        />

        {mode === 'text' ? (
          <>
            <Space.Compact style={{ width: '100%', maxWidth: 720 }}>
              <Input
                size="large"
                allowClear
                placeholder="请输入目标描述，例如：黑色轿车"
                value={text}
                onChange={(e) => setText(e.target.value)}
                onPressEnter={() => runSearch(text)}
                prefix={<SearchOutlined />}
              />
              <Button
                size="large"
                type="primary"
                loading={loading}
                onClick={() => runSearch(text)}
              >
                检索
              </Button>
            </Space.Compact>
            <div style={{ marginTop: 8 }}>
              <Text type="secondary" style={{ fontSize: 12, marginRight: 8 }}>
                示例：
              </Text>
              {EXAMPLES.map((e) => (
                <Tag
                  key={e}
                  style={{ cursor: 'pointer' }}
                  onClick={() => {
                    setText(e);
                    void runSearch(e);
                  }}
                >
                  {e}
                </Tag>
              ))}
            </div>
            <div style={{ marginTop: 8 }}>
              <Text type="secondary" style={{ fontSize: 12, marginRight: 8 }}>
                返回数量：
              </Text>
              <Segmented
                size="small"
                value={String(topK)}
                onChange={(v) => setTopK(Number(v))}
                options={['10', '20', '50'].map((n) => ({ label: n, value: n }))}
              />
            </div>
          </>
        ) : (
          <>
            <Space.Compact style={{ width: '100%', maxWidth: 720 }}>
              <Input
                size="large"
                allowClear
                placeholder="请输入车牌号（支持部分匹配），例如：A12345"
                value={plate}
                onChange={(e) => setPlate(e.target.value)}
                onPressEnter={() => runSearch(plate)}
              />
              <Button
                size="large"
                type="primary"
                loading={loading}
                onClick={() => runSearch(plate)}
              >
                查询
              </Button>
            </Space.Compact>
            <Alert
              style={{ marginTop: 8 }}
              type="info"
              showIcon
              message="车牌查询支持部分匹配，返回基本匹配信息"
            />
          </>
        )}
      </Card>

      {error && <Alert type="error" showIcon message="检索失败" description={error} />}

      {loading && (
        <Card>
          <div style={{ textAlign: 'center', padding: 32 }}>
            <Spin size="large" />
            <div style={{ marginTop: 12 }}>
              <Text type="secondary">正在查找…</Text>
            </div>
          </div>
        </Card>
      )}

      {!loading && searched && (
        <Card
          title={
            <Space>
              <span>候选目标</span>
              <Tag color="blue">{candidates.length} 个</Tag>
            </Space>
          }
          extra={
            <Space>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {selected ? `已选 #${selected.rank ?? '--'}` : '未选择'}
              </Text>
              <Button
                type="primary"
                icon={<AimOutlined />}
                disabled={!selected}
                loading={confirming}
                onClick={onConfirm}
              >
                确认目标并回溯
              </Button>
            </Space>
          }
        >
          {candidates.length === 0 ? (
            <Empty description="没有匹配的候选目标" />
          ) : (
            <>
              {/* 预览组：点击任意候选图可在全屏预览中左右翻页 */}
              <div style={{ display: 'none' }}>
                <Image.PreviewGroup items={previewItems}>
                  {previewItems.map((u) => (
                    <Image key={u} src={u} />
                  ))}
                </Image.PreviewGroup>
              </div>

              <Row gutter={[12, 12]}>
                {candidates.map((c) => (
                  <Col key={c.instance_id} xs={24} sm={12} md={8} lg={6} xxl={4}>
                    <CandidateCard
                      candidate={c}
                      selected={selected?.instance_id === c.instance_id}
                      onSelect={setSelected}
                    />
                  </Col>
                ))}
              </Row>
            </>
          )}
        </Card>
      )}

      {!loading && !searched && (
        <Card>
          <Empty
            description="输入描述或车牌后开始检索"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        </Card>
      )}

      <Text type="secondary" style={{ fontSize: 12 }}>
        点击候选图可放大查看；确认目标后即可进入跨镜回溯。
      </Text>
    </Space>
  );
}
