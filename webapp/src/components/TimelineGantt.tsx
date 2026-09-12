/**
 * TimelineGantt —— 跨镜观测链时间轴（ECharts custom series 甘特图）
 *
 * 展示两类区间，并在视觉上严格区分（本项目核心语义：观测 vs 推断）：
 *   1. 观测段 observation_segments —— 实心矩形，颜色按 basis（强身份绿 / 概率推断橙）
 *   2. 推断关联 inference_segments —— 虚线连接线，从来源摄像头连到目标摄像头
 *
 * 关于时间的诚实处理：
 *   - `observation_segments` 只给 "HH:MM:SS"（不含日期），而 `observation_nodes`
 *     带完整日期。因此这里**用同摄像头的节点时间戳补出日期**，属于对已知信息的
 *     归一化，不是编造；若拿不到日期，则退化为「仅时刻」模式并在图上标注。
 *   - 推断段的 `estimated_travel_time` / `actual_travel_time` 实测为 null，
 *     此时连接线照常绘制（关联关系是真实的），但 tooltip 显示「未知」，
 *     **不画宽度来假装知道行程时长**。
 */

import { useMemo } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';
import type { ObservationNode, ObservationSegment, InferenceSegment } from '../api/types';
import { basisStyle } from '../utils/basis';
import { fmtClock, isNum, isStr } from '../utils/format';
import EChart from './EChart';

interface Props {
  nodes: ObservationNode[];
  segments: ObservationSegment[];
  inferences: InferenceSegment[];
  /** 摄像头顺序（用于 y 轴排列），来自 camera_sequence */
  cameraSequence: string[];
}

interface Row {
  cameraId: string;
  cameraName: string;
}

/** 把 "HH:MM:SS" + 日期 组装成毫秒时间；失败返回 null（不猜时间） */
function toMs(dateStr: string | null, timeStr: string | null | undefined): number | null {
  if (!isStr(timeStr)) return null;
  const clock = fmtClock(timeStr);
  const full = `${dateStr ?? '1970-01-01'}T${clock}`;
  const t = new Date(full).getTime();
  return Number.isFinite(t) ? t : null;
}

export default function TimelineGantt({ nodes, segments, inferences, cameraSequence }: Props) {
  const model = useMemo(() => {
    // 1) 行：优先用 camera_sequence，其次用出现过的摄像头补全，避免丢数据
    const rows: Row[] = [];
    const seen = new Set<string>();
    const push = (id: string, name?: string | null) => {
      if (!id || seen.has(id)) return;
      seen.add(id);
      rows.push({ cameraId: id, cameraName: isStr(name) ? name : id });
    };
    cameraSequence.forEach((id) => push(id, null));
    nodes.forEach((n) => push(n.camera_id, n.camera_name));
    segments.forEach((s) => push(s.camera_id, s.camera_name));

    const rowIndex = new Map(rows.map((r, i) => [r.cameraId, i]));

    // 2) 日期锚点：camera_id -> "YYYY-MM-DD"（来自带完整时间的观测节点）
    const dateByCamera = new Map<string, string>();
    for (const n of nodes) {
      const m = isStr(n.timestamp) ? n.timestamp.match(/(\d{4}-\d{2}-\d{2})/) : null;
      if (m && !dateByCamera.has(n.camera_id)) dateByCamera.set(n.camera_id, m[1]);
    }
    // 全局兜底日期：任取一个节点日期
    const globalDate =
      dateByCamera.size > 0 ? (Array.from(dateByCamera.values())[0] as string) : null;
    const dateFor = (cam: string) => dateByCamera.get(cam) ?? globalDate;

    // 3) 观测段 -> 甘特条
    const bars: {
      value: [number, number, number];
      itemStyle: { color: string; borderColor: string; borderWidth: number; borderType: string };
      raw: ObservationSegment;
    }[] = [];
    let skipped = 0;
    for (const s of segments) {
      const idx = rowIndex.get(s.camera_id);
      if (idx === undefined) {
        skipped += 1;
        continue;
      }
      const start = toMs(dateFor(s.camera_id), s.start_time);
      const end = toMs(dateFor(s.camera_id), s.end_time);
      if (start === null || end === null || end < start) {
        skipped += 1; // 时间缺失的段不画，避免用假宽度误导
        continue;
      }
      const st = basisStyle(s.basis);
      bars.push({
        value: [idx, start, end],
        itemStyle: {
          color: st.color,
          borderColor: st.color,
          borderWidth: 0,
          borderType: 'solid',
        },
        raw: s,
      });
    }

    // 4) 推断段 -> 虚线连接线
    const links: {
      value: [number, number, number, number];
      itemStyle: { color: string; borderType: string; opacity: number };
      raw: InferenceSegment;
    }[] = [];
    for (const inf of inferences) {
      const a = rowIndex.get(inf.source_camera_id);
      const b = rowIndex.get(inf.target_camera_id);
      if (a === undefined || b === undefined) continue;
      // 用来源段结束时刻 -> 目标段开始时刻作为连接端点（两者都是真实观测边界）
      const srcSeg = segments.find((s) => s.camera_id === inf.source_camera_id);
      const dstSeg = segments.find((s) => s.camera_id === inf.target_camera_id);
      const t1 = srcSeg
        ? (toMs(dateFor(inf.source_camera_id), srcSeg.end_time) ??
           toMs(dateFor(inf.source_camera_id), srcSeg.start_time))
        : null;
      const t2 = dstSeg
        ? (toMs(dateFor(inf.target_camera_id), dstSeg.start_time) ??
           toMs(dateFor(inf.target_camera_id), dstSeg.end_time))
        : null;
      if (t1 === null || t2 === null) continue;
      const st = basisStyle(inf.basis ?? 'probabilistic_inference');
      links.push({
        value: [a, t1, b, t2],
        itemStyle: { color: st.color, borderType: 'dashed', opacity: 0.9 },
        raw: inf,
      });
    }

    return { rows, bars, links, skipped, globalDate, rowIndex };
  }, [nodes, segments, inferences, cameraSequence]);

  const option = useMemo<EChartsOption>(() => {
    const rows = model.rows;

    /** 观测段矩形 */
    const barSeries = {
      type: 'custom' as const,
      name: '观测段',
      renderItem: (params: any, api: any) => {
        const catIdx = api.value(0);
        const start = api.coord([api.value(1), catIdx]);
        const end = api.coord([api.value(2), catIdx]);
        const h = api.size([0, 1])[1] * 0.42;
        const rect = echarts.graphic.clipRectByRect(
          { x: start[0], y: start[1] - h / 2, width: end[0] - start[0], height: h },
          {
            x: params.coordSys.x,
            y: params.coordSys.y,
            width: params.coordSys.width,
            height: params.coordSys.height,
          },
        );
        if (!rect) return undefined;
        return { type: 'rect', shape: rect, style: api.style() };
      },
      encode: { x: [1, 2], y: 0 },
      data: model.bars,
      tooltip: {
        formatter: (p: any) => {
          const raw = p.data?.raw as ObservationSegment | undefined;
          if (!raw) return '';
          const st = basisStyle(raw.basis);
          return [
            `<b>${raw.camera_name ?? raw.camera_id}</b>`,
            `区间：${fmtClock(raw.start_time)} ~ ${fmtClock(raw.end_time)}`,
            `方向：${isStr(raw.direction) ? raw.direction : '未知'}`,
            `依据：<span style="color:${st.color}">${st.label}</span>`,
            isStr(raw.basis_text) ? `<span style="opacity:.7">${raw.basis_text}</span>` : '',
          ]
            .filter(Boolean)
            .join('<br/>');
        },
      },
      z: 3,
    };

    /** 推断段虚线连接 */
    const linkSeries = {
      type: 'custom' as const,
      name: '推断关联',
      renderItem: (_params: any, api: any) => {
        const a = api.coord([api.value(1), api.value(0)]);
        const b = api.coord([api.value(3), api.value(2)]);
        return {
          type: 'line',
          shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] },
          style: {
            stroke: api.style().stroke ?? '#faad14',
            lineWidth: 1.5,
            lineDash: [5, 4],
            opacity: 0.85,
          },
          z: 2,
        };
      },
      encode: { x: [1, 3], y: [0, 2] },
      data: model.links,
      tooltip: {
        formatter: (p: any) => {
          const raw = p.data?.raw as InferenceSegment | undefined;
          if (!raw) return '';
          const st = basisStyle(raw.basis ?? 'probabilistic_inference');
          const travel = isNum(raw.estimated_travel_time)
            ? `${raw.estimated_travel_time.toFixed(1)} 秒（估计）`
            : '未知';
          const actual = isNum(raw.actual_travel_time)
            ? `${raw.actual_travel_time.toFixed(1)} 秒（实测）`
            : '未知';
          return [
            `<b>推断段</b> ${raw.source_camera_name ?? raw.source_camera_id} → ${
              raw.target_camera_name ?? raw.target_camera_id
            }`,
            `置信度：${isNum(raw.confidence) ? raw.confidence.toFixed(3) : '未知'}`,
            `估计行程时间：${travel}`,
            `实际行程时间：${actual}`,
            `依据：<span style="color:${st.color}">${st.label}</span>`,
            isStr(raw.route_description)
              ? `<span style="opacity:.7">${raw.route_description}</span>`
              : '',
          ]
            .filter(Boolean)
            .join('<br/>');
        },
      },
      z: 2,
    };

    return {
      backgroundColor: 'transparent',
      grid: { left: 130, right: 24, top: 28, bottom: 56 },
      legend: {
        top: 0,
        textStyle: { color: 'rgba(255,255,255,0.7)' },
        data: ['观测段', '推断关联'],
      },
      tooltip: {
        trigger: 'item',
        backgroundColor: 'rgba(20,28,43,0.95)',
        borderColor: 'rgba(255,255,255,0.15)',
        textStyle: { color: 'rgba(255,255,255,0.88)', fontSize: 12 },
      },
      xAxis: {
        type: 'time',
        axisLabel: {
          color: 'rgba(255,255,255,0.65)',
          formatter: (v: number) => {
            const d = new Date(v);
            const p = (n: number) => String(n).padStart(2, '0');
            return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
          },
        },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } },
      },
      yAxis: {
        type: 'category',
        data: rows.map((r) => r.cameraName),
        axisLabel: { color: 'rgba(255,255,255,0.75)' },
        axisLine: { lineStyle: { color: 'rgba(255,255,255,0.2)' } },
        splitLine: { show: true, lineStyle: { color: 'rgba(255,255,255,0.06)' } },
      },
      series: [barSeries, linkSeries] as unknown as EChartsOption['series'],
      dataZoom: [
        { type: 'inside', xAxisIndex: 0 },
        { type: 'slider', xAxisIndex: 0, height: 16, bottom: 12, textStyle: { color: '#aaa' } },
      ],
    };
  }, [model]);

  const hasData = model.bars.length > 0 || model.links.length > 0;
  if (!hasData) {
    return (
      <EChart
        option={{}}
        empty
        height={340}
        emptyText="该目标没有可绘制的观测段（后端未返回有效时间区间）"
      />
    );
  }

  return (
    <div>
      <EChart option={option} height={340} />
      <div style={{ color: 'rgba(255,255,255,0.45)', fontSize: 12, marginTop: 4 }}>
        {model.globalDate
          ? `日期锚定：${model.globalDate}（由观测节点时间戳补全，观测段本身只含时刻）`
          : '未取到日期，时间轴仅表示时刻'}
        {model.skipped > 0 &&
          ` · 有 ${model.skipped} 段因时间字段缺失未绘制（不估算、不补值）`}
      </div>
    </div>
  );
}
