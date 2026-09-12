/**
 * EChart —— 自封装的 ECharts 容器
 *
 * 选型说明：不引入 `echarts-for-react` 包装层，直接用 echarts 核心 API 封装。
 * 好处是完全掌控类型（`EChartsOption` 强类型）、实例销毁与 resize 行为，
 * 避免第三方包装层的类型不匹配问题。
 */

import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';

interface Props {
  option: EChartsOption;
  height?: number | string;
  /** option 变化时是否清空旧配置（默认 true，避免残留系列） */
  notMerge?: boolean;
  style?: React.CSSProperties;
  /** 图表为空时的替代内容 */
  empty?: boolean;
  emptyText?: string;
}

export default function EChart({
  option,
  height = 320,
  notMerge = true,
  style,
  empty = false,
  emptyText = '暂无数据',
}: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  // 初始化 + 销毁
  useEffect(() => {
    if (empty || !ref.current) return;
    const chart = echarts.init(ref.current, 'dark', { renderer: 'canvas' });
    chartRef.current = chart;

    const onResize = () => chart.resize();
    window.addEventListener('resize', onResize);

    // 容器尺寸变化（侧边栏折叠、Tab 切换）也要重绘
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined' && ref.current) {
      ro = new ResizeObserver(() => chart.resize());
      ro.observe(ref.current);
    }

    return () => {
      window.removeEventListener('resize', onResize);
      ro?.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, [empty]);

  // 更新配置
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.setOption(option, { notMerge });
  }, [option, notMerge]);

  if (empty) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'rgba(255,255,255,0.45)',
          border: '1px dashed rgba(255,255,255,0.15)',
          borderRadius: 8,
          ...style,
        }}
      >
        {emptyText}
      </div>
    );
  }

  return <div ref={ref} style={{ height, width: '100%', ...style }} />;
}
