/**
 * CameraMap —— 跨镜路径地图（Leaflet + react-leaflet v4）
 *
 * 展示：
 *   - 观测节点：实心圆点（颜色按 basis），弹窗显示摄像头、时间、置信度
 *   - 观测路径：实线折线（按 camera_sequence 串联**有经纬度**的节点）
 *   - 推断段：虚线折线（从来源摄像头到目标摄像头）
 *
 * 坐标诚实性：只有后端真的给了 latitude/longitude 的摄像头才会上图。
 * 缺坐标的摄像头不会用「相邻摄像头坐标」或默认点代替——那属于编造位置。
 * 缺坐标的摄像头数量会明确提示出来。
 *
 * 瓦片源：
 *   - 默认 CartoDB dark（免 key，深色主题协调）
 *   - 可切换 OSM 标准瓦片
 *   - 生产升级：天地图 WMTS（需申请 key，CGCS2000 坐标系，无需 GCJ-02 纠偏）。
 *     接入方式见下方 TIANDITU 常量注释。
 */

import { useMemo, useState } from 'react';
import { MapContainer, Marker, Polyline, Popup, TileLayer, Tooltip as LTooltip } from 'react-leaflet';
import L from 'leaflet';
import { Segmented, Typography } from 'antd';
import type { ObservationNode, ObservationSegment, InferenceSegment } from '../api/types';
import { basisStyle } from '../utils/basis';
import { fmtClock, fmtLatLng, isNum, isStr } from '../utils/format';
import { palette } from '../theme';

const { Text } = Typography;

type TileKey = 'dark' | 'osm';

const TILES: Record<TileKey, { url: string; attribution: string; label: string }> = {
  dark: {
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    label: '深色地图',
  },
  osm: {
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; OpenStreetMap contributors',
    label: 'OSM 标准',
  },
};

/**
 * 生产环境可替换为天地图 WMTS（免 GCJ-02 纠偏）：
 *   https://t{s}.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0
 *     &LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles
 *     &TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=<YOUR_KEY>
 * 注意：天地图需在官网申请 tk，且为 CGCS2000 坐标，与后端经纬度一致，不需要纠偏。
 */

/** 用 divIcon 自定义标记，规避 Leaflet 默认图标在打包器下的图片路径问题 */
function dotIcon(color: string, label: string): L.DivIcon {
  return L.divIcon({
    className: 'camera-map-dot',
    html: `<div style="
        width:26px;height:26px;border-radius:50%;
        background:${color};border:2px solid rgba(0,0,0,0.55);
        box-shadow:0 0 0 3px ${color}44;
        display:flex;align-items:center;justify-content:center;
        color:#fff;font-size:11px;font-weight:600;line-height:1;">${label}</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 13],
  });
}

interface Props {
  nodes: ObservationNode[];
  segments: ObservationSegment[];
  inferences: InferenceSegment[];
  cameraSequence: string[];
}

interface Placed {
  node: ObservationNode;
  lat: number;
  lng: number;
  order: number; // 在 camera_sequence 中的次序
}

export default function CameraMap({ nodes, segments, inferences, cameraSequence }: Props) {
  const [tile, setTile] = useState<TileKey>('dark');

  const { placed, missing, center, zoom } = useMemo(() => {
    const order = new Map(cameraSequence.map((c, i) => [c, i]));

    const ok: Placed[] = [];
    let missingCount = 0;
    for (const n of nodes) {
      if (isNum(n.latitude) && isNum(n.longitude)) {
        ok.push({
          node: n,
          lat: n.latitude,
          lng: n.longitude,
          order: order.get(n.camera_id) ?? Number.MAX_SAFE_INTEGER,
        });
      } else {
        missingCount += 1;
      }
    }
    // 按 camera_sequence 排序，保证折线顺序 = 时序
    ok.sort((a, b) => a.order - b.order);

    // 兜底视野：数据集真实位置为美国 Iowa Dubuque（场景中心），不用虚构坐标
    let c: [number, number] = [42.525678, -90.723601];
    if (ok.length > 0) {
      const lat = ok.reduce((s, p) => s + p.lat, 0) / ok.length;
      const lng = ok.reduce((s, p) => s + p.lng, 0) / ok.length;
      c = [lat, lng];
    }

    return { placed: ok, missing: missingCount, center: c, zoom: ok.length > 1 ? 15 : 16 };
  }, [nodes, cameraSequence]);

  /** 观测路径折线（只连有坐标的节点） */
  const solidLines = useMemo(() => {
    const byCamera = new Map(placed.map((p) => [p.node.camera_id, p]));
    const ordered: [number, number][] = [];
    for (const cam of cameraSequence) {
      const p = byCamera.get(cam);
      if (p) ordered.push([p.lat, p.lng]);
    }
    // camera_sequence 里没有但节点里有的，追加到末尾，避免丢点
    for (const p of placed) {
      if (!cameraSequence.includes(p.node.camera_id)) ordered.push([p.lat, p.lng]);
    }
    return ordered;
  }, [placed, cameraSequence]);

  /** 推断段虚线（用推断段自带的源/目标坐标，缺失则跳过） */
  const dashedLines = useMemo(() => {
    const out: { pts: [number, number][]; inf: InferenceSegment }[] = [];
    for (const inf of inferences) {
      if (
        isNum(inf.source_latitude) &&
        isNum(inf.source_longitude) &&
        isNum(inf.target_latitude) &&
        isNum(inf.target_longitude)
      ) {
        out.push({
          pts: [
            [inf.source_latitude, inf.source_longitude],
            [inf.target_latitude, inf.target_longitude],
          ],
          inf,
        });
      }
    }
    return out;
  }, [inferences]);

  const segOf = (cam: string) => segments.find((s) => s.camera_id === cam);

  if (placed.length === 0) {
    return (
      <div
        style={{
          height: 380,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: palette.textDim,
          border: `1px dashed ${palette.border}`,
          borderRadius: 8,
          textAlign: 'center',
          padding: 16,
        }}
      >
        后端未返回任何带经纬度的观测节点，无法绘图。
        <br />
        （不使用默认坐标占位，避免展示虚构位置）
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
        <Segmented
          size="small"
          value={tile}
          onChange={(v) => setTile(v as TileKey)}
          options={[
            { label: TILES.dark.label, value: 'dark' },
            { label: TILES.osm.label, value: 'osm' },
          ]}
        />
        <Text type="secondary" style={{ fontSize: 12 }}>
          观测 {placed.length} 点 · 推断 {dashedLines.length} 段
          {missing > 0 && ` · ${missing} 个节点无经纬度未上图`}
        </Text>
      </div>

      <MapContainer
        center={center}
        zoom={zoom}
        style={{ height: 380, width: '100%', borderRadius: 8, background: '#0b1220' }}
        scrollWheelZoom
      >
        <TileLayer url={TILES[tile].url} attribution={TILES[tile].attribution} />

        {/* 观测路径实线 */}
        {solidLines.length > 1 && (
          <Polyline
            positions={solidLines}
            pathOptions={{ color: palette.strong, weight: 2, opacity: 0.75 }}
          />
        )}

        {/* 推断段虚线 */}
        {dashedLines.map((d, i) => (
          <Polyline
            key={`inf-${i}`}
            positions={d.pts}
            pathOptions={{
              color: basisStyle(d.inf.basis ?? 'probabilistic_inference').color,
              weight: 2,
              dashArray: '6 6',
              opacity: 0.9,
            }}
          >
            <LTooltip sticky>
              推断段 {d.inf.source_camera_name ?? d.inf.source_camera_id} →{' '}
              {d.inf.target_camera_name ?? d.inf.target_camera_id}
              <br />
              置信度：{isNum(d.inf.confidence) ? d.inf.confidence.toFixed(3) : '未知'}
              <br />
              估计行程时间：
              {isNum(d.inf.estimated_travel_time)
                ? `${d.inf.estimated_travel_time.toFixed(1)} 秒`
                : '未知'}
            </LTooltip>
          </Polyline>
        ))}

        {/* 观测节点标记 */}
        {placed.map((p, i) => {
          const st = basisStyle(p.node.basis);
          const seg = segOf(p.node.camera_id);
          return (
            <Marker
              key={p.node.camera_id}
              position={[p.lat, p.lng]}
              icon={dotIcon(st.color, String(i + 1))}
            >
              <Popup>
                <div style={{ fontSize: 12, lineHeight: 1.7 }}>
                  <b>{p.node.camera_name ?? p.node.camera_id}</b>
                  <br />
                  次序：第 {i + 1} 站（camera_sequence）
                  <br />
                  时间：{isStr(p.node.timestamp) ? p.node.timestamp : '未知'}
                  <br />
                  置信度：
                  {isNum(p.node.confidence) ? p.node.confidence.toFixed(3) : '未知'}
                  <br />
                  依据：
                  <span style={{ color: st.color }}>{st.label}</span>
                  <br />
                  近似位置（场景中心）：{fmtLatLng(p.lat, p.lng)}
                  {seg && (
                    <>
                      <br />
                      观测区间：{fmtClock(seg.start_time)} ~ {fmtClock(seg.end_time)}
                    </>
                  )}
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>

      <div style={{ marginTop: 6, fontSize: 12, color: palette.textDim }}>
        实线 = 观测到的摄像头序列；虚线 = 摄像头之间的推断段（未观测）。
        瓦片为免 key 的 {TILES[tile].label}；生产可换天地图 WMTS（需申请 key）。
      </div>
    </div>
  );
}
