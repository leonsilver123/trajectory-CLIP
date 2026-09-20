/**
 * TrajectoryPlayer —— 轨迹还原回放（PLAN4）
 *
 * ## 它解决什么问题
 *
 * 原来的轨迹展示是一张静态甘特图：能看到"这辆车经过了哪几个摄像头、各待了多久"，
 * 但看不到"它在画面里怎么动"。本组件把离散观测链**按时间放起来**。
 *
 * ## 为什么不做地图动画（重要）
 *
 * 本数据集 46 个摄像头**只有 4 组不同的 GPS 坐标** —— 每个场景内所有摄像头
 * 共用同一个坐标（场景中心近似值）。地图上的车会在 4 个点之间瞬移，场景内毫无运动。
 * 任何基于 GPS 的动画都是假的。因此这里采用**非地理**的三栏联动方案。
 *
 * ## 三条诚实红线
 *
 * 1. **跨摄像头时不做插值**。那段路没有摄像头拍到，平滑移动等于编造。
 *    推断段只显示占位与说明，**不渲染任何车辆位置**。
 * 2. **画面内的位置是真实的**：来自该帧检测框的归一化中心，不是补出来的。
 * 3. **速度是估算的**：没有相机标定，只能给"像素/秒"，且必须标注。
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Button, Card, Empty, Slider, Space, Tag, Typography } from "antd";
import {
  CaretRightOutlined,
  PauseOutlined,
  StepBackwardOutlined,
  StepForwardOutlined,
} from "@ant-design/icons";
import type { TrackFrame, Trajectory } from "../api/types";
import { staticUrl } from "../utils/media";
import { Nullable, Value } from "./Value";

const { Text, Paragraph } = Typography;

/** 回放阶段：观测段（实拍）或推断段（概率） */
interface Phase {
  kind: "observed" | "inferred";
  cameraId: string;
  /** 推断段才有：从哪个摄像头到哪个摄像头 */
  toCameraId?: string;
  startMs: number;
  endMs: number;
  frames: TrackFrame[];
}

const SPEEDS = [1, 4, 16] as const;

/** 把 "YYYY-MM-DD HH:MM:SS" 解析成毫秒；失败返回 null（不用 0 冒充） */
function parseTs(ts: string | null | undefined): number | null {
  if (!ts) return null;
  const normalized = ts.replace(" ", "T");
  const ms = Date.parse(normalized);
  return Number.isNaN(ms) ? null : ms;
}

/** 取帧的“可比时间”：优先全局时间（跨摄像头可比），退回本机时间 */
function frameTime(f: TrackFrame): number | null {
  return parseTs(f.global_timestamp) ?? parseTs(f.timestamp);
}

/**
 * 把离散观测链摊平成可播放的阶段序列
 *
 * 观测段的时长用该摄像头内首末帧的**全局时间**差；推断段是相邻摄像头之间的空白，
 * 由前一个摄像头的离开时间到下一个摄像头的到达时间构成。
 * 拿不到全局时间时整体退化为"按序等分"，并在 UI 上标注（不假装有真实时长）。
 */
function buildPhases(traj: Trajectory): {
  phases: Phase[];
  timeAccurate: boolean;
} {
  const cams = traj.camera_sequence ?? [];
  if (cams.length === 0) return { phases: [], timeAccurate: false };

  // 先试真实时间轴
  const spans = cams.map((cam) => {
    const frames = cam.frames ?? [];
    const times = frames.map(frameTime).filter((t): t is number => t !== null);
    return {
      cam,
      frames,
      start: times[0] ?? null,
      end: times[times.length - 1] ?? null,
    };
  });

  const timeAccurate = spans.every((s) => s.start !== null && s.end !== null);

  const phases: Phase[] = [];
  if (timeAccurate) {
    let cursor = 0;
    spans.forEach((s, i) => {
      // 观测段
      const dur = Math.max((s.end as number) - (s.start as number), 1);
      // 与前一个摄像头之间的推断段（空白时长）
      if (i > 0) {
        const prevEnd = spans[i - 1].end as number;
        const gap = (s.start as number) - prevEnd;
        if (gap > 0) {
          phases.push({
            kind: "inferred",
            cameraId: spans[i - 1].cam.camera_id,
            toCameraId: s.cam.camera_id,
            startMs: cursor,
            endMs: cursor + gap,
            frames: [],
          });
          cursor += gap;
        }
      }
      phases.push({
        kind: "observed",
        cameraId: s.cam.camera_id,
        startMs: cursor,
        endMs: cursor + dur,
        frames: s.frames,
      });
      cursor += dur;
    });
  } else {
    // 退化：每个观测段等分 1 单位，推断段占 0.5 单位，仅用于"能播"，不声称真实时长
    const UNIT = 1000;
    let cursor = 0;
    spans.forEach((s, i) => {
      if (i > 0) {
        phases.push({
          kind: "inferred",
          cameraId: spans[i - 1].cam.camera_id,
          toCameraId: s.cam.camera_id,
          startMs: cursor,
          endMs: cursor + UNIT / 2,
          frames: [],
        });
        cursor += UNIT / 2;
      }
      phases.push({
        kind: "observed",
        cameraId: s.cam.camera_id,
        startMs: cursor,
        endMs: cursor + UNIT,
        frames: s.frames,
      });
      cursor += UNIT;
    });
  }

  return { phases, timeAccurate };
}

/** 当前时刻落在哪个阶段 */
function phaseAt(phases: Phase[], cursorMs: number): Phase | null {
  for (const p of phases) {
    if (cursorMs >= p.startMs && cursorMs < p.endMs) return p;
  }
  return phases.length ? phases[phases.length - 1] : null;
}

/** 阶段内离当前时刻最近的帧 */
function nearestFrame(
  phase: Phase | null,
  cursorMs: number,
  totalMs: number,
): TrackFrame | null {
  if (!phase || phase.frames.length === 0) return null;
  const span = Math.max(phase.endMs - phase.startMs, 1);
  const ratio = Math.min(Math.max((cursorMs - phase.startMs) / span, 0), 1);
  const idx = Math.min(
    Math.floor(ratio * phase.frames.length),
    phase.frames.length - 1,
  );
  void totalMs;
  return phase.frames[idx];
}

/**
 * 段内速度估算（像素/秒）
 *
 * **这不是 km/h**：把像素换算成米需要相机标定，本项目没有标定数据。
 * 因此只给像素位移速率，并在 UI 上明确标注"估算"。
 */
function estimateSpeed(phase: Phase | null): number | null {
  if (!phase || phase.kind !== "observed" || phase.frames.length < 2)
    return null;
  const withNorm = phase.frames.filter(
    (f) => f.bbox_norm && f.bbox_norm.length >= 2,
  );
  if (withNorm.length < 2) return null;

  const a = withNorm[0].bbox_norm as number[];
  const b = withNorm[withNorm.length - 1].bbox_norm as number[];
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dist = Math.sqrt(dx * dx + dy * dy); // 归一化单位（画面宽/高的比例）

  const t0 = frameTime(withNorm[0]);
  const t1 = frameTime(withNorm[withNorm.length - 1]);
  if (t0 === null || t1 === null || t1 <= t0) return null;

  const seconds = (t1 - t0) / 1000;
  if (seconds <= 0) return null;
  return dist / seconds; // 单位：画面比例/秒
}

interface Props {
  trajectory: Trajectory;
}

export default function TrajectoryPlayer({ trajectory }: Props) {
  const { phases, timeAccurate } = useMemo(
    () => buildPhases(trajectory),
    [trajectory],
  );
  const totalMs = phases.length ? phases[phases.length - 1].endMs : 0;

  const [cursorMs, setCursorMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(4);
  const rafRef = useRef<number | null>(null);
  const lastTsRef = useRef<number | null>(null);

  // 播放循环：按真实经过时间推进游标（与帧率无关）
  useEffect(() => {
    if (!playing || totalMs <= 0) return;
    const step = (now: number) => {
      const last = lastTsRef.current;
      lastTsRef.current = now;
      if (last !== null) {
        const delta = (now - last) * speed;
        setCursorMs((c) => (c + delta >= totalMs ? 0 : c + delta));
      }
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      lastTsRef.current = null;
    };
  }, [playing, speed, totalMs]);

  if (phases.length === 0) {
    return (
      <Card size="small" title="轨迹还原回放">
        <Empty description="该目标没有可回放的摄像头序列" />
      </Card>
    );
  }

  const current = phaseAt(phases, cursorMs);
  const frame = nearestFrame(current, cursorMs, totalMs);
  const isInferred = current?.kind === "inferred";
  const speedEst = estimateSpeed(current);

  const jump = (dir: 1 | -1) => {
    const idx = phases.findIndex((p) => p === current);
    const next = phases[Math.min(Math.max(idx + dir, 0), phases.length - 1)];
    if (next) setCursorMs(next.startMs);
  };

  const norm = frame?.bbox_norm ?? null;
  const cropUrl = staticUrl(frame?.crop_path ?? null);
  const frameSizeKnown =
    frame?.frame_size_source === "inferred_from_detections";

  return (
    <Card size="small" title="轨迹还原回放">
      {/* ── 播放控制 ── */}
      <Space wrap style={{ marginBottom: 12 }}>
        <Button
          icon={playing ? <PauseOutlined /> : <CaretRightOutlined />}
          type="primary"
          onClick={() => setPlaying((p) => !p)}
        >
          {playing ? "暂停" : "播放"}
        </Button>
        <Button icon={<StepBackwardOutlined />} onClick={() => jump(-1)}>
          上一段
        </Button>
        <Button icon={<StepForwardOutlined />} onClick={() => jump(1)}>
          下一段
        </Button>
        <Space size={4}>
          <Text type="secondary">速度</Text>
          {SPEEDS.map((s) => (
            <Tag
              key={s}
              color={s === speed ? "blue" : undefined}
              style={{ cursor: "pointer" }}
              onClick={() => setSpeed(s)}
            >
              {s}x
            </Tag>
          ))}
        </Space>
        <Button
          size="small"
          onClick={() => {
            setCursorMs(0);
            setPlaying(false);
          }}
        >
          回到开头
        </Button>
      </Space>

      <Slider
        min={0}
        max={Math.max(totalMs, 1)}
        value={cursorMs}
        onChange={(v) => {
          setCursorMs(v);
          setPlaying(false);
        }}
        tooltip={{
          formatter: () => (timeAccurate ? formatClock(cursorMs) : "相对进度"),
        }}
      />

      {!timeAccurate && (
        <Paragraph type="warning" style={{ fontSize: 12, marginTop: -4 }}>
          部分帧缺少全局时间戳，时长按段等分显示 —— <b>这不是真实时长</b>。
        </Paragraph>
      )}

      <div
        style={{
          display: "flex",
          gap: 16,
          flexWrap: "wrap",
          alignItems: "flex-start",
        }}
      >
        {/* ── ① 摄像头泳道 ── */}
        <div style={{ flex: "2 1 380px", minWidth: 300 }}>
          <Text strong style={{ fontSize: 13 }}>
            ① 观测链泳道
          </Text>
          <Text type="secondary" style={{ fontSize: 12, marginLeft: 8 }}>
            实心=观测段（实拍） 虚线=推断段（概率）
          </Text>
          <div style={{ marginTop: 8 }}>
            {phases.map((p, i) => {
              const left = totalMs > 0 ? (p.startMs / totalMs) * 100 : 0;
              const width =
                totalMs > 0 ? ((p.endMs - p.startMs) / totalMs) * 100 : 0;
              const active = p === current;
              const isObs = p.kind === "observed";
              return (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    marginBottom: 3,
                  }}
                >
                  <div
                    style={{
                      width: 148,
                      flex: "none",
                      fontFamily: "monospace",
                      fontSize: 11,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      color: active ? "#1668dc" : undefined,
                    }}
                    title={
                      isObs ? p.cameraId : `${p.cameraId} → ${p.toCameraId}`
                    }
                  >
                    {isObs ? p.cameraId : `${p.cameraId} → ${p.toCameraId}`}
                  </div>
                  <div
                    style={{
                      flex: 1,
                      position: "relative",
                      height: 16,
                      background: "#f0f0f0",
                      borderRadius: 2,
                    }}
                  >
                    <div
                      title={
                        isObs ? "观测段（实拍）" : "推断段（概率推断，非实拍）"
                      }
                      style={{
                        position: "absolute",
                        left: `${left}%`,
                        width: `${Math.max(width, 0.6)}%`,
                        height: "100%",
                        borderRadius: 2,
                        background: isObs
                          ? active
                            ? "#1668dc"
                            : "#91caff"
                          : "transparent",
                        border: isObs ? "none" : "1px dashed #d48806",
                        boxSizing: "border-box",
                      }}
                    />
                  </div>
                </div>
              );
            })}
            {/* 游标 */}
            <div style={{ display: "flex", marginTop: 2 }}>
              <div style={{ width: 148, flex: "none" }} />
              <div style={{ flex: 1, position: "relative", height: 6 }}>
                <div
                  style={{
                    position: "absolute",
                    left: `${totalMs > 0 ? (cursorMs / totalMs) * 100 : 0}%`,
                    width: 2,
                    height: 6,
                    background: "#cf1322",
                    borderRadius: 1,
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* ── ② 当前视野 ── */}
        <div style={{ flex: "1 1 260px", minWidth: 240 }}>
          <Text strong style={{ fontSize: 13 }}>
            ② 画面内位置
          </Text>
          <div style={{ marginTop: 8 }}>
            {isInferred ? (
              <div
                style={{
                  border: "1px dashed #d48806",
                  borderRadius: 4,
                  padding: 20,
                  textAlign: "center",
                  color: "#d48806",
                  fontSize: 12,
                  background: "#fffbe6",
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: 4 }}>
                  推断段 · 非实拍
                </div>
                <div>
                  该区间没有摄像头拍到目标，
                  <br />
                  系统仅给出时空可达性判断，
                  <br />
                  <b>不渲染任何车辆位置</b>。
                </div>
              </div>
            ) : norm ? (
              <div
                style={{
                  position: "relative",
                  width: "100%",
                  aspectRatio: "4 / 3",
                  background: "#fafafa",
                  border: "1px solid #d9d9d9",
                  borderRadius: 4,
                  overflow: "hidden",
                }}
              >
                {/* 画面内真实位置（来自该帧检测框的归一化中心） */}
                <div
                  style={{
                    position: "absolute",
                    left: `${(norm[0] - norm[2] / 2) * 100}%`,
                    top: `${(norm[1] - norm[3] / 2) * 100}%`,
                    width: `${norm[2] * 100}%`,
                    height: `${norm[3] * 100}%`,
                    border: "2px solid #cf1322",
                    borderRadius: 2,
                    background: "rgba(207,19,34,0.08)",
                    boxSizing: "border-box",
                  }}
                />
                <div
                  style={{
                    position: "absolute",
                    left: 6,
                    bottom: 4,
                    fontSize: 11,
                    color: "#8c8c8c",
                  }}
                >
                  {frame?.camera_id ?? current?.cameraId}
                </div>
              </div>
            ) : (
              <div
                style={{
                  border: "1px solid #d9d9d9",
                  borderRadius: 4,
                  padding: 20,
                  textAlign: "center",
                  color: "#8c8c8c",
                  fontSize: 12,
                  background: "#fafafa",
                }}
              >
                该帧没有检测框数据
              </div>
            )}

            {!isInferred && !frameSizeKnown && frame && (
              <Text type="secondary" style={{ fontSize: 11 }}>
                画面比例未知，上方矩形为固定比例占位
              </Text>
            )}

            {cropUrl && !isInferred && (
              <div style={{ marginTop: 8 }}>
                <Text type="secondary" style={{ fontSize: 11 }}>
                  该帧实拍裁剪图
                </Text>
                <img
                  src={cropUrl}
                  alt="目标裁剪图"
                  style={{
                    width: "100%",
                    marginTop: 2,
                    borderRadius: 3,
                    border: "1px solid #f0f0f0",
                  }}
                />
              </div>
            )}
          </div>
        </div>

        {/* ── ③ 参数面板 ── */}
        <div style={{ flex: "1 1 220px", minWidth: 200 }}>
          <Text strong style={{ fontSize: 13 }}>
            ③ 参数
          </Text>
          <div style={{ marginTop: 8, fontSize: 12 }}>
            <Row
              label="阶段"
              value={isInferred ? "推断段（概率）" : "观测段（实拍）"}
            />
            <Row label="摄像头" value={current?.cameraId} />
            {isInferred && <Row label="前往" value={current?.toCameraId} />}
            <Row label="全局时间" value={frame?.global_timestamp} />
            <Row label="本机时间" value={frame?.timestamp} />
            <Row
              label="方向"
              value={directionOf(trajectory, current?.cameraId)}
            />
            <Row label="检测置信度" value={frame?.confidence} digits={3} />
            <Row
              label="段内速度"
              value={speedEst}
              digits={4}
              suffix={speedEst !== null ? " 画面比例/秒（估算）" : undefined}
            />
            <Row label="帧号" value={frame?.frame_id} digits={0} />
          </div>
          <Paragraph
            type="secondary"
            style={{ fontSize: 11, marginTop: 8, marginBottom: 0 }}
          >
            速度由检测框位移估算，<b>不是 km/h</b> ——
            像素到米的换算需要相机标定，本项目没有标定数据。
          </Paragraph>
        </div>
      </div>
    </Card>
  );
}

function directionOf(
  traj: Trajectory,
  cameraId?: string | null,
): string | null {
  if (!cameraId) return null;
  const cam = (traj.camera_sequence ?? []).find(
    (c) => c.camera_id === cameraId,
  );
  return cam?.direction ?? null;
}

function formatClock(ms: number): string {
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** 一行「标签 : 值」；值缺失时统一显示 —（数字走 Value，字符串走 Nullable） */
function Row({
  label,
  value,
  digits,
  suffix,
}: {
  label: string;
  value: unknown;
  digits?: number;
  suffix?: string;
}) {
  const isNumber = typeof value === "number";
  const missing = value === null || value === undefined || value === "";
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        gap: 8,
        padding: "2px 0",
      }}
    >
      <Text type="secondary" style={{ fontSize: 12 }}>
        {label}
      </Text>
      <span style={{ textAlign: "right" }}>
        {isNumber ? (
          <Value value={value as number} digits={digits} />
        ) : (
          <Nullable value={value as string | null | undefined} />
        )}
        {suffix && !missing && (
          <Text type="secondary" style={{ fontSize: 11 }}>
            {suffix}
          </Text>
        )}
      </span>
    </div>
  );
}
