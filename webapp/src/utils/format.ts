/**
 * 空值安全的格式化工具 —— 「不编造数据」红线的唯一落点
 *
 * 项目红线：后端大量字段可能为 null（实测 `actual_travel_time`、`plate_number`、
 * `observation_nodes[].keyframe_path` 等均为 null）。UI 必须显式呈现「未知」，
 * **绝不允许**用 0 / 0.5 / 随机数等假数字替代。
 *
 * 因此本模块所有函数：
 *   - 输入 null / undefined / NaN / 空串 时，一律返回 fallback（默认 '--'）
 *   - 只有拿到真实数值才做格式化
 * 任何展示数值的地方都必须经过这里，不要在页面里直接 `value.toFixed(2)`。
 */

/** 统一的「无数据」占位符 */
export const DASH = '--';
/** 语义更明确的「未知」（用于置信度/时间等推断类字段） */
export const UNKNOWN = '未知';

/** 判断是否为可用的有限数字（null / undefined / NaN / Infinity 都算缺失） */
export function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/** 判断是否为可用字符串（非空、非纯空白） */
export function isStr(v: unknown): v is string {
  return typeof v === 'string' && v.trim().length > 0;
}

/** 通用兜底：缺失则返回 DASH */
export function orDash(v: string | number | null | undefined): string {
  if (isNum(v)) return String(v);
  if (isStr(v)) return v;
  return DASH;
}

/** 数值保留 n 位小数；缺失返回 DASH */
export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (!isNum(v)) return DASH;
  return v.toFixed(digits);
}

/** 分数展示：0~1 的分数同时给出百分比，便于阅读；缺失返回 DASH */
export function fmtScore(v: number | null | undefined, digits = 4): string {
  if (!isNum(v)) return DASH;
  return v.toFixed(digits);
}

/** 百分比展示（输入 0~1）；缺失返回 DASH */
export function fmtPercent(v: number | null | undefined, digits = 1): string {
  if (!isNum(v)) return DASH;
  return `${(v * 100).toFixed(digits)}%`;
}

/** antd Progress 的 percent 入参：缺失返回 null，让调用方渲染「未知」而不是 0% */
export function toPercent(v: number | null | undefined): number | null {
  if (!isNum(v)) return null;
  return Math.round(v * 1000) / 10; // 保留一位小数
}

/** 时长（秒）展示；缺失返回 DASH */
export function fmtDuration(seconds: number | null | undefined): string {
  if (!isNum(seconds)) return DASH;
  if (seconds < 60) return `${seconds.toFixed(seconds % 1 === 0 ? 0 : 1)} 秒`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m} 分 ${s} 秒`;
}

/** 距离（米）；缺失返回 DASH */
export function fmtMeters(v: number | null | undefined): string {
  if (!isNum(v)) return DASH;
  if (v >= 1000) return `${(v / 1000).toFixed(2)} km`;
  return `${v.toFixed(1)} m`;
}

/** 置信度语义化：保留数字，附加「未知」提示由调用方决定 */
export function fmtConfidence(v: number | null | undefined): string {
  if (!isNum(v)) return UNKNOWN;
  return v.toFixed(4);
}

/** 角度/方向（后端 direction 可能是数字或中文描述） */
export function fmtDirection(v: string | number | null | undefined): string {
  if (isNum(v)) return `${v.toFixed(0)}°`;
  if (isStr(v)) return v;
  return DASH;
}

/** 坐标对：任一缺失就整体视为未知，避免出现「纬度有、经度没有」的假点 */
export function fmtLatLng(
  lat: number | null | undefined,
  lng: number | null | undefined,
): string {
  if (!isNum(lat) || !isNum(lng)) return DASH;
  return `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
}

/** 安全取属性值（attributes 的键是中文，如 颜色/车型） */
export function attrValue(
  attrs: Record<string, string | number | null> | null | undefined,
  key: string,
): string {
  if (!attrs) return DASH;
  const v = attrs[key];
  if (isNum(v)) return String(v);
  if (isStr(v)) return v;
  return DASH;
}

/** 把 attributes 对象转成可展示的 [键, 值] 列表（过滤掉空值键） */
export function attrEntries(
  attrs: Record<string, string | number | null> | null | undefined,
): [string, string][] {
  if (!attrs) return [];
  return Object.entries(attrs)
    .filter(([k]) => isStr(k))
    .map(([k, v]) => [k, isNum(v) ? String(v) : isStr(v) ? v : DASH] as [string, string]);
}

/**
 * 时间戳展示：直接使用后端返回的字符串（后端给的就是带日期的可读字符串）
 * 缺失返回 DASH。不改写、不猜测时区。
 */
export function fmtTimestamp(v: string | null | undefined): string {
  if (!isStr(v)) return DASH;
  return v;
}

/** 只取 "HH:MM:SS" 部分用于时间轴标签；缺失返回 DASH */
export function fmtClock(v: string | null | undefined): string {
  if (!isStr(v)) return DASH;
  const m = v.match(/(\d{1,2}:\d{2}:\d{2})/);
  return m ? m[1] : v;
}

/** 取日期部分 "YYYY-MM-DD"；缺失返回 null（供时间轴锚定日期用） */
export function datePart(v: string | null | undefined): string | null {
  if (!isStr(v)) return null;
  const m = v.match(/(\d{4}-\d{2}-\d{2})/);
  return m ? m[1] : null;
}

/**
 * 解析时间戳为毫秒时间。解析失败返回 null。
 * 注意：后端时间戳形如 "2020-01-01 00:01:02.799"，在 Safari/部分环境
 * `new Date("2020-01-01 00:01:02")` 会失败，因此显式替换空格为 T。
 */
export function parseTime(v: string | null | undefined): number | null {
  if (!isStr(v)) return null;
  const normalized = v.trim().replace(' ', 'T');
  const t = new Date(normalized).getTime();
  return Number.isFinite(t) ? t : null;
}

/**
 * 目标类型本地化。后端返回 vehicle / pedestrian / non_motor_vehicle，
 * 未知取值原样返回，不做臆测。
 */
export function fmtTargetType(v: string | null | undefined): string {
  if (!isStr(v)) return DASH;
  const map: Record<string, string> = {
    vehicle: '车辆',
    pedestrian: '行人',
    non_motor_vehicle: '非机动车',
    车辆: '车辆',
    行人: '行人',
    非机动车: '非机动车',
  };
  return map[v] ?? v;
}
