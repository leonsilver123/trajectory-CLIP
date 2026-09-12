/**
 * BasisTag —— 依据（强身份 / 概率推断）可视化标签
 *
 * 观测链的核心语义：用户必须一眼看出哪些结论是硬标识匹配、哪些是概率推断。
 * basis 缺失时显示「依据未知」，不默认成强身份。
 */

import { Tag, Tooltip } from 'antd';
import type { Basis } from '../api/types';
import { basisStyle } from '../utils/basis';

interface Props {
  basis: Basis | null | undefined;
  /** 是否显示 tooltip 解释 */
  withTooltip?: boolean;
  size?: 'default' | 'small';
}

export default function BasisTag({ basis, withTooltip = true, size = 'default' }: Props) {
  const s = basisStyle(basis);
  const tag = (
    <Tag
      color={s.color}
      style={{
        marginInlineEnd: 0,
        fontSize: size === 'small' ? 11 : 12,
        lineHeight: size === 'small' ? '16px' : '20px',
        borderStyle: s.key === 'unknown' ? 'dashed' : 'solid',
      }}
    >
      {s.label}
    </Tag>
  );
  if (!withTooltip) return tag;
  return <Tooltip title={s.description}>{tag}</Tooltip>;
}
