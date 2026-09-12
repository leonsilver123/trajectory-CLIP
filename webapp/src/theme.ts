/**
 * 深色主题配置
 *
 * 用 antd v5 的 theme.darkAlgorithm，并统一控制大屏观感（顶栏/侧边栏/卡片底色）。
 * 所有颜色集中在这里，页面里不要再散落硬编码色值。
 */

import { theme } from 'antd';
import type { ThemeConfig } from 'antd';

/** 深色底色调色板 */
export const palette = {
  bg: '#0b1220',
  bgElevated: '#141c2b',
  bgHeader: '#111a29',
  border: 'rgba(255,255,255,0.10)',
  text: 'rgba(255,255,255,0.88)',
  textDim: 'rgba(255,255,255,0.55)',
  /** 强身份（与 utils/basis.ts 保持一致） */
  strong: '#52c41a',
  /** 概率推断（与 utils/basis.ts 保持一致） */
  inference: '#faad14',
  accent: '#1668dc',
  danger: '#ff4d4f',
};

export const appTheme: ThemeConfig = {
  algorithm: theme.darkAlgorithm,
  token: {
    colorPrimary: palette.accent,
    colorBgBase: palette.bg,
    borderRadius: 8,
    fontSize: 14,
  },
  components: {
    Layout: {
      headerBg: palette.bgHeader,
      bodyBg: palette.bg,
      siderBg: palette.bgHeader,
    },
    Card: {
      colorBgContainer: palette.bgElevated,
    },
  },
};
