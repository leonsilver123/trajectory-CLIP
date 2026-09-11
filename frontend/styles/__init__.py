"""
frontend.styles - 公安交通业务平台设计系统

统一视觉规范：颜色 / 字体 / 组件
浅色主题为唯一主题（公安蓝+白底工作区）。
所有 CSS 通过 st.markdown 注入，不依赖外部文件。
"""

# ═══════════════════════════════════════════════
# 1. 颜色体系 — Python 常量
# ═══════════════════════════════════════════════

class Colors:
    """公安交通业务平台标准色板"""
    # 主色系
    PRIMARY_BLUE   = "#0B4EC2"   # 顶部主蓝
    PRIMARY_DARK   = "#083B91"   # 顶部深蓝
    BUTTON_BLUE    = "#1677FF"   # 按钮蓝

    # 背景色
    PAGE_BG        = "#F3F6FA"   # 页面背景
    PANEL_WHITE    = "#FFFFFF"   # 左侧面板 / 卡片背景
    MAP_FILTER_BG  = "#EAF6FF"   # 地图筛选面板
    TABLE_HEADER   = "#E8F1FF"   # 表格表头

    # 边框
    TABLE_BORDER   = "#D9E2EF"   # 表格边框 / 通用边框

    # 文字
    TEXT_PRIMARY   = "#1F2D3D"   # 正文文字
    TEXT_SECONDARY = "#667085"   # 次级文字

    # 语义色
    RED_ALERT      = "#E53935"   # 红色提醒
    ORANGE_WARN    = "#F59E0B"   # 橙色预警
    GREEN_OK       = "#2E7D32"   # 绿色正常

    # 补充
    EXCLUDED_GRAY  = "#9AA4B2"   # 已排除
    MONO_FONT      = "'JetBrains Mono', 'Fira Code', 'Consolas', monospace"


# ═══════════════════════════════════════════════
# 2. 字体规范
# ═══════════════════════════════════════════════

FONT_FAMILY   = '"PingFang SC", "Microsoft YaHei", "Source Han Sans CN", sans-serif'
FONT_MONO     = Colors.MONO_FONT
FONT_SIZE_H1  = "16px"
FONT_SIZE_H2  = "14px"
FONT_SIZE_H3  = "12px"
FONT_SIZE_BODY = "14px"
FONT_SIZE_AUX  = "12px"


# ═══════════════════════════════════════════════
# 3. 浅色主题 CSS（唯一主题）
# ═══════════════════════════════════════════════

LIGHT_THEME_CSS = """
<style>
/* ═══════════════════════════════════════════════════
   公安交通业务平台 — 浅色主题
   公安蓝 #0B4EC2 + 白底工作区
   ═══════════════════════════════════════════════════ */

:root {
    --primary:         #0B4EC2;
    --primary-dark:    #083B91;
    --primary-light:   #E8F1FF;
    --btn-blue:        #1677FF;
    --page-bg:         #F3F6FA;
    --panel-bg:        #FFFFFF;
    --map-filter-bg:   #EAF6FF;
    --table-header:    #E8F1FF;
    --table-border:    #D9E2EF;
    --border:          #D9E2EF;
    --text-primary:    #1F2D3D;
    --text-secondary:  #667085;
    --red-alert:       #E53935;
    --orange-warn:     #F59E0B;
    --green-ok:        #2E7D32;
    --excluded-gray:   #9AA4B2;
    --font-main:       "PingFang SC", "Microsoft YaHei", "Source Han Sans CN", sans-serif;
    --font-mono:       "JetBrains Mono", "Fira Code", "Consolas", monospace;
    --shadow-sm:       0 1px 3px rgba(0, 0, 0, 0.06);
    --shadow-md:       0 2px 8px rgba(0, 0, 0, 0.08);
    --shadow-lg:       0 4px 16px rgba(0, 0, 0, 0.10);
    --radius-sm:       4px;
    --radius-md:       6px;
    --gradient-topbar: linear-gradient(135deg, #0B4EC2 0%, #083B91 100%);
}

/* ── 全局重置 ─────────────────────────────────── */
html, body,
[class*="css"] {
    font-family: var(--font-main) !important;
    font-size: 14px;
    color: var(--text-primary);
}

/* ── Streamlit 主背景 ─────────────────────────── */
.stApp {
    background-color: var(--page-bg) !important;
}

/* ── 主容器 ───────────────────────────────────── */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 8px !important;
    padding-left: 14px !important;
    padding-right: 14px !important;
    max-width: 100% !important;
    min-width: 1280px !important;
}

/* ── 页面切换淡入动效 ─────────────────────────── */
.main .block-container {
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(3px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ════════════════════════════════════════════════
   顶部导航栏
   ════════════════════════════════════════════════ */
.top-title-bar {
    background: var(--gradient-topbar);
    border-bottom: none;
    padding: 10px 24px;
    margin: 0 -14px 10px -14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 8px rgba(11, 78, 194, 0.18);
}
.top-title-bar .title-text {
    color: #FFFFFF;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 3px;
}
.top-title-bar .sub-text {
    color: rgba(255, 255, 255, 0.75);
    font-size: 12px;
}

/* ════════════════════════════════════════════════
   左侧导航面板（Sidebar）
   ════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: var(--panel-bg) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 200px !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
/* 侧边栏 radio 选项 */
[data-testid="stSidebar"] .stRadio label {
    color: var(--text-primary) !important;
    padding: 6px 10px !important;
    border-radius: var(--radius-sm) !important;
    font-size: 13px !important;
    transition: background-color 0.2s, color 0.2s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background-color: var(--primary-light) !important;
    color: var(--primary) !important;
}
/* 选中项：蓝色背景白字 */
[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label {
    background-color: var(--primary) !important;
    color: #FFFFFF !important;
    border-left: 3px solid var(--btn-blue) !important;
}

/* 侧边栏头部 */
.sidebar-header {
    padding: 14px 12px;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin: -16px -16px 10px -16px;
}
.sidebar-header .logo-text {
    color: #0B4EC2;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 2px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── 侧边栏导航按钮统一大小 ─────────────────── */
[data-testid="stSidebar"] .stButton > button {
    min-height: 42px !important;
    height: 42px !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
    margin-bottom: 2px !important;
    display: flex !important;
    align-items: center !important;
    text-align: left !important;
    line-height: 1.3 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stSidebar"] .stButton {
    margin-bottom: 0 !important;
}

/* 侧边栏 expander */
[data-testid="stSidebar"] .streamlit-expanderHeader {
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 6px 8px !important;
    transition: background-color 0.2s !important;
}
[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
    background-color: var(--primary-light) !important;
}

/* ════════════════════════════════════════════════
   标题层级
   ════════════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-family: var(--font-main) !important;
}
h1 { font-size: 16px !important; }
h2 { font-size: 14px !important; }
h3 { font-size: 12px !important; }

/* ════════════════════════════════════════════════
   按钮体系
   ════════════════════════════════════════════════ */
/* 默认辅助按钮：白底蓝边 */
.stButton > button {
    background-color: #FFFFFF !important;
    color: var(--primary) !important;
    border: 1px solid var(--primary) !important;
    border-radius: var(--radius-sm) !important;
    padding: 5px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
    font-family: var(--font-main) !important;
}
.stButton > button:hover {
    background-color: var(--primary) !important;
    color: #FFFFFF !important;
    border-color: var(--primary) !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
/* 主按钮：蓝底白字 */
.stButton > button[kind="primary"] {
    background-color: var(--btn-blue) !important;
    color: #FFFFFF !important;
    border-color: var(--btn-blue) !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
}

/* ════════════════════════════════════════════════
   表单输入
   ════════════════════════════════════════════════ */
.stTextInput input,
.stTextArea textarea,
.stSelectbox select,
.stNumberInput input,
[data-testid="stWidget"] input {
    background-color: #FFFFFF !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--table-border) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 13px !important;
    padding: 5px 10px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    font-family: var(--font-main) !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus,
.stSelectbox select:focus,
.stNumberInput input:focus {
    border-color: var(--btn-blue) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(22, 119, 255, 0.12) !important;
}
.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stNumberInput label {
    color: var(--text-secondary) !important;
    font-size: 12px !important;
}

/* ════════════════════════════════════════════════
   滑块
   ════════════════════════════════════════════════ */
.stSlider label {
    color: var(--text-secondary) !important;
    font-size: 12px !important;
}

/* ════════════════════════════════════════════════
   表格（st.dataframe / st.data_editor）
   ════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--table-border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stDataFrame"] table {
    background-color: #FFFFFF !important;
    color: var(--text-primary) !important;
    font-size: 13px !important;
    font-family: var(--font-main) !important;
}
[data-testid="stDataFrame"] th {
    background-color: var(--table-header) !important;
    color: var(--primary) !important;
    border-bottom: 1px solid var(--table-border) !important;
    padding: 6px 10px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
}
[data-testid="stDataFrame"] td {
    border-bottom: 1px solid #EEF2F7 !important;
    padding: 4px 10px !important;
    line-height: 38px !important;
    font-size: 13px !important;
}
/* 斑马纹 */
[data-testid="stDataFrame"] tr:nth-child(even) {
    background-color: #F8FAFD !important;
}
/* 行悬停 */
[data-testid="stDataFrame"] tr:hover {
    background-color: var(--primary-light) !important;
    transition: background-color 0.15s !important;
}

/* ════════════════════════════════════════════════
   Metric 指标卡
   ════════════════════════════════════════════════ */
[data-testid="stMetricValue"] {
    font-size: 22px !important;
    color: var(--primary) !important;
    font-weight: 700 !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 12px !important;
}
[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    padding: 8px 12px !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ════════════════════════════════════════════════
   卡片组件（自定义 class）
   ════════════════════════════════════════════════ */
.panel-card {
    background-color: #FFFFFF;
    border: 1px solid var(--border);
    padding: 10px 12px;
    box-shadow: var(--shadow-sm);
    border-radius: var(--radius-md);
    transition: transform 0.2s, box-shadow 0.2s;
}
.panel-card:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}
.panel-title {
    color: var(--text-primary);
    font-size: 14px;
    font-weight: 600;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 8px;
}
.panel-value {
    color: var(--primary);
    font-size: 22px;
    font-weight: 700;
    font-family: var(--font-mono);
}
.panel-label {
    color: var(--text-secondary);
    font-size: 12px;
}

/* 搜索结果卡片 */
.result-card {
    background-color: #FFFFFF;
    border: 1px solid var(--border);
    padding: 0;
    box-shadow: var(--shadow-sm);
    border-radius: var(--radius-md);
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
}
.result-card:hover {
    border-color: var(--primary);
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}
.result-card-img {
    width: 100%;
    display: block;
}
.result-card-body {
    padding: 8px 10px;
}
.result-card-title {
    color: var(--text-primary);
    font-size: 13px;
    font-weight: 600;
}
.result-card-attr {
    color: var(--text-secondary);
    font-size: 12px;
    margin-top: 2px;
}
.result-card-score {
    color: var(--btn-blue);
    font-size: 13px;
    font-weight: 600;
    margin-top: 4px;
    font-family: var(--font-mono);
}

/* ════════════════════════════════════════════════
   容器 / container
   ════════════════════════════════════════════════ */
.stContainer, [data-testid="stVerticalBorder"] {
    background-color: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 8px 10px !important;
}

/* ── Divider ──────────────────────────────────── */
hr {
    border-color: var(--border) !important;
    margin: 8px 0 !important;
}

/* ── Caption / 普通文字 ───────────────────────── */
.stCaption, .stMarkdown p {
    color: var(--text-secondary) !important;
    font-size: 12px !important;
}

/* ── Spinner ──────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--primary) !important;
}

/* ════════════════════════════════════════════════
   Alert 提示框
   ════════════════════════════════════════════════ */
.stAlert {
    background-color: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    padding: 8px 14px !important;
    font-size: 13px !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ════════════════════════════════════════════════
   Plotly 图表 / pydeck 地图
   ════════════════════════════════════════════════ */
.js-plotly-plot {
    background-color: #FFFFFF !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stPydeck"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Checkbox ─────────────────────────────────── */
.stCheckbox label {
    color: var(--text-primary) !important;
    font-size: 13px !important;
}

/* ════════════════════════════════════════════════
   Expander 折叠面板
   ════════════════════════════════════════════════ */
.streamlit-expanderHeader {
    color: var(--text-primary) !important;
    font-size: 13px !important;
    transition: background-color 0.2s !important;
}
.streamlit-expanderHeader:hover {
    background-color: var(--primary-light) !important;
}
.streamlit-expanderContent {
    background-color: #FFFFFF !important;
}

/* ════════════════════════════════════════════════
   Tabs 标签页
   ════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background-color: #FFFFFF !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    border-radius: 0 !important;
    font-size: 13px !important;
    padding: 6px 16px !important;
    transition: color 0.2s, border-color 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--primary) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
    font-weight: 600 !important;
}

/* ════════════════════════════════════════════════
   状态标签
   ════════════════════════════════════════════════ */
.status-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 12px;
    font-weight: 500;
    line-height: 18px;
}
.status-tag-pending {
    background-color: #E53935;
    color: #FFFFFF;
}
.status-tag-confirmed {
    background-color: #2E7D32;
    color: #FFFFFF;
}
.status-tag-suspect {
    background-color: #F59E0B;
    color: #FFFFFF;
}
.status-tag-excluded {
    background-color: #E8ECF0;
    color: #9AA4B2;
}
.status-tag-low-conf {
    background-color: transparent;
    color: #E53935;
    border: 1px solid #E53935;
}
.status-tag-exported {
    background-color: #1677FF;
    color: #FFFFFF;
}

/* 状态指示灯 */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}
.status-online  { background-color: var(--green-ok); }
.status-offline { background-color: var(--red-alert); }

/* ════════════════════════════════════════════════
   弹窗 / 对话框
   ════════════════════════════════════════════════ */
[data-testid="stDialog"] {
    background-color: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-lg) !important;
}
[data-testid="stDialog"] > div {
    background-color: #FFFFFF !important;
}

/* ════════════════════════════════════════════════
   分页器
   ════════════════════════════════════════════════ */
[data-testid="stPagination"] {
    background-color: #F0F3F8 !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stPagination"] button {
    color: var(--text-primary) !important;
    font-size: 13px !important;
}
[data-testid="stPagination"] button[aria-current="page"] {
    background-color: var(--btn-blue) !important;
    color: #FFFFFF !important;
}

/* ════════════════════════════════════════════════
   面包屑
   ════════════════════════════════════════════════ */
.breadcrumb {
    color: var(--text-secondary);
    font-size: 12px;
}
.breadcrumb a {
    color: var(--text-secondary);
    text-decoration: none;
}
.breadcrumb a:hover {
    color: var(--primary);
}

/* ════════════════════════════════════════════════
   时间轴节点
   ════════════════════════════════════════════════ */
.timeline-node {
    background-color: #FFFFFF;
    border: 1px solid var(--border);
    padding: 8px;
    text-align: center;
    border-radius: var(--radius-md);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.timeline-node:hover {
    border-color: var(--primary);
    box-shadow: var(--shadow-sm);
}
.timeline-node.selected {
    border-color: var(--primary);
    border-width: 2px;
}
.timeline-node-time {
    color: var(--primary);
    font-size: 12px;
    font-weight: 600;
    font-family: var(--font-mono);
}
.timeline-node-cam {
    color: var(--text-secondary);
    font-size: 11px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ════════════════════════════════════════════════
   功能树节点
   ════════════════════════════════════════════════ */
.nav-tree-item {
    display: block;
    padding: 6px 10px;
    color: var(--text-primary);
    font-size: 13px;
    border-left: 3px solid transparent;
    cursor: pointer;
    text-decoration: none;
    transition: all 0.2s;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}
.nav-tree-item:hover {
    background-color: var(--primary-light);
    color: var(--primary);
}
.nav-tree-item.active {
    background-color: var(--primary-light);
    color: var(--primary);
    border-left-color: var(--primary);
    font-weight: 600;
}

/* ── 导出按钮组 ───────────────────────────────── */
.export-btn-group {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
}

/* ════════════════════════════════════════════════
   隐藏 Streamlit 默认元素
   ════════════════════════════════════════════════ */
footer { visibility: hidden !important; }
#MainMenu { visibility: hidden !important; }
header  { visibility: hidden !important; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebarNavContainer"] { display: none !important; }
nav[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebarNav"] ul,
[data-testid="stSidebarNav"] li { display: none !important; }

/* ════════════════════════════════════════════════
   响应式基础
   ════════════════════════════════════════════════ */
@media (max-width: 1280px) {
    .block-container {
        min-width: 100% !important;
    }
}

/* 侧边栏折叠预留接口 */
.sidebar-collapsed [data-testid="stSidebar"] {
    width: 48px !important;
    min-width: 48px !important;
    overflow: hidden;
}

</style>
"""


# ═══════════════════════════════════════════════
# 4. 主题注入函数
# ═══════════════════════════════════════════════

def inject_theme_css(theme: str | None = None):
    """注入全局浅色主题 CSS。

    Parameters
    ----------
    theme : str, optional
        保留参数以兼容旧调用，始终注入浅色主题。
    """
    import streamlit as st
    st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# 5. 顶部标题栏渲染
# ═══════════════════════════════════════════════

def render_title_bar(title: str = "交通风险感知集成指挥平台",
                     subtitle: str = ""):
    """渲染顶部标题栏 HTML"""
    import streamlit as st
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sub_display = f"{subtitle} &nbsp;|&nbsp; {now}" if subtitle else now
    st.markdown(f"""
    <div class="top-title-bar">
        <div>
            <span class="title-text">{title}</span>
        </div>
        <div class="sub-text">{sub_display}</div>
    </div>
    """, unsafe_allow_html=True)
