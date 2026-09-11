"""
frontend.app - Streamlit 主入口

交通风险感知集成指挥平台 — 公安业务平台标准布局：
  顶部蓝色导航栏 + 左侧图标导航 + 主工作区
启动方式: streamlit run frontend/app.py --server.port 8501
"""

from __future__ import annotations

import sys
import os

# 禁用 Streamlit 文件监控，避免与 PyTorch 的路径检查冲突导致 RuntimeError
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# 确保项目根目录在 Python 路径中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st

from frontend.styles import inject_theme_css
from frontend.components import render_icon

# ============================================================
# 页面配置（必须是第一个 Streamlit 命令）
# ============================================================

st.set_page_config(
    page_title="交通风险感知集成指挥平台",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 注入全局主题 CSS
# ============================================================

inject_theme_css()

# ============================================================
# 导航项定义
# ============================================================

_NAV_ITEMS = [
    {"label": "指挥工作台", "page": "home"},
    {"label": "目标检索",   "page": "search"},
    {"label": "目标确认",   "page": "confirm"},
    {"label": "轨迹回溯",   "page": "trajectory"},
    {"label": "时间轴",     "page": "timeline"},
    {"label": "数据管理",   "page": "data"},
]

_PAGE_TITLES = {item["page"]: item["label"] for item in _NAV_ITEMS}
_ALL_PAGES = [item["page"] for item in _NAV_ITEMS]

# ============================================================
# 侧边栏导航 CSS
# 将 st.radio 彻底重样式化为导航菜单
# ============================================================

_SIDEBAR_NAV_CSS = """
<style>
/* ── 侧边栏头部 ─────────────────────────────── */
.sidebar-header {
    padding: 10px 8px 6px;
}
.sidebar-header .logo-text {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 15px;
    font-weight: 700;
    color: #0B4EC2;
    letter-spacing: 1px;
}

/* ── 导航分区标题 ─────────────────────────── */
.sb-nav-title {
    font-size: 11px;
    font-weight: 600;
    color: #667085;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 4px 0 2px;
    margin-top: 2px;
}

/* ── 侧边栏辅助样式 ─────────────────────────── */
.sb-divider {
    border-top: 1px solid #D9E2EF;
    margin: 10px 8px;
}
</style>
"""

# ============================================================
# 顶部导航栏 CSS
# ============================================================

_TOPBAR_CSS = """
<style>
.top-navbar {
    background: linear-gradient(135deg, #0B4EC2 0%, #083B91 100%);
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    margin: 0 -14px 10px -14px;
    box-shadow: 0 2px 8px rgba(11, 78, 194, 0.18);
}
.top-navbar-left {
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 2px;
    white-space: nowrap;
}
.top-navbar-right {
    display: flex;
    align-items: center;
    gap: 14px;
    color: rgba(255, 255, 255, 0.85);
    font-size: 12px;
}
.top-navbar-right .nb-sep {
    color: rgba(255, 255, 255, 0.35);
}
.top-navbar-right .nb-icon {
    cursor: pointer;
    opacity: 0.8;
    transition: opacity 0.2s;
    display: inline-flex;
    align-items: center;
}
.top-navbar-right .nb-icon:hover {
    opacity: 1;
}
</style>
"""

# ============================================================
# 面包屑 CSS
# ============================================================

_BREADCRUMB_CSS = """
<style>
.nb-breadcrumb {
    color: #667085;
    font-size: 12px;
    padding: 4px 0 8px 0;
    margin-bottom: 4px;
}
</style>
"""


# ============================================================
# 顶部导航栏渲染
# ============================================================

def _render_top_navbar():
    """渲染顶部蓝色导航栏"""
    icon_bell = render_icon("bell", size=14, color="#FFFFFF")
    icon_settings = render_icon("settings", size=14, color="#FFFFFF")
    icon_logout = render_icon("log-out", size=14, color="#FFFFFF")

    st.markdown(_TOPBAR_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="top-navbar">
        <div class="top-navbar-left">交通风险感知集成指挥平台</div>
        <div class="top-navbar-right">
            <span>管理员</span>
            <span class="nb-icon" title="消息通知">{icon_bell}</span>
            <span class="nb-icon" title="系统设置">{icon_settings}</span>
            <span class="nb-icon" title="退出登录">{icon_logout}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 侧边栏导航渲染
# ============================================================

def _render_sidebar_nav() -> str:
    """渲染侧边栏导航菜单，返回当前选中的页面 key。

    使用 st.button 管理导航（每个按钮直接切换页面），
    无 radio 文本框、无点击区域偏移问题。
    """
    current = st.session_state.get("current_page", "home")
    if current not in _ALL_PAGES:
        current = "home"

    st.markdown(_SIDEBAR_NAV_CSS, unsafe_allow_html=True)

    with st.sidebar:
        # ── 侧边栏头部 ──
        icon_shield = render_icon("shield-check", size=16, color="#0B4EC2")
        st.markdown(
            f'<div class="sidebar-header">'
            f'<div class="logo-text">{icon_shield} 集成指挥平台</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sb-nav-title">导航菜单</div>',
            unsafe_allow_html=True,
        )

        # ── 导航按钮 ──
        for item in _NAV_ITEMS:
            is_active = (item["page"] == current)
            btn_type = "primary" if is_active else "secondary"
            btn_label = f"▸ {item['label']}" if is_active else f"  {item['label']}"
            if st.button(
                btn_label,
                key=f"sb_nav_{item['page']}",
                use_container_width=True,
                type=btn_type,
            ):
                if not is_active:
                    st.session_state["current_page"] = item["page"]
                    st.rerun()

    return current


# ============================================================
# 面包屑渲染
# ============================================================

def _render_breadcrumb(page: str):
    """渲染面包屑导航"""
    title = _PAGE_TITLES.get(page, "")
    st.markdown(_BREADCRUMB_CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="nb-breadcrumb">指挥工作台 &gt; {title}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 主入口
# ============================================================

def main():
    """主入口函数"""
    # 初始化 session state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "home"

    # ── 顶部蓝色导航栏 ──
    _render_top_navbar()

    # ── 侧边栏导航 ──
    page = _render_sidebar_nav()

    # ── 面包屑 ──
    _render_breadcrumb(page)

    # ── 路由到对应页面 ──
    if page == "home":
        from frontend.home import render as render_home
        render_home()

    elif page == "search":
        from frontend.pages.search import render as render_search
        render_search()

    elif page == "confirm":
        from frontend.pages.confirm import render as render_confirm
        render_confirm()

    elif page == "trajectory":
        from frontend.pages.trajectory import render as render_trajectory
        render_trajectory()

    elif page == "timeline":
        from frontend.pages.timeline import render as render_timeline
        render_timeline()

    elif page == "data" or page == "dashboard" or page == "export":
        from frontend.pages.dashboard import render as render_dashboard
        render_dashboard()


if __name__ == "__main__":
    main()
else:
    # streamlit run 时直接执行
    main()
