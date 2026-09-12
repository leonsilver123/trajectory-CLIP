/**
 * AppLayout —— 全局布局（顶栏 + 侧边导航 + 内容区）
 *
 * 借鉴 ant-design-pro 的布局观感，但不引入 Umi 框架。
 */

import { useMemo } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Layout, Menu, Typography } from 'antd';
import {
  ApartmentOutlined,
  DashboardOutlined,
  NodeIndexOutlined,
  RadarChartOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import { palette } from '../theme';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const NAV = [
  { key: '/', icon: <SearchOutlined />, label: '查找目标' },
  { key: '/backtrack', icon: <NodeIndexOutlined />, label: '跨镜回溯' },
  { key: '/trajectory', icon: <ApartmentOutlined />, label: '完整轨迹' },
  { key: '/dashboard', icon: <DashboardOutlined />, label: '统计概览' },
];

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation();

  // 选中项：取最长匹配前缀，保证 /backtrack 下高亮「跨镜回溯」
  const selected = useMemo(() => {
    const hit = NAV.filter((n) => n.key !== '/' && pathname.startsWith(n.key));
    return hit.length ? hit[hit.length - 1].key : '/';
  }, [pathname]);

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          paddingInline: 20,
          borderBottom: `1px solid ${palette.border}`,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div
            style={{
              width: 28,
              height: 28,
              borderRadius: 6,
              background: palette.accent,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              fontSize: 15,
            }}
          >
            <RadarChartOutlined />
          </div>
          <Text strong style={{ fontSize: 16 }}>
            交通风险感知子系统
          </Text>
        </div>
      </Header>

      <Layout>
        <Sider
          width={200}
          breakpoint="lg"
          collapsedWidth={64}
          style={{ borderRight: `1px solid ${palette.border}` }}
        >
          <Menu
            mode="inline"
            selectedKeys={[selected]}
            style={{ background: 'transparent', borderInlineEnd: 'none', paddingTop: 8 }}
            items={NAV.map((n) => ({
              key: n.key,
              icon: n.icon,
              label: <Link to={n.key}>{n.label}</Link>,
            }))}
          />
        </Sider>

        <Content style={{ padding: 16, overflow: 'auto' }}>{children}</Content>
      </Layout>
    </Layout>
  );
}
