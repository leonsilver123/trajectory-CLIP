/**
 * App —— 路由入口
 *
 * 会话串联方式：search 页确认目标后，跳到
 *   /backtrack?instance_id=<id>&query_id=<qid>
 * 用 URL 查询参数承载（可刷新、可分享、可回退），不依赖内存状态。
 */

import { Navigate, Route, Routes } from 'react-router-dom';
import AppLayout from './components/AppLayout';
import SearchPage from './pages/SearchPage';
import BacktrackPage from './pages/BacktrackPage';
import DashboardPage from './pages/DashboardPage';
import TrajectoryPage from './pages/TrajectoryPage';

export default function App() {
  return (
    <AppLayout>
      <Routes>
        <Route path="/" element={<SearchPage />} />
        <Route path="/backtrack" element={<BacktrackPage />} />
        <Route path="/trajectory" element={<TrajectoryPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppLayout>
  );
}
