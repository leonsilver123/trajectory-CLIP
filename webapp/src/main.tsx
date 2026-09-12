import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { ConfigProvider, App as AntApp } from 'antd';

import 'antd/dist/reset.css';
// Leaflet 的地图容器与控件样式（缺了会导致瓦片错位、缩放按钮样式丢失）
import 'leaflet/dist/leaflet.css';
import './index.css';

import App from './App';
import { appTheme } from './theme';

const container = document.getElementById('root');
if (!container) {
  throw new Error('找不到 #root 挂载点，请检查 index.html');
}

createRoot(container).render(
  <StrictMode>
    <ConfigProvider theme={appTheme}>
      {/* AntApp 提供 message/modal/notification 的静态上下文 */}
      <AntApp>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </AntApp>
    </ConfigProvider>
  </StrictMode>,
);
