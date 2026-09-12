import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

/**
 * Vite 配置
 *
 * 开发期（npm run dev）通过 proxy 把 /api 与 /static 转发到 FastAPI(:8000)，
 * 这样前端代码里始终用相对路径，不需要在开发/生产之间切换 baseURL。
 * 若前端需要独立部署到别的域名，设置 VITE_API_BASE=http://host:8000 直连后端即可。
 */
const BACKEND = process.env.VITE_BACKEND_ORIGIN ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // REST 接口
      '/api': { target: BACKEND, changeOrigin: true },
      // 后端的 output/ 静态资源（候选图、关键帧）
      '/static': { target: BACKEND, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    // 由 FastAPI 以 StaticFiles 托管，无需调整资源前缀
    chunkSizeWarningLimit: 1600,
  },
})
