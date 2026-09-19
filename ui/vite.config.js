import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Build straight into the directory web_gui.py serves from, so there is one
// process to run rather than a dev server alongside the robot server.
export default defineConfig({
  plugins: [react()],
  build: { outDir: '../static', emptyOutDir: true },
  server: {
    // During `npm run dev`, forward API and MJPEG calls to the Python server.
    proxy: {
      '/api': 'http://localhost:8752',
      '/stream': 'http://localhost:8752',
    },
  },
})
