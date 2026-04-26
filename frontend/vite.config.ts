import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { VitePWA } from 'vite-plugin-pwa'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'wave-icon.svg'],
      manifest: {
        name: 'WAVE — Event Recommender',
        short_name: 'WAVE',
        description: 'Context-aware event recommendations powered by weather and your preferences.',
        theme_color: '#0096c7',
        background_color: '#03045e',
        display: 'standalone',
        orientation: 'any',
        start_url: '/',
        icons: [
          { src: '/wave-icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/wave-icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' }
        ]
      }
    })
  ],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') }
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, '')
      }
    }
  }
})
