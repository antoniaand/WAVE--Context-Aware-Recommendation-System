import { useEffect } from 'react'
import { AuthProvider } from '@/contexts/AuthContext'
import { AppRouter } from '@/router/AppRouter'

/* Apply saved theme before first paint (also handled by inline script in index.html) */
function applyTheme() {
  const stored = localStorage.getItem('wave_theme')
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  document.documentElement.setAttribute('data-theme', stored ?? (prefersDark ? 'dark' : 'light'))
}

export default function App() {
  useEffect(() => { applyTheme() }, [])

  return (
    <AuthProvider>
      <AppRouter />
    </AuthProvider>
  )
}
