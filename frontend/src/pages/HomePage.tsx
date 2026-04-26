import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAuth } from '@/hooks/useAuth'
import { OceanBackground } from '@/components/OceanBackground'
import { ThemeToggle } from '@/components/ThemeToggle'

export function HomePage() {
  const { user, logout, hasProfile } = useAuth()
  const navigate = useNavigate()

  /* Redirect to onboarding if no profile — guarded auth users land here */
  useEffect(() => {
    if (!hasProfile) navigate('/onboarding', { replace: true })
  }, [hasProfile, navigate])

  return (
    <motion.div
      className="auth-page"
      style={{ flexDirection: 'column', gap: 24, alignItems: 'center' }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    >
      <OceanBackground />
      <ThemeToggle />

      <div style={{ position: 'relative', zIndex: 10, textAlign: 'center' }}>
        <motion.div
          className="auth-card"
          style={{ maxWidth: 480 }}
          initial={{ opacity: 0, y: 28, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ type: 'spring', stiffness: 260, damping: 22 }}
        >
          <div className="auth-logo" style={{ justifyContent: 'center' }}>
            <span className="wave-symbol">≋</span>
            <span className="wave-wordmark">WAVE</span>
          </div>

          <p style={{
            color: 'var(--text-secondary)',
            fontSize: '0.9375rem',
            marginBottom: 4,
          }}>
            Bun venit înapoi,{' '}
            <strong style={{ color: 'var(--text-primary)' }}>
              {user?.email ?? 'utilizator'}
            </strong>
          </p>
          <p style={{
            fontSize: '0.8125rem',
            color: 'var(--text-muted)',
            marginBottom: 32,
            lineHeight: 1.6,
          }}>
            Feed-ul de recomandări context-aware va fi construit în Etapa 3.
          </p>

          {/* Placeholder cards */}
          {['Concert · București · 18°C', 'Festival · Cluj-Napoca · 22°C', 'Theatre · Timișoara · 15°C'].map((label, i) => (
            <motion.div
              key={label}
              style={{
                padding: '14px 18px',
                borderRadius: 12,
                border: '1px solid var(--border-input)',
                marginBottom: 10,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: '0.875rem',
              }}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.15 + i * 0.08, type: 'spring', stiffness: 300, damping: 24 }}
            >
              <span style={{ color: 'var(--text-primary)' }}>{label}</span>
              <span style={{
                fontSize: '0.8125rem',
                fontWeight: 600,
                color: 'var(--accent-light)',
              }}>
                {(0.72 + i * 0.08).toFixed(2)}
              </span>
            </motion.div>
          ))}

          <motion.button
            className="btn-ocean"
            style={{ marginTop: 16 }}
            onClick={() => { logout(); navigate('/login', { replace: true }) }}
            whileHover={{ scale: 1.025 }}
            whileTap={{ scale: 0.97 }}
            transition={{ type: 'spring', stiffness: 420, damping: 22 }}
          >
            Deconectare
          </motion.button>
        </motion.div>
      </div>
    </motion.div>
  )
}
