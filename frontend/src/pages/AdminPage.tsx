import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { OceanBackground } from '@/components/OceanBackground'
import { ThemeToggle } from '@/components/ThemeToggle'
import { useAuth } from '@/hooks/useAuth'

function StubCard({ title, description }: { title: string; description: string }) {
  return (
    <div style={{
      padding: '18px 20px',
      borderRadius: 14,
      border: '1px solid var(--border-input)',
      background: 'var(--bg-input)',
    }}>
      <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>{title}</div>
      <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>{description}</div>
    </div>
  )
}

export function AdminPage() {
  const navigate = useNavigate()
  const { user } = useAuth()

  return (
    <motion.div
      className="auth-page"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    >
      <OceanBackground />
      <ThemeToggle />

      <div className="auth-center" style={{ position: 'relative', zIndex: 10, padding: '24px 16px' }}>
        <motion.div
          className="auth-card"
          initial={{ opacity: 0, y: 28, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ type: 'spring', stiffness: 260, damping: 22 }}
        >
          <div className="auth-logo" style={{ marginBottom: 8 }}>
            <span className="wave-symbol">≋</span>
            <span className="wave-wordmark">WAVE</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 24 }}>
            <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Signed in as</span>
            <span style={{
              display: 'inline-block',
              background: 'rgba(180,0,100,0.2)',
              color: '#f472b6',
              border: '1px solid rgba(180,0,100,0.35)',
              borderRadius: 99,
              padding: '2px 12px',
              fontSize: '0.75rem',
              fontWeight: 700,
              textTransform: 'uppercase',
            }}>
              Admin
            </span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{user?.email}</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 24 }}>
            <StubCard
              title="Model Metrics"
              description="Accuracy, F1-score, and precision/recall per model (LGBM, XGB, RF Strict). Coming in Stage 3."
            />
            <StubCard
              title="User Analytics"
              description="Registration trends, profile completion rate, and recommendation engagement. Coming in Stage 3."
            />
          </div>

          <motion.button
            type="button"
            className="btn-ocean"
            onClick={() => navigate('/home')}
            whileHover={{ scale: 1.025 }}
            whileTap={{ scale: 0.97 }}
            transition={{ type: 'spring', stiffness: 420, damping: 22 }}
          >
            ← Back to home
          </motion.button>
        </motion.div>
      </div>
    </motion.div>
  )
}
