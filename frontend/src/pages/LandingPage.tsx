import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { OceanBackground } from '@/components/OceanBackground'
import { ThemeToggle } from '@/components/ThemeToggle'

export function LandingPage() {
  return (
    <motion.div
      className="auth-page"
      style={{ flexDirection: 'column', gap: 0 }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    >
      <OceanBackground />
      <ThemeToggle />

      <div style={{ position: 'relative', zIndex: 10, textAlign: 'center', padding: '0 24px', maxWidth: 560 }}>
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, type: 'spring', stiffness: 260, damping: 22 }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 16 }}>
            <span className="wave-symbol" style={{ fontSize: '3rem' }}>≋</span>
            <span className="wave-wordmark" style={{ fontSize: '3.5rem' }}>WAVE</span>
          </div>
        </motion.div>

        <motion.p
          style={{
            fontFamily: '"Outfit", sans-serif',
            fontSize: 'clamp(1rem, 2.5vw, 1.2rem)',
            color: 'var(--text-secondary)',
            lineHeight: 1.65,
            marginBottom: 40,
          }}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.22 }}
        >
          Recomandări de evenimente în funcție de <strong style={{ color: 'var(--text-primary)' }}>vremea reală</strong> și preferințele tale personale.
        </motion.p>

        <motion.div
          style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.32 }}
        >
          <Link to="/login">
            <motion.button
              className="btn-ocean"
              style={{ width: 'auto', paddingInline: 32 }}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              transition={{ type: 'spring', stiffness: 420, damping: 22 }}
            >
              Conectează-te
            </motion.button>
          </Link>
          <Link to="/register">
            <motion.button
              style={{
                padding: '13px 32px',
                background: 'transparent',
                border: '1px solid var(--border-card)',
                borderRadius: 12,
                color: 'var(--text-primary)',
                fontFamily: '"Outfit", sans-serif',
                fontWeight: 500,
                fontSize: '0.9375rem',
                cursor: 'pointer',
                transition: 'border-color 0.2s',
              }}
              whileHover={{ scale: 1.04, borderColor: 'var(--border-focus)' } as object}
              whileTap={{ scale: 0.97 }}
              transition={{ type: 'spring', stiffness: 420, damping: 22 }}
            >
              Creează cont
            </motion.button>
          </Link>
        </motion.div>
      </div>
    </motion.div>
  )
}
