import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { OceanBackground } from '@/components/OceanBackground'
import { ThemeToggle } from '@/components/ThemeToggle'

export function NotFoundPage() {
  return (
    <motion.div
      className="auth-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <OceanBackground />
      <ThemeToggle />
      <div style={{ position: 'relative', zIndex: 10, textAlign: 'center' }}>
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: 'spring', stiffness: 260, damping: 22 }}
        >
          <p style={{ fontSize: '5rem', margin: '0 0 8px', lineHeight: 1 }}>404</p>
          <p className="wave-wordmark" style={{ fontSize: '1.5rem', display: 'block', marginBottom: 8 }}>
            Pagina nu există
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: 28 }}>
            S-a rătăcit în adâncuri.
          </p>
          <Link to="/">
            <motion.button
              className="btn-ocean"
              style={{ width: 'auto', paddingInline: 28 }}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
            >
              Înapoi acasă
            </motion.button>
          </Link>
        </motion.div>
      </div>
    </motion.div>
  )
}
