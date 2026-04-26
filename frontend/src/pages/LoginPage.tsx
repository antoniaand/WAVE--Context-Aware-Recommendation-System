import { useState, useCallback } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAuth } from '@/hooks/useAuth'
import { OceanBackground } from '@/components/OceanBackground'
import { ThemeToggle } from '@/components/ThemeToggle'

/* ── Inline icons ─────────────────────────────────────────── */
const EyeIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
  </svg>
)
const EyeOffIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/>
    <line x1="1" y1="1" x2="23" y2="23"/>
  </svg>
)
const ArrowRight = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
  </svg>
)

/* ── Stagger animation variants ───────────────────────────── */
const container = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.07, delayChildren: 0.18 } },
}
const item = {
  hidden: { opacity: 0, x: -18 },
  visible: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 320, damping: 24 } },
}

export function LoginPage() {
  const navigate = useNavigate()
  const { login } = useAuth()

  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [showPass, setShowPass] = useState(false)
  const [error, setError]       = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [shake, setShake]       = useState(false)

  const triggerShake = useCallback(() => {
    setShake(true)
    setTimeout(() => setShake(false), 450)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)
    try {
      await login(email, password)
      navigate('/home', { replace: true })
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail
      setError(detail ?? 'Email sau parolă incorecte.')
      triggerShake()
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <motion.div
      className="auth-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.18 }}
    >
      <OceanBackground />
      <ThemeToggle />

      <div className="auth-center">
        <motion.div
          className="auth-card"
          layoutId="auth-card"
          layout="size"
          transition={{ type: 'spring', stiffness: 280, damping: 28 }}
        >
          {/* ── Brand ────────────────────────────────────── */}
          <motion.div
            layoutId="auth-brand"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
          >
            <div className="auth-logo">
              <span className="wave-symbol">≋</span>
              <span className="wave-wordmark">WAVE</span>
            </div>
            <p className="auth-tagline">evenimentele tale, în contextul lor real</p>
          </motion.div>

          {/* ── Tab switcher ──────────────────────────────── */}
          <motion.div
            className="auth-tabs"
            layoutId="auth-tabs"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.13 }}
          >
            <span className="auth-tab auth-tab--active">Conectare</span>
            <Link to="/register" className="auth-tab">Cont nou</Link>
          </motion.div>

          {/* ── Form (fades with route change) ───────────── */}
          <motion.div
            key="login-fields"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.22 }}
          >
            <form
              onSubmit={handleSubmit}
              className={`auth-form${shake ? ' auth-form--shake' : ''}`}
              noValidate
            >
              {/* Error banner */}
              {error && (
                <motion.div
                  className="auth-error"
                  initial={{ opacity: 0, scaleY: 0.85, originY: 0 }}
                  animate={{ opacity: 1, scaleY: 1 }}
                >
                  {error}
                </motion.div>
              )}

              <motion.div
                className="auth-form"
                variants={container}
                initial="hidden"
                animate="visible"
                style={{ gap: 18, display: 'flex', flexDirection: 'column' }}
              >
                {/* Email */}
                <motion.div className="field-group" variants={item}>
                  <label className="field-label" htmlFor="l-email">Email</label>
                  <input
                    id="l-email"
                    type="email"
                    className="field-input"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    placeholder="tu@exemplu.com"
                    required
                    autoComplete="email"
                    autoFocus
                  />
                </motion.div>

                {/* Password */}
                <motion.div className="field-group" variants={item}>
                  <label className="field-label" htmlFor="l-pass">Parolă</label>
                  <div className="field-password-wrap">
                    <input
                      id="l-pass"
                      type={showPass ? 'text' : 'password'}
                      className="field-input"
                      value={password}
                      onChange={e => setPassword(e.target.value)}
                      placeholder="••••••••"
                      required
                      autoComplete="current-password"
                    />
                    <button
                      type="button"
                      className="field-eye"
                      onClick={() => setShowPass(s => !s)}
                      aria-label={showPass ? 'Ascunde parola' : 'Arată parola'}
                    >
                      {showPass ? <EyeOffIcon /> : <EyeIcon />}
                    </button>
                  </div>
                </motion.div>

                {/* Submit */}
                <motion.div variants={item}>
                  <motion.button
                    type="submit"
                    className="btn-ocean"
                    disabled={isLoading}
                    whileHover={{ scale: isLoading ? 1 : 1.025 }}
                    whileTap={{ scale: isLoading ? 1 : 0.965 }}
                    transition={{ type: 'spring', stiffness: 420, damping: 22 }}
                  >
                    {isLoading
                      ? <span className="btn-spinner" />
                      : <><span>Conectează-te</span><ArrowRight /></>
                    }
                  </motion.button>
                </motion.div>
              </motion.div>
            </form>

            <p className="auth-footer">
              Ești nou pe WAVE?{' '}
              <Link to="/register">Creează un cont</Link>
            </p>
          </motion.div>
        </motion.div>
      </div>
    </motion.div>
  )
}
