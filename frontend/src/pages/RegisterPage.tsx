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

const container = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.065, delayChildren: 0.18 } },
}
const item = {
  hidden: { opacity: 0, x: -18 },
  visible: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 320, damping: 24 } },
}

export function RegisterPage() {
  const navigate = useNavigate()
  const { register } = useAuth()

  const [email, setEmail]         = useState('')
  const [password, setPassword]   = useState('')
  const [confirm, setConfirm]     = useState('')
  const [showPass, setShowPass]   = useState(false)
  const [showConf, setShowConf]   = useState(false)
  const [gdprChecked, setGdprChecked] = useState(false)
  const [academicChecked, setAcademicChecked] = useState(true)
  const [error, setError]         = useState('')
  const [shake, setShake]         = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const triggerShake = useCallback(() => {
    setShake(true)
    setTimeout(() => setShake(false), 450)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!gdprChecked) {
      setError('You must accept the Terms & Conditions to continue.')
      triggerShake()
      return
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters.')
      triggerShake()
      return
    }
    if (password !== confirm) {
      setError('Passwords do not match.')
      triggerShake()
      return
    }

    setIsLoading(true)
    try {
      // Register immediately — no profile required. Profile is set in onboarding.
      await register(email, password)
      navigate('/onboarding', { replace: false })
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail
      setError(detail ?? 'Could not create account. Please try again.')
      triggerShake()
    } finally {
      setIsLoading(false)
    }
  }

  const strength = (() => {
    if (!password) return 0
    let s = 0
    if (password.length >= 6)  s++
    if (password.length >= 10) s++
    if (/[A-Z]/.test(password)) s++
    if (/[0-9!@#$%^&*]/.test(password)) s++
    return s
  })()

  const strengthLabel = ['', 'Weak', 'Fair', 'Good', 'Strong'][strength]
  const strengthColor = ['', '#ef4444', '#f59e0b', '#0096c7', '#22c55e'][strength]

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
          <motion.div layoutId="auth-brand">
            <div className="auth-logo">
              <span className="wave-symbol">≋</span>
              <span className="wave-wordmark">WAVE</span>
            </div>
            <p className="auth-tagline">your events, in their real context</p>
          </motion.div>

          {/* ── Tab switcher ──────────────────────────────── */}
          <motion.div className="auth-tabs" layoutId="auth-tabs">
            <Link to="/login" className="auth-tab">Sign In</Link>
            <span className="auth-tab auth-tab--active">New account</span>
          </motion.div>

          {/* ── Form ─────────────────────────────────────── */}
          <motion.div
            key="register-fields"
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
                style={{ gap: 16, display: 'flex', flexDirection: 'column' }}
                variants={container}
                initial="hidden"
                animate="visible"
              >
                {/* Email */}
                <motion.div className="field-group" variants={item}>
                  <label className="field-label" htmlFor="r-email">Email</label>
                  <input
                    id="r-email"
                    type="email"
                    className="field-input"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    required
                    autoComplete="email"
                    autoFocus
                  />
                </motion.div>

                {/* Password */}
                <motion.div className="field-group" variants={item}>
                  <label className="field-label" htmlFor="r-pass">Password</label>
                  <div className="field-password-wrap">
                    <input
                      id="r-pass"
                      type={showPass ? 'text' : 'password'}
                      className="field-input"
                      value={password}
                      onChange={e => setPassword(e.target.value)}
                      placeholder="Min. 6 characters"
                      required
                      autoComplete="new-password"
                    />
                    <button type="button" className="field-eye" onClick={() => setShowPass(s => !s)} aria-label="Toggle password">
                      {showPass ? <EyeOffIcon /> : <EyeIcon />}
                    </button>
                  </div>
                  {password.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      style={{ marginTop: 6 }}
                    >
                      <div style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
                        {[1,2,3,4].map(i => (
                          <div key={i} style={{
                            flex: 1, height: 3, borderRadius: 99,
                            background: i <= strength ? strengthColor : 'var(--border-input)',
                            transition: 'background 0.3s',
                          }} />
                        ))}
                      </div>
                      <span style={{ fontSize: '0.75rem', color: strengthColor, transition: 'color 0.3s' }}>
                        {strengthLabel}
                      </span>
                    </motion.div>
                  )}
                </motion.div>

                {/* Confirm password */}
                <motion.div className="field-group" variants={item}>
                  <label className="field-label" htmlFor="r-conf">Confirm password</label>
                  <div className="field-password-wrap">
                    <input
                      id="r-conf"
                      type={showConf ? 'text' : 'password'}
                      className="field-input"
                      value={confirm}
                      onChange={e => setConfirm(e.target.value)}
                      placeholder="Repeat password"
                      required
                      autoComplete="new-password"
                      style={confirm.length > 0 && confirm !== password ? { borderColor: 'rgba(239,68,68,0.5)' } : {}}
                    />
                    <button type="button" className="field-eye" onClick={() => setShowConf(s => !s)} aria-label="Toggle confirm">
                      {showConf ? <EyeOffIcon /> : <EyeIcon />}
                    </button>
                  </div>
                  {confirm.length > 0 && confirm === password && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      style={{ fontSize: '0.75rem', color: '#22c55e', marginTop: 4 }}
                    >
                      ✓ Passwords match
                    </motion.span>
                  )}
                </motion.div>

                {/* GDPR consent */}
                <motion.div variants={item} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <label style={{ display: 'flex', gap: 10, alignItems: 'flex-start', cursor: 'pointer', fontSize: '0.8125rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    <input
                      type="checkbox"
                      checked={gdprChecked}
                      onChange={e => setGdprChecked(e.target.checked)}
                      style={{ marginTop: 2, accentColor: 'var(--accent)', flexShrink: 0 }}
                    />
                    I have read and agree to the{' '}
                    <span style={{ color: 'var(--accent-light)', textDecoration: 'underline', cursor: 'pointer' }}>Terms & Conditions</span>
                    {' '}and{' '}
                    <span style={{ color: 'var(--accent-light)', textDecoration: 'underline', cursor: 'pointer' }}>Privacy Policy</span>.
                  </label>
                  <label style={{ display: 'flex', gap: 10, alignItems: 'flex-start', cursor: 'pointer', fontSize: '0.8125rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                    <input
                      type="checkbox"
                      checked={academicChecked}
                      onChange={e => setAcademicChecked(e.target.checked)}
                      style={{ marginTop: 2, accentColor: 'var(--accent)', flexShrink: 0 }}
                    />
                    I consent to the anonymous use of my responses for academic research purposes.
                  </label>
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
                      : <><span>Continue</span><ArrowRight /></>
                    }
                  </motion.button>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 10, textAlign: 'center' }}>
                    You can set your event preferences in the next step
                  </p>
                </motion.div>
              </motion.div>
            </form>

            <p className="auth-footer">
              Already have an account?{' '}
              <Link to="/login">Sign in</Link>
            </p>
          </motion.div>
        </motion.div>
      </div>
    </motion.div>
  )
}
