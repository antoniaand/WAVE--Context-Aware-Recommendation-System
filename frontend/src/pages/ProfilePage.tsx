import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { OceanBackground } from '@/components/OceanBackground'
import { ThemeToggle } from '@/components/ThemeToggle'
import { useAuth } from '@/hooks/useAuth'
import { encodeProfile, decodeProfile, type SurveyAnswers } from '@/utils/profileEncoding'
import {
  STEP_LABELS,
  DEFAULT_ANSWERS,
  StepProfile,
  StepWeather,
  StepScenarios,
  StepReview,
} from '@/components/SurveyForm'

const slide = {
  enter:  (d: number) => ({ opacity: 0, x: d * 40 }),
  center: { opacity: 1, x: 0 },
  exit:   (d: number) => ({ opacity: 0, x: d * -40 }),
}

export function ProfilePage() {
  const { user, updateProfile } = useAuth()

  const initial = user?.profile ? decodeProfile(user.profile) : DEFAULT_ANSWERS

  const [step,      setStep]      = useState(1) // start at step 1 (no hook screen)
  const [dir,       setDir]       = useState(1)
  const [answers,   setAnswers]   = useState<SurveyAnswers>(initial)
  const [isLoading, setIsLoading] = useState(false)
  const [error,     setError]     = useState('')
  const [success,   setSuccess]   = useState(false)

  const set = useCallback(
    <K extends keyof SurveyAnswers>(k: K, v: SurveyAnswers[K]) =>
      setAnswers(a => ({ ...a, [k]: v })),
    [],
  )

  const goTo = (next: number, d: number) => {
    setDir(d)
    setStep(next)
    setError('')
    setSuccess(false)
  }

  const goForward = useCallback(() => {
    setDir(1)
    setStep(s => s + 1)
    setError('')
  }, [])

  const submit = async () => {
    setIsLoading(true)
    setError('')
    setSuccess(false)
    try {
      await updateProfile(encodeProfile(answers))
      setSuccess(true)
    } catch {
      setError('Failed to save your profile. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const isScenarioStep = step === 3
  const isReviewStep   = step === 4
  const stepIdx        = step - 1

  return (
    <motion.div
      className="auth-page"
      style={{ alignItems: 'flex-start', paddingTop: 40 }}
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
          style={{ maxHeight: '90dvh', overflowY: 'auto' }}
        >
          {/* Header */}
          <div className="auth-logo" style={{ marginBottom: 4 }}>
            <span className="wave-symbol">≋</span>
            <span className="wave-wordmark">WAVE</span>
          </div>
          <p className="auth-tagline">Your profile preferences</p>

          {/* Progress bar */}
          <div style={{ display: 'flex', gap: 6, margin: '0 0 24px' }}>
            {STEP_LABELS.map((label, i) => (
              <div key={label} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
                <div style={{
                  height: 4, width: '100%', borderRadius: 99,
                  background: i < stepIdx ? 'var(--accent)' : i === stepIdx ? 'var(--accent-light)' : 'var(--border-input)',
                  transition: 'background 0.3s',
                }} />
                <span style={{ fontSize: '0.625rem', color: i === stepIdx ? 'var(--accent-light)' : 'var(--text-muted)', fontWeight: i === stepIdx ? 600 : 400 }}>
                  {label}
                </span>
              </div>
            ))}
          </div>

          {/* Step content */}
          <AnimatePresence mode="wait" custom={dir}>
            <motion.div
              key={step}
              variants={slide}
              custom={dir}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              {step === 1 && <StepProfile   answers={answers} set={set} />}
              {step === 2 && <StepWeather   answers={answers} set={set} />}
              {step === 3 && <StepScenarios answers={answers} set={set} onComplete={goForward} />}
              {step === 4 && <StepReview    answers={answers} />}
            </motion.div>
          </AnimatePresence>

          {/* Success toast */}
          <AnimatePresence>
            {success && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                style={{
                  marginTop: 16,
                  padding: '10px 14px',
                  background: 'rgba(0,150,80,0.15)',
                  border: '1px solid rgba(0,200,100,0.3)',
                  borderRadius: 10,
                  fontSize: '0.875rem',
                  color: '#4ade80',
                }}
              >
                ✓ Profile saved successfully.
              </motion.div>
            )}
          </AnimatePresence>

          {/* Error */}
          {error && (
            <motion.div
              className="auth-error"
              initial={{ opacity: 0, scaleY: 0.85, originY: 0 }}
              animate={{ opacity: 1, scaleY: 1 }}
              style={{ marginTop: 16 }}
            >
              {error}
            </motion.div>
          )}

          {/* Navigation */}
          {!isScenarioStep && (
            <div style={{ display: 'flex', gap: 10, marginTop: 24 }}>
              {step > 1 && (
                <button
                  type="button"
                  onClick={() => goTo(step - 1, -1)}
                  style={{ flex: 1, padding: '11px', borderRadius: 10, cursor: 'pointer', background: 'var(--bg-input)', border: '1.5px solid var(--border-input)', color: 'var(--text-secondary)', fontFamily: 'inherit', fontSize: '0.9rem' }}
                >
                  ← Back
                </button>
              )}
              <motion.button
                type="button"
                className="btn-ocean"
                style={{ flex: 2 }}
                disabled={isLoading}
                onClick={isReviewStep ? submit : () => goTo(step + 1, 1)}
                whileHover={{ scale: isLoading ? 1 : 1.025 }}
                whileTap={{ scale: isLoading ? 1 : 0.97 }}
                transition={{ type: 'spring', stiffness: 420, damping: 22 }}
              >
                {isLoading ? <span className="btn-spinner" /> : isReviewStep ? 'Save Profile →' : 'Continue →'}
              </motion.button>
            </div>
          )}

          {isScenarioStep && (
            <div style={{ marginTop: 24 }}>
              <button
                type="button"
                onClick={() => goTo(step - 1, -1)}
                style={{ padding: '11px 20px', borderRadius: 10, cursor: 'pointer', background: 'var(--bg-input)', border: '1.5px solid var(--border-input)', color: 'var(--text-secondary)', fontFamily: 'inherit', fontSize: '0.9rem' }}
              >
                ← Back
              </button>
            </div>
          )}
        </motion.div>
      </div>
    </motion.div>
  )
}
