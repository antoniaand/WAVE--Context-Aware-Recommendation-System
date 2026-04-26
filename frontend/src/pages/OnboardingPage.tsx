import { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { OceanBackground } from '@/components/OceanBackground'
import { ThemeToggle } from '@/components/ThemeToggle'
import { useAuth } from '@/hooks/useAuth'
import type { UserProfile } from '@/types'

/* ── Types ───────────────────────────────────────────────── */
type Draft = Partial<UserProfile>

/* ── Step metadata ───────────────────────────────────────── */
const STEPS = ['Profil', 'Evenimente', 'Vreme', 'Confirmare']

/* ── Slider component ────────────────────────────────────── */
function Slider({ label, value, min = 0, max = 10, onChange }: {
  label: string; value: number; min?: number; max?: number
  onChange: (v: number) => void
}) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span className="field-label" style={{ marginBottom: 0 }}>{label}</span>
        <span style={{ fontSize: '0.8125rem', color: 'var(--accent)', fontWeight: 600 }}>{value}</span>
      </div>
      <input
        type="range" min={min} max={max} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: 2 }}>
        <span>{min}</span><span>{max}</span>
      </div>
    </div>
  )
}

/* ── Chip selector ───────────────────────────────────────── */
function ChipGroup({ options, value, onChange, multi = false }: {
  options: string[]; value: string | string[]
  onChange: (v: string) => void; multi?: boolean
}) {
  const selected = multi ? (value as string).split(',').filter(Boolean) : [value as string]
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
      {options.map(opt => {
        const active = selected.includes(opt)
        return (
          <button
            key={opt} type="button"
            onClick={() => {
              if (multi) {
                const next = active ? selected.filter(s => s !== opt) : [...selected, opt]
                onChange(next.join(','))
              } else {
                onChange(opt)
              }
            }}
            style={{
              padding: '6px 14px', borderRadius: 99, fontSize: '0.8125rem', cursor: 'pointer',
              border: `1.5px solid ${active ? 'var(--accent)' : 'var(--border-input)'}`,
              background: active ? 'rgba(0,180,216,0.15)' : 'var(--bg-input)',
              color: active ? 'var(--accent)' : 'var(--text-secondary)',
              transition: 'all 0.15s',
              fontFamily: 'inherit',
            }}
          >
            {opt}
          </button>
        )
      })}
    </div>
  )
}

/* ── Step panels ─────────────────────────────────────────── */
function StepProfile({ draft, set }: { draft: Draft; set: (k: keyof UserProfile, v: unknown) => void }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="field-group">
        <label className="field-label">Gen</label>
        <ChipGroup options={['F', 'M']} value={draft.gender ?? 'F'} onChange={v => set('gender', v)} />
      </div>
      <div className="field-group">
        <label className="field-label">Grupă de vârstă</label>
        <ChipGroup
          options={['18-24', '25-34', '35-44', '45-54', '55+']}
          value={draft.age_range ?? '25-34'}
          onChange={v => set('age_range', v)}
        />
      </div>
      <div className="field-group">
        <label className="field-label">Cât de des mergi la evenimente?</label>
        <ChipGroup
          options={['Never', 'Rarely', 'Occasionally', 'Often', 'Very often']}
          value={draft.attendance_freq ?? 'Occasionally'}
          onChange={v => set('attendance_freq', v)}
        />
      </div>
    </div>
  )
}

function StepEvents({ draft, set }: { draft: Draft; set: (k: keyof UserProfile, v: unknown) => void }) {
  const EVENT_TYPES = ['Concert', 'Festival', 'Sports', 'Theatre', 'Conference']
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="field-group">
        <label className="field-label">Tipul tău preferat de eveniment</label>
        <ChipGroup
          options={EVENT_TYPES}
          value={draft.top_event ?? 'Concert'}
          onChange={v => set('top_event', v)}
        />
      </div>
      <div className="field-group">
        <label className="field-label">Tipuri de evenimente care te interesează (poți alege mai multe)</label>
        <ChipGroup
          options={EVENT_TYPES}
          value={draft.preferred_event_types ?? 'Concert'}
          onChange={v => set('preferred_event_types', v)}
          multi
        />
      </div>
      <div className="field-group">
        <label className="field-label">Preferință locație</label>
        <ChipGroup
          options={['Interior', 'Exterior']}
          value={draft.indoor_outdoor === 1 ? 'Exterior' : 'Interior'}
          onChange={v => set('indoor_outdoor', v === 'Exterior' ? 1 : 0)}
        />
      </div>
      <div style={{ borderTop: '1px solid var(--border-input)', paddingTop: 16 }}>
        <p className="field-label" style={{ marginBottom: 12 }}>Interes scenariu (0 = deloc, 10 = foarte mult)</p>
        <Slider label="Concert" value={draft.scenario_concert ?? 5} onChange={v => set('scenario_concert', v)} />
        <Slider label="Festival" value={draft.scenario_festival ?? 5} onChange={v => set('scenario_festival', v)} />
        <Slider label="Sport" value={draft.scenario_sports ?? 5} onChange={v => set('scenario_sports', v)} />
        <Slider label="Teatru" value={draft.scenario_theatre ?? 5} onChange={v => set('scenario_theatre', v)} />
        <Slider label="Conferință" value={draft.scenario_conference ?? 5} onChange={v => set('scenario_conference', v)} />
      </div>
    </div>
  )
}

function StepWeather({ draft, set }: { draft: Draft; set: (k: keyof UserProfile, v: unknown) => void }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginBottom: 12 }}>
        Aceste valori ajustează recomandările în funcție de condițiile meteo. (0 = minim, 10 = maxim)
      </p>
      <Slider label="Evit ploaia" value={draft.rain_avoid ?? 5} onChange={v => set('rain_avoid', v)} />
      <Slider label="Toleranță la frig" value={draft.cold_tolerance ?? 5} onChange={v => set('cold_tolerance', v)} />
      <Slider label="Sensibilitate la căldură" value={draft.heat_sensitivity ?? 5} onChange={v => set('heat_sensitivity', v)} />
      <Slider label="Sensibilitate la vânt" value={draft.wind_sensitivity ?? 5} onChange={v => set('wind_sensitivity', v)} />
      <div className="field-group" style={{ marginTop: 8 }}>
        <label className="field-label">Mergi la evenimente indiferent de vreme?</label>
        <ChipGroup
          options={['Nu', 'Da']}
          value={draft.override_weather === 1 ? 'Da' : 'Nu'}
          onChange={v => set('override_weather', v === 'Da' ? 1 : 0)}
        />
      </div>
    </div>
  )
}

function StepConfirm({ draft, email }: { draft: Draft; email: string }) {
  const rows: [string, string][] = [
    ['Email', email],
    ['Gen', draft.gender ?? '—'],
    ['Vârstă', draft.age_range ?? '—'],
    ['Frecvență', draft.attendance_freq ?? '—'],
    ['Eveniment preferat', draft.top_event ?? '—'],
    ['Tipuri alese', draft.preferred_event_types ?? '—'],
    ['Locație', draft.indoor_outdoor === 1 ? 'Exterior' : 'Interior'],
    ['Evit ploaia', String(draft.rain_avoid ?? 5)],
    ['Toleranță frig', String(draft.cold_tolerance ?? 5)],
    ['Sensibilitate căldură', String(draft.heat_sensitivity ?? 5)],
    ['Sensibilitate vânt', String(draft.wind_sensitivity ?? 5)],
    ['Ignoră vremea', draft.override_weather === 1 ? 'Da' : 'Nu'],
  ]
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginBottom: 8 }}>
        Verifică profilul tău înainte de a crea contul.
      </p>
      {rows.map(([label, val]) => (
        <div key={label} style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '7px 0', borderBottom: '1px solid var(--border-input)',
          fontSize: '0.8125rem',
        }}>
          <span style={{ color: 'var(--text-muted)' }}>{label}</span>
          <span style={{ color: 'var(--text-primary)', fontWeight: 500, textAlign: 'right', maxWidth: '55%' }}>{val}</span>
        </div>
      ))}
    </div>
  )
}

/* ── Main page ───────────────────────────────────────────── */
export function OnboardingPage() {
  const location = useLocation()
  const navigate = useNavigate()
  const { register } = useAuth()

  const { email = '', password = '' } = (location.state as { email?: string; password?: string }) ?? {}

  const [step, setStep] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const [draft, setDraft] = useState<Draft>({
    gender: 'F',
    age_range: '25-34',
    attendance_freq: 'Occasionally',
    top_event: 'Concert',
    preferred_event_types: 'Concert',
    indoor_outdoor: 0,
    rain_avoid: 5,
    cold_tolerance: 5,
    heat_sensitivity: 5,
    wind_sensitivity: 5,
    override_weather: 0,
    scenario_concert: 5,
    scenario_festival: 5,
    scenario_sports: 5,
    scenario_theatre: 5,
    scenario_conference: 5,
  })

  const set = (k: keyof UserProfile, v: unknown) =>
    setDraft(d => ({ ...d, [k]: v }))

  const isLast = step === STEPS.length - 1

  const handleNext = async () => {
    setError('')
    if (!isLast) { setStep(s => s + 1); return }

    if (!email || !password) {
      setError('Lipsesc datele de autentificare. Întoarce-te la înregistrare.')
      return
    }
    setIsLoading(true)
    try {
      await register(email, password, draft as UserProfile)
      navigate('/home', { replace: true })
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail
      setError(detail ?? 'Nu s-a putut crea contul. Încearcă din nou.')
    } finally {
      setIsLoading(false)
    }
  }

  const slideVariants = {
    enter: { opacity: 0, x: 30 },
    center: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -30 },
  }

  return (
    <motion.div
      className="auth-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
    >
      <OceanBackground />
      <ThemeToggle />

      <div className="auth-center" style={{ position: 'relative', zIndex: 10, padding: '24px 16px' }}>
        <motion.div
          className="auth-card"
          initial={{ opacity: 0, y: 32, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ type: 'spring', stiffness: 260, damping: 22 }}
          style={{ maxHeight: '90dvh', overflowY: 'auto' }}
        >
          {/* Brand */}
          <div className="auth-logo">
            <span className="wave-symbol">≋</span>
            <span className="wave-wordmark">WAVE</span>
          </div>
          <p className="auth-tagline" style={{ marginBottom: 20 }}>
            Hai să te cunoaștem mai bine{email ? `, ${email.split('@')[0]}` : ''}
          </p>

          {/* Progress bar */}
          <div style={{ display: 'flex', gap: 6, marginBottom: 24 }}>
            {STEPS.map((label, i) => (
              <div key={label} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
                <div style={{
                  height: 4, width: '100%', borderRadius: 99,
                  background: i <= step ? 'var(--accent)' : 'var(--border-input)',
                  transition: 'background 0.3s',
                }} />
                <span style={{
                  fontSize: '0.625rem', color: i === step ? 'var(--accent)' : 'var(--text-muted)',
                  transition: 'color 0.3s', fontWeight: i === step ? 600 : 400,
                }}>
                  {label}
                </span>
              </div>
            ))}
          </div>

          {/* Step content */}
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={step}
              variants={slideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.2 }}
            >
              {step === 0 && <StepProfile draft={draft} set={set} />}
              {step === 1 && <StepEvents draft={draft} set={set} />}
              {step === 2 && <StepWeather draft={draft} set={set} />}
              {step === 3 && <StepConfirm draft={draft} email={email} />}
            </motion.div>
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

          {/* Navigation buttons */}
          <div style={{ display: 'flex', gap: 10, marginTop: 24 }}>
            {step > 0 && (
              <button
                type="button"
                onClick={() => setStep(s => s - 1)}
                style={{
                  flex: 1, padding: '11px', borderRadius: 10, cursor: 'pointer',
                  background: 'var(--bg-input)', border: '1.5px solid var(--border-input)',
                  color: 'var(--text-secondary)', fontFamily: 'inherit', fontSize: '0.9rem',
                }}
              >
                ← Înapoi
              </button>
            )}
            <motion.button
              type="button"
              className="btn-ocean"
              style={{ flex: 2 }}
              disabled={isLoading}
              onClick={handleNext}
              whileHover={{ scale: isLoading ? 1 : 1.025 }}
              whileTap={{ scale: isLoading ? 1 : 0.97 }}
              transition={{ type: 'spring', stiffness: 420, damping: 22 }}
            >
              {isLoading
                ? <span className="btn-spinner" />
                : isLast ? 'Creează contul →' : 'Continuă →'
              }
            </motion.button>
          </div>

          <p className="auth-footer" style={{ marginTop: 14 }}>
            <button
              type="button"
              onClick={() => navigate('/register')}
              style={{ background: 'none', border: 'none', color: 'var(--accent-light)', cursor: 'pointer', fontFamily: 'inherit', fontSize: '0.8125rem' }}
            >
              ← Înapoi la înregistrare
            </button>
          </p>
        </motion.div>
      </div>
    </motion.div>
  )
}
