/**
 * Shared survey step components and constants used by OnboardingPage and ProfilePage.
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { SurveyAnswers, ScenarioChoice } from '@/utils/profileEncoding'

/* ── Constants ────────────────────────────────────────────── */

export const STEP_LABELS = ['Profile', 'Weather', 'Scenarios', 'Review'] as const

export const GENDERS   = ['Female', 'Male', 'Non-binary', 'Prefer not to say'] as const
export const AGES      = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'] as const
export const FREQS     = ['Never', '1-2/year', 'Monthly', 'Weekly'] as const
export const EVT_TYPES = ['Concert', 'Festival', 'Sports', 'Theatre', 'Conference', 'Exhibition'] as const

export const SCENARIO_CHOICES: ScenarioChoice[] = ['Would', 'Probably', 'Probably not', 'Would not']

export const SCENARIOS = [
  {
    field:        'scenario_concert'    as const,
    icon:         '🎵',
    weatherIcon:  '🌧️',
    weatherLabel: 'Heavy rain',
    question:     'Heavy rain is forecast for tonight. Would you still go to an outdoor concert?',
  },
  {
    field:        'scenario_festival'   as const,
    icon:         '🎪',
    weatherIcon:  '🌬️',
    weatherLabel: 'Strong winds',
    question:     'Strong winds are expected all day. Would you still attend an outdoor festival?',
  },
  {
    field:        'scenario_sports'     as const,
    icon:         '⚽',
    weatherIcon:  '🌡️',
    weatherLabel: 'Extreme heat',
    question:     'Temperatures above 35 °C are expected. Would you attend an outdoor sports event?',
  },
  {
    field:        'scenario_theatre'    as const,
    icon:         '🎭',
    weatherIcon:  '⛈️',
    weatherLabel: 'Thunderstorm',
    question:     'A thunderstorm is raging outside. Would you still go to a theatre performance?',
  },
  {
    field:        'scenario_conference' as const,
    icon:         '🎤',
    weatherIcon:  '❄️',
    weatherLabel: 'Heavy snow',
    question:     'Heavy snowfall and icy roads. Would you still attend a conference?',
  },
] as const

export const WEATHER_QUESTIONS = [
  { field: 'rain_avoid_likert'        as const, question: 'Rain makes me avoid outdoor events',                   low: 'Never',    high: 'Always'    },
  { field: 'cold_tolerance_likert'    as const, question: 'Cold weather makes me less likely to attend events',    low: 'Not at all', high: 'Very much' },
  { field: 'heat_sensitivity_likert'  as const, question: 'Heat makes me less likely to attend outdoor events',    low: 'Not at all', high: 'Very much' },
  { field: 'wind_sensitivity_likert'  as const, question: 'Strong winds deter me from attending events',           low: 'Not at all', high: 'Very much' },
  { field: 'override_weather_likert'  as const, question: 'I attend events regardless of the weather',             low: 'Never',    high: 'Always'    },
] as const

export const INDOOR_LABELS = ['Fully indoor', 'Mostly indoor', 'Mixed', 'Mostly outdoor', 'Fully outdoor']

export const SCENARIO_DISPLAY: Record<ScenarioChoice, string> = {
  Would:          'Would attend',
  Probably:       'Probably would',
  'Probably not': 'Probably not',
  'Would not':    'Would not attend',
}

export const DEFAULT_ANSWERS: SurveyAnswers = {
  gender:                  'Female',
  age:                     '25-34',
  freq:                    'Monthly',
  preferred_types:         ['Concert'],
  indoor_outdoor_slider:   3,
  top_event:               'Concert',
  rain_avoid_likert:       3,
  cold_tolerance_likert:   3,
  heat_sensitivity_likert: 3,
  wind_sensitivity_likert: 3,
  override_weather_likert: 3,
  scenario_concert:        'Probably',
  scenario_festival:       'Probably',
  scenario_sports:         'Probably',
  scenario_theatre:        'Probably',
  scenario_conference:     'Probably',
}

/* ── Chip ─────────────────────────────────────────────────── */

export function Chip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        padding: '6px 14px',
        borderRadius: 99,
        fontSize: '0.8125rem',
        cursor: 'pointer',
        border: `1.5px solid ${active ? 'var(--accent)' : 'var(--border-input)'}`,
        background: active ? 'rgba(0,150,199,0.18)' : 'var(--bg-input)',
        color: active ? 'var(--accent-light)' : 'var(--text-secondary)',
        fontWeight: active ? 600 : 400,
        transition: 'all 0.15s',
        fontFamily: 'inherit',
      }}
    >
      {label}
    </button>
  )
}

export function ChipGroup({ options, value, onChange }: {
  options: readonly string[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
      {options.map(o => (
        <Chip key={o} label={o} active={value === o} onClick={() => onChange(o)} />
      ))}
    </div>
  )
}

export function MultiChipGroup({ options, value, onChange }: {
  options: readonly string[]
  value: string[]
  onChange: (v: string[]) => void
}) {
  const toggle = (o: string) => {
    const next = value.includes(o) ? value.filter(x => x !== o) : [...value, o]
    if (next.length > 0) onChange(next)
  }
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
      {options.map(o => (
        <Chip key={o} label={o} active={value.includes(o)} onClick={() => toggle(o)} />
      ))}
    </div>
  )
}

/* ── Likert row ───────────────────────────────────────────── */

export function LikertRow({ question, low, high, value, onChange }: {
  question: string
  low: string
  high: string
  value: 1 | 2 | 3 | 4 | 5
  onChange: (v: 1 | 2 | 3 | 4 | 5) => void
}) {
  return (
    <div style={{ marginBottom: 22 }}>
      <p style={{ fontSize: '0.875rem', color: 'var(--text-primary)', lineHeight: 1.45, margin: '0 0 10px' }}>
        {question}
      </p>
      <div style={{ display: 'flex', gap: 6 }}>
        {([1, 2, 3, 4, 5] as const).map(n => (
          <button
            key={n}
            type="button"
            onClick={() => onChange(n)}
            style={{
              flex: 1,
              padding: '9px 0',
              borderRadius: 8,
              fontSize: '0.9375rem',
              fontWeight: 600,
              cursor: 'pointer',
              border: `1.5px solid ${value === n ? 'var(--accent)' : 'var(--border-input)'}`,
              background: value === n ? 'rgba(0,150,199,0.18)' : 'var(--bg-input)',
              color: value === n ? 'var(--accent-light)' : 'var(--text-muted)',
              transition: 'all 0.15s',
              fontFamily: 'inherit',
            }}
          >
            {n}
          </button>
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: 4 }}>
        <span>{low}</span><span>{high}</span>
      </div>
    </div>
  )
}

/* ── Step 1 – Profile ─────────────────────────────────────── */

export function StepProfile({ answers, set }: {
  answers: SurveyAnswers
  set: <K extends keyof SurveyAnswers>(k: K, v: SurveyAnswers[K]) => void
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
      <div className="field-group">
        <label className="field-label">Gender</label>
        <ChipGroup options={GENDERS} value={answers.gender} onChange={v => set('gender', v as SurveyAnswers['gender'])} />
      </div>
      <div className="field-group">
        <label className="field-label">Age range</label>
        <ChipGroup options={AGES} value={answers.age} onChange={v => set('age', v as SurveyAnswers['age'])} />
      </div>
      <div className="field-group">
        <label className="field-label">How often do you attend events?</label>
        <ChipGroup options={FREQS} value={answers.freq} onChange={v => set('freq', v as SurveyAnswers['freq'])} />
      </div>
      <div className="field-group">
        <label className="field-label">Event types you enjoy (select all that apply)</label>
        <MultiChipGroup
          options={EVT_TYPES}
          value={answers.preferred_types}
          onChange={v => set('preferred_types', v as SurveyAnswers['preferred_types'])}
        />
      </div>
      <div className="field-group">
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
          <label className="field-label" style={{ marginBottom: 0 }}>Venue preference</label>
          <span style={{ fontSize: '0.8125rem', color: 'var(--accent)', fontWeight: 600 }}>
            {INDOOR_LABELS[answers.indoor_outdoor_slider - 1]}
          </span>
        </div>
        <input
          type="range" min={1} max={5} value={answers.indoor_outdoor_slider}
          onChange={e => set('indoor_outdoor_slider', Number(e.target.value) as SurveyAnswers['indoor_outdoor_slider'])}
          style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6875rem', color: 'var(--text-muted)', marginTop: 2 }}>
          <span>Indoor</span><span>Outdoor</span>
        </div>
      </div>
      <div className="field-group">
        <label className="field-label">Your favourite type of event</label>
        <ChipGroup options={EVT_TYPES} value={answers.top_event} onChange={v => set('top_event', v as SurveyAnswers['top_event'])} />
      </div>
    </div>
  )
}

/* ── Step 2 – Weather ─────────────────────────────────────── */

export function StepWeather({ answers, set }: {
  answers: SurveyAnswers
  set: <K extends keyof SurveyAnswers>(k: K, v: SurveyAnswers[K]) => void
}) {
  return (
    <div>
      <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginBottom: 22 }}>
        Rate how much weather affects your decision to attend events. (1 = not at all, 5 = a lot)
      </p>
      {WEATHER_QUESTIONS.map(({ field, question, low, high }) => (
        <LikertRow
          key={field}
          question={question}
          low={low}
          high={high}
          value={answers[field]}
          onChange={v => set(field, v)}
        />
      ))}
    </div>
  )
}

/* ── Step 3 – Scenarios ───────────────────────────────────── */

export function StepScenarios({ answers, set, onComplete }: {
  answers: SurveyAnswers
  set: <K extends keyof SurveyAnswers>(k: K, v: SurveyAnswers[K]) => void
  onComplete: () => void
}) {
  const [idx, setIdx] = useState(0)
  const [pending, setPending] = useState(false)

  const scenario = SCENARIOS[idx]
  const currentAnswer = answers[scenario.field]

  const choose = (choice: ScenarioChoice) => {
    if (pending) return
    set(scenario.field, choice)
    setPending(true)
    setTimeout(() => {
      setPending(false)
      if (idx < SCENARIOS.length - 1) {
        setIdx(i => i + 1)
      } else {
        onComplete()
      }
    }, 400)
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 5, marginBottom: 22 }}>
        {SCENARIOS.map((_, i) => (
          <div key={i} style={{
            flex: 1, height: 3, borderRadius: 99,
            background: i <= idx ? 'var(--accent)' : 'var(--border-input)',
            transition: 'background 0.3s',
          }} />
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={idx}
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -30 }}
          transition={{ duration: 0.18 }}
        >
          <div style={{
            background: 'var(--bg-input)',
            border: '1px solid var(--border-input)',
            borderRadius: 16,
            padding: '22px 20px',
            marginBottom: 18,
            textAlign: 'center',
          }}>
            <div style={{ fontSize: '2.25rem', marginBottom: 10, lineHeight: 1 }}>
              {scenario.weatherIcon}&nbsp;{scenario.icon}
            </div>
            <span style={{
              display: 'inline-block',
              background: 'rgba(0,150,199,0.15)',
              color: 'var(--accent-light)',
              borderRadius: 99,
              padding: '3px 12px',
              fontSize: '0.75rem',
              fontWeight: 600,
              marginBottom: 14,
            }}>
              {scenario.weatherLabel}
            </span>
            <p style={{ fontSize: '0.9375rem', color: 'var(--text-primary)', lineHeight: 1.5, margin: 0 }}>
              {scenario.question}
            </p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {SCENARIO_CHOICES.map(choice => {
              const isSelected = currentAnswer === choice
              const isConfirming = isSelected && pending
              return (
                <motion.button
                  key={choice}
                  type="button"
                  onClick={() => choose(choice)}
                  disabled={pending}
                  whileTap={{ scale: pending ? 1 : 0.97 }}
                  style={{
                    padding: '11px 16px',
                    borderRadius: 10,
                    fontSize: '0.9rem',
                    cursor: pending ? 'default' : 'pointer',
                    border: `1.5px solid ${isConfirming ? 'var(--accent)' : isSelected ? 'rgba(0,150,199,0.4)' : 'var(--border-input)'}`,
                    background: isConfirming ? 'rgba(0,150,199,0.22)' : isSelected ? 'rgba(0,150,199,0.08)' : 'var(--bg-input)',
                    color: isSelected ? 'var(--accent-light)' : 'var(--text-secondary)',
                    transition: 'all 0.15s',
                    fontFamily: 'inherit',
                    textAlign: 'left',
                  }}
                >
                  {choice}
                </motion.button>
              )
            })}
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

/* ── Step 4 – Review ──────────────────────────────────────── */

export function StepReview({ answers }: { answers: SurveyAnswers }) {
  const rows: [string, string][] = [
    ['Gender',             answers.gender],
    ['Age range',          answers.age],
    ['Frequency',          answers.freq],
    ['Preferred types',    answers.preferred_types.join(', ')],
    ['Venue preference',   INDOOR_LABELS[answers.indoor_outdoor_slider - 1]],
    ['Favourite event',    answers.top_event],
    ['Rain aversion',      `${answers.rain_avoid_likert} / 5`],
    ['Cold tolerance',     `${answers.cold_tolerance_likert} / 5`],
    ['Heat sensitivity',   `${answers.heat_sensitivity_likert} / 5`],
    ['Wind sensitivity',   `${answers.wind_sensitivity_likert} / 5`],
    ['Weather override',   `${answers.override_weather_likert} / 5`],
    ['Concert in rain',    SCENARIO_DISPLAY[answers.scenario_concert]],
    ['Festival in wind',   SCENARIO_DISPLAY[answers.scenario_festival]],
    ['Sports in heat',     SCENARIO_DISPLAY[answers.scenario_sports]],
    ['Theatre in storm',   SCENARIO_DISPLAY[answers.scenario_theatre]],
    ['Conference in snow', SCENARIO_DISPLAY[answers.scenario_conference]],
  ]
  return (
    <div>
      <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginBottom: 16 }}>
        Review your answers before saving.
      </p>
      {rows.map(([label, val]) => (
        <div key={label} style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          padding: '8px 0', borderBottom: '1px solid var(--border-input)', fontSize: '0.8125rem', gap: 12,
        }}>
          <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{label}</span>
          <span style={{ color: 'var(--text-primary)', fontWeight: 500, textAlign: 'right' }}>{val}</span>
        </div>
      ))}
    </div>
  )
}
