import { motion } from 'framer-motion'
import type { WeatherContext } from '@/types'

/* ── Condition classification ─────────────────────────────── */

type Condition = 'sunny' | 'rainy' | 'snowy' | 'windy' | 'cloudy'

function classify(w: WeatherContext): Condition {
  const p    = w.precip_mm      ?? 0
  const t    = w.temp_C         ?? 10
  const wind = w.wind_speed_kmh ?? 0
  const hum  = w.humidity_pct   ?? 0
  if (p > 2 && t < 2) return 'snowy'
  if (p > 2)          return 'rainy'
  if (wind > 40)      return 'windy'
  if (hum > 80)       return 'cloudy'
  return 'sunny'
}

const CONDITION_LABEL: Record<Condition, string> = {
  sunny:  'Sunny',
  rainy:  'Rainy',
  snowy:  'Snowy',
  windy:  'Windy',
  cloudy: 'Cloudy',
}

/* ── Animated icons ───────────────────────────────────────── */

function SunIcon() {
  const rays = [0, 45, 90, 135, 180, 225, 270, 315]
  return (
    <motion.svg
      width={52} height={52} viewBox="0 0 52 52"
      animate={{ rotate: 360 }}
      transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
    >
      <circle cx={26} cy={26} r={10} fill="#fbbf24" />
      {rays.map(deg => {
        const r = (deg * Math.PI) / 180
        const x1 = 26 + 15 * Math.sin(r), y1 = 26 - 15 * Math.cos(r)
        const x2 = 26 + 22 * Math.sin(r), y2 = 26 - 22 * Math.cos(r)
        return <line key={deg} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#fbbf24" strokeWidth={3} strokeLinecap="round" />
      })}
    </motion.svg>
  )
}

function RainyIcon() {
  return (
    <div style={{ width: 52, height: 52, position: 'relative' }}>
      <div style={{ position: 'absolute', top: 2, left: 2, fontSize: 30 }}>🌥️</div>
      {[0, 1, 2].map(i => (
        <motion.div
          key={i}
          style={{
            position: 'absolute',
            width: 3,
            height: 9,
            background: '#48cae4',
            borderRadius: 99,
            top: 34,
            left: 14 + i * 11,
          }}
          animate={{ y: [0, 14, 14], opacity: [0.9, 0.9, 0] }}
          transition={{ duration: 0.9, repeat: Infinity, delay: i * 0.28, ease: 'linear' }}
        />
      ))}
    </div>
  )
}

function SnowyIcon() {
  return (
    <div style={{ width: 52, height: 52, position: 'relative' }}>
      <div style={{ position: 'absolute', top: 2, left: 2, fontSize: 28 }}>🌥️</div>
      {[0, 1, 2].map(i => (
        <motion.div
          key={i}
          style={{ position: 'absolute', fontSize: 11, top: 32, left: 10 + i * 13, color: '#caf0f8', userSelect: 'none' }}
          animate={{ y: [0, 16, 16], opacity: [1, 1, 0], rotate: [0, 120] }}
          transition={{ duration: 1.3, repeat: Infinity, delay: i * 0.4, ease: 'linear' }}
        >
          ❄
        </motion.div>
      ))}
    </div>
  )
}

function WindyIcon() {
  return (
    <div style={{ width: 56, height: 44, position: 'relative', overflow: 'hidden' }}>
      {[0, 1, 2].map(i => (
        <motion.div
          key={i}
          style={{
            position: 'absolute',
            height: 3,
            borderRadius: 99,
            background: 'var(--accent-light)',
            top: 8 + i * 14,
            width: i === 1 ? 44 : 34,
            left: -56,
          }}
          animate={{ x: [0, 112] }}
          transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.18, ease: 'linear' }}
        />
      ))}
    </div>
  )
}

function CloudyIcon() {
  return (
    <motion.div
      style={{ fontSize: 46, lineHeight: 1 }}
      animate={{ y: [0, -5, 0] }}
      transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
    >
      ☁️
    </motion.div>
  )
}

function WeatherIcon({ condition }: { condition: Condition }) {
  if (condition === 'sunny') return <SunIcon />
  if (condition === 'rainy') return <RainyIcon />
  if (condition === 'snowy') return <SnowyIcon />
  if (condition === 'windy') return <WindyIcon />
  return <CloudyIcon />
}

/* ── WeatherWidget ────────────────────────────────────────── */

export function WeatherWidget({ weather }: { weather: WeatherContext }) {
  const condition = classify(weather)
  const temp      = weather.temp_C         != null ? `${Math.round(weather.temp_C)}°C`       : '—'
  const hum       = weather.humidity_pct   != null ? `${Math.round(weather.humidity_pct)}%`  : '—'
  const wind      = weather.wind_speed_kmh != null ? `${Math.round(weather.wind_speed_kmh)} km/h` : '—'

  return (
    <motion.div
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 20,
        background: 'var(--bg-card)',
        border: '1px solid var(--border-card)',
        borderRadius: 18,
        padding: '16px 22px',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
      }}
    >
      {/* Icon */}
      <div style={{ flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', width: 56 }}>
        <WeatherIcon condition={condition} />
      </div>

      {/* Data */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
          <span style={{ fontFamily: '"Syne", sans-serif', fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1 }}>
            {temp}
          </span>
          <span style={{ display: 'inline-block', background: 'rgba(0,150,199,0.18)', color: 'var(--accent-light)', borderRadius: 99, padding: '2px 10px', fontSize: '0.75rem', fontWeight: 600 }}>
            {CONDITION_LABEL[condition]}
          </span>
        </div>

        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', fontWeight: 600, marginTop: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {weather.city}
        </div>

        <div style={{ display: 'flex', gap: 14, marginTop: 6, fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
          <span>💧 {hum}</span>
          <span>🌬 {wind}</span>
        </div>
      </div>
    </motion.div>
  )
}
