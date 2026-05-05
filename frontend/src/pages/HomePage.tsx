import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { ComposableMap, Geographies, Geography, Marker } from 'react-simple-maps'
import { OceanBackground } from '@/components/OceanBackground'
import { ThemeToggle } from '@/components/ThemeToggle'
import { WeatherWidget } from '@/components/WeatherWidget'
import { EventCard } from '@/components/EventCard'
import { useAuth } from '@/hooks/useAuth'
import { recommendService } from '@/services/recommendService'
import type { EventRecommendation, WeatherContext } from '@/types'

/* ── Constants ────────────────────────────────────────────── */

const GEO_URL = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json'
const CITY_KEY = 'wave_city'

const CITIES: { name: string; coords: [number, number] }[] = [
  { name: 'București',    coords: [26.10, 44.44] },
  { name: 'Cluj-Napoca', coords: [23.60, 46.77] },
  { name: 'Timișoara',   coords: [21.23, 45.75] },
  { name: 'Iași',        coords: [27.60, 47.16] },
  { name: 'Constanța',   coords: [28.65, 44.18] },
  { name: 'Brașov',      coords: [25.61, 45.65] },
  { name: 'Craiova',     coords: [23.80, 44.32] },
  { name: 'Galați',      coords: [28.05, 45.44] },
  { name: 'Oradea',      coords: [21.93, 47.06] },
  { name: 'Ploiești',    coords: [26.02, 44.94] },
  { name: 'Sibiu',       coords: [24.15, 45.80] },
  { name: 'Târgu Mureș', coords: [24.56, 46.54] },
  { name: 'Arad',        coords: [21.32, 46.18] },
  { name: 'Bacău',       coords: [26.92, 46.57] },
  { name: 'Pitești',     coords: [24.87, 44.86] },
]

/* ── Grouping helpers ─────────────────────────────────────── */

function groupByDate(events: EventRecommendation[]): [string, EventRecommendation[]][] {
  const map = new Map<string, EventRecommendation[]>()
  for (const e of events) {
    const key = new Date(e.event_date).toLocaleDateString('en-GB', {
      weekday: 'long', day: 'numeric', month: 'long',
    })
    if (!map.has(key)) map.set(key, [])
    map.get(key)!.push(e)
  }
  return [...map.entries()]
}

function groupByWeek(events: EventRecommendation[]): [string, EventRecommendation[]][] {
  const map = new Map<string, EventRecommendation[]>()
  for (const e of events) {
    const d = new Date(e.event_date)
    const mon = new Date(d)
    mon.setDate(d.getDate() - ((d.getDay() + 6) % 7)) // Monday of that week
    const key = `Week of ${mon.toLocaleDateString('en-GB', { day: 'numeric', month: 'long' })}`
    if (!map.has(key)) map.set(key, [])
    map.get(key)!.push(e)
  }
  return [...map.entries()]
}

/* ── Skeleton card ────────────────────────────────────────── */

function CardSkeleton() {
  return (
    <div style={{
      height: 66,
      borderRadius: 14,
      background: 'var(--bg-input)',
      border: '1px solid var(--border-input)',
      animation: 'pulse 1.5s ease-in-out infinite',
    }} />
  )
}

/* ── Section header ───────────────────────────────────────── */

function SectionHeader({ title }: { title: string }) {
  return (
    <h2 style={{
      fontFamily: '"Syne", sans-serif',
      fontSize: '1rem',
      fontWeight: 700,
      color: 'var(--text-primary)',
      margin: '24px 0 10px',
      letterSpacing: '0.01em',
    }}>
      {title}
    </h2>
  )
}

/* ── Event group ──────────────────────────────────────────── */

function EventGroup({ label, events }: { label: string; events: EventRecommendation[] }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{
        fontSize: '0.75rem',
        fontWeight: 600,
        color: 'var(--text-muted)',
        marginBottom: 8,
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
      }}>
        {label}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {events.map((e, i) => (
          <EventCard key={`${e.event_name ?? e.event_type}-${i}`} event={e} />
        ))}
      </div>
    </div>
  )
}

/* ── City modal ───────────────────────────────────────────── */

function CityModal({
  current,
  onSelect,
  onClose,
}: {
  current: string
  onSelect: (c: string) => void
  onClose: () => void
}) {
  const [search, setSearch] = useState('')
  const [geoLoading, setGeoLoading] = useState(false)

  const filtered = CITIES.filter(c =>
    c.name.toLowerCase().includes(search.toLowerCase())
  )

  function handleGeolocate() {
    if (!navigator.geolocation) return
    setGeoLoading(true)
    navigator.geolocation.getCurrentPosition(
      ({ coords: { latitude: lat, longitude: lon } }) => {
        let best = CITIES[0]
        let bestDist = Infinity
        for (const c of CITIES) {
          const d = Math.hypot(c.coords[0] - lon, c.coords[1] - lat)
          if (d < bestDist) { bestDist = d; best = c }
        }
        setGeoLoading(false)
        onSelect(best.name)
      },
      () => setGeoLoading(false),
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0,
        background: 'rgba(3,4,94,0.72)',
        backdropFilter: 'blur(6px)',
        WebkitBackdropFilter: 'blur(6px)',
        zIndex: 200,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '24px 16px',
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: 24, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 16, scale: 0.97 }}
        transition={{ type: 'spring', stiffness: 280, damping: 24 }}
        onClick={e => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: 560,
          maxHeight: '85dvh', overflowY: 'auto',
          background: 'var(--bg-card)', border: '1px solid var(--border-card)',
          borderRadius: 20, padding: '24px',
          backdropFilter: 'blur(28px)', WebkitBackdropFilter: 'blur(28px)',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <span style={{ fontFamily: '"Syne", sans-serif', fontSize: '1.125rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Select city
          </span>
          <button
            type="button"
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.25rem', padding: '4px', lineHeight: 1 }}
          >
            ✕
          </button>
        </div>

        {/* Geolocation button */}
        <button
          type="button"
          onClick={handleGeolocate}
          disabled={geoLoading}
          style={{
            display: 'flex', alignItems: 'center', gap: 8,
            width: '100%', padding: '10px 14px', borderRadius: 10,
            border: '1px solid var(--border-input)',
            background: 'var(--bg-input)', color: 'var(--text-secondary)',
            fontSize: '0.875rem', cursor: geoLoading ? 'not-allowed' : 'pointer',
            marginBottom: 12, boxSizing: 'border-box',
            opacity: geoLoading ? 0.6 : 1,
          }}
        >
          <span>📍</span>
          {geoLoading ? 'Detecting location…' : 'Use my location'}
        </button>

        {/* Text search */}
        <input
          type="text"
          placeholder="Search city…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{
            width: '100%', padding: '10px 14px', borderRadius: 10,
            border: '1px solid var(--border-input)',
            background: 'var(--bg-input)', color: 'var(--text-primary)',
            fontSize: '0.875rem', outline: 'none',
            marginBottom: 16, boxSizing: 'border-box',
          }}
        />

        {/* SVG map — only shown when not searching */}
        {!search && (
          <div style={{
            borderRadius: 12, overflow: 'hidden',
            marginBottom: 16,
            background: 'rgba(0,80,120,0.12)',
            border: '1px solid var(--border-input)',
          }}>
            <ComposableMap
              projection="geoMercator"
              projectionConfig={{ center: [25, 45.8], scale: 2800 }}
              style={{ width: '100%', height: 220 }}
            >
              <Geographies geography={GEO_URL}>
                {({ geographies }: { geographies: any[] }) =>
                  geographies
                    .filter((g: any) => g.properties.name === 'Romania')
                    .map((geo: any) => (
                      <Geography
                        key={geo.rsmKey}
                        geography={geo}
                        fill="var(--bg-input)"
                        stroke="var(--border-input)"
                        strokeWidth={1}
                        style={{
                          default: { outline: 'none' },
                          hover: { outline: 'none' },
                          pressed: { outline: 'none' },
                        }}
                      />
                    ))
                }
              </Geographies>

              {CITIES.map(c => (
                <Marker
                  key={c.name}
                  coordinates={c.coords}
                  onClick={() => onSelect(c.name)}
                >
                  <circle
                    r={c.name === current ? 7 : 5}
                    fill={c.name === current ? 'var(--accent)' : 'var(--accent-light)'}
                    stroke="#fff"
                    strokeWidth={1.5}
                    style={{ cursor: 'pointer' }}
                  />
                  <text
                    textAnchor="middle"
                    y={-10}
                    style={{
                      fontFamily: 'inherit',
                      fontSize: 7,
                      fill: 'var(--text-muted)',
                      pointerEvents: 'none',
                      userSelect: 'none',
                    }}
                  >
                    {c.name}
                  </text>
                </Marker>
              ))}
            </ComposableMap>
          </div>
        )}

        {/* City list */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {filtered.map(c => (
            <button
              key={c.name}
              type="button"
              onClick={() => onSelect(c.name)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '9px 12px', borderRadius: 9,
                border: c.name === current ? '1px solid var(--accent)' : '1px solid transparent',
                background: c.name === current ? 'rgba(0,150,199,0.1)' : 'transparent',
                color: c.name === current ? 'var(--accent-light)' : 'var(--text-secondary)',
                fontSize: '0.875rem', cursor: 'pointer', textAlign: 'left',
                fontWeight: c.name === current ? 600 : 400,
                width: '100%',
              }}
            >
              {c.name === current && <span style={{ fontSize: '0.75rem' }}>✓</span>}
              {c.name}
            </button>
          ))}
        </div>
      </motion.div>
    </motion.div>
  )
}

/* ── HomePage ─────────────────────────────────────────────── */

export function HomePage() {
  const { logout, hasProfile } = useAuth()
  const navigate = useNavigate()

  const [city, setCity] = useState<string>(
    () => localStorage.getItem(CITY_KEY) ?? 'București'
  )
  const [showCityModal, setShowCityModal] = useState(false)
  const [showBanner, setShowBanner] = useState(true)

  const [weather, setWeather] = useState<WeatherContext | null>(null)
  const [todayEvents, setTodayEvents] = useState<EventRecommendation[]>([])
  const [weekEvents, setWeekEvents] = useState<EventRecommendation[]>([])
  const [monthEvents, setMonthEvents] = useState<EventRecommendation[]>([])

  const [todayLoading, setTodayLoading] = useState(true)
  const [weekLoading,  setWeekLoading]  = useState(true)
  const [monthLoading, setMonthLoading] = useState(true)

  const fetchAll = useCallback(async (c: string) => {
    setTodayLoading(true)
    setWeekLoading(true)
    setMonthLoading(true)

    const [todayRes, weekRes, monthRes] = await Promise.allSettled([
      recommendService.getRecommendations({ city: c, horizon: 'today', top_n: 5 }),
      recommendService.getRecommendations({ city: c, horizon: 'week',  top_n: 15 }),
      recommendService.getRecommendations({ city: c, horizon: 'month', top_n: 15 }),
    ])

    if (todayRes.status === 'fulfilled') {
      setWeather(todayRes.value.weather ?? null)
      setTodayEvents(todayRes.value.recommendations)
    }
    setTodayLoading(false)

    if (weekRes.status === 'fulfilled') setWeekEvents(weekRes.value.recommendations)
    setWeekLoading(false)

    if (monthRes.status === 'fulfilled') setMonthEvents(monthRes.value.recommendations)
    setMonthLoading(false)
  }, [])

  useEffect(() => {
    localStorage.setItem(CITY_KEY, city)
    fetchAll(city)
  }, [city, fetchAll])

  function selectCity(c: string) {
    setCity(c)
    setShowCityModal(false)
  }

  const weekGroups  = groupByDate(weekEvents)
  const monthGroups = groupByWeek(monthEvents)

  return (
    <div style={{ minHeight: '100dvh', position: 'relative', overflowX: 'hidden' }}>
      <OceanBackground />
      <ThemeToggle />

      <div style={{
        position: 'relative',
        zIndex: 10,
        maxWidth: 640,
        margin: '0 auto',
        padding: '0 16px 80px',
      }}>

        {/* ── Top bar ── */}
        <div style={{
          display: 'flex', alignItems: 'center',
          justifyContent: 'space-between',
          paddingTop: 20, paddingBottom: 12,
        }}>
          <div className="auth-logo" style={{ gap: 6 }}>
            <span className="wave-symbol">≋</span>
            <span className="wave-wordmark">WAVE</span>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button
              type="button"
              onClick={() => navigate('/profile')}
              title="Profile"
              style={{
                background: 'var(--bg-card)', border: '1px solid var(--border-card)',
                borderRadius: 10, cursor: 'pointer',
                color: 'var(--text-muted)', fontSize: '1.1rem',
                padding: '6px 10px', lineHeight: 1,
                backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
              }}
            >
              👤
            </button>
            <button
              type="button"
              onClick={() => { logout(); navigate('/login', { replace: true }) }}
              style={{
                background: 'var(--bg-card)', border: '1px solid var(--border-card)',
                borderRadius: 10, cursor: 'pointer',
                color: 'var(--text-muted)', fontSize: '0.8rem',
                padding: '7px 12px', lineHeight: 1,
                backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
              }}
            >
              Sign out
            </button>
          </div>
        </div>

        {/* ── Profile-incomplete banner ── */}
        <AnimatePresence>
          {showBanner && !hasProfile && (
            <motion.div
              initial={{ opacity: 0, height: 0, marginBottom: 0 }}
              animate={{ opacity: 1, height: 'auto', marginBottom: 14 }}
              exit={{ opacity: 0, height: 0, marginBottom: 0 }}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                background: 'rgba(0,119,182,0.18)',
                border: '1px solid rgba(0,150,199,0.35)',
                borderRadius: 12, padding: '10px 14px',
                fontSize: '0.875rem', color: 'var(--text-secondary)',
                overflow: 'hidden',
              }}
            >
              <span>
                Complete your profile for personalised picks.{' '}
                <span
                  onClick={() => navigate('/profile')}
                  style={{ color: 'var(--accent-light)', cursor: 'pointer', fontWeight: 600 }}
                >
                  Set up now →
                </span>
              </span>
              <button
                type="button"
                onClick={() => setShowBanner(false)}
                style={{
                  background: 'none', border: 'none', color: 'var(--text-muted)',
                  cursor: 'pointer', fontSize: '1rem', padding: '0 0 0 10px', lineHeight: 1,
                  flexShrink: 0,
                }}
              >
                ✕
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Weather + city picker row ── */}
        <div style={{ display: 'flex', alignItems: 'stretch', gap: 10, marginBottom: 4 }}>
          <div style={{ flex: 1 }}>
            {weather ? (
              <WeatherWidget weather={weather} />
            ) : (
              <div style={{
                height: 84, borderRadius: 18,
                background: 'var(--bg-input)', border: '1px solid var(--border-input)',
                animation: todayLoading ? 'pulse 1.5s ease-in-out infinite' : 'none',
              }} />
            )}
          </div>

          <motion.button
            type="button"
            onClick={() => setShowCityModal(true)}
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.97 }}
            transition={{ type: 'spring', stiffness: 400, damping: 22 }}
            style={{
              flexShrink: 0,
              background: 'var(--bg-card)', border: '1px solid var(--border-card)',
              borderRadius: 14, padding: '10px 14px',
              cursor: 'pointer',
              backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)',
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', gap: 4,
              minWidth: 72,
            }}
          >
            <span style={{ fontSize: '1.25rem', lineHeight: 1 }}>📍</span>
            <span style={{
              fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600,
              maxWidth: 68, textAlign: 'center',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {city}
            </span>
          </motion.button>
        </div>

        {/* ── Today in {city} ── */}
        <SectionHeader title={`Today in ${city}`} />
        {todayLoading ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[...Array(5)].map((_, i) => <CardSkeleton key={i} />)}
          </div>
        ) : todayEvents.length === 0 ? (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', margin: 0 }}>
            No events found for today.
          </p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {todayEvents.map((e, i) => <EventCard key={`today-${i}`} event={e} />)}
          </div>
        )}

        {/* ── Happening this week ── */}
        <SectionHeader title="Happening this week" />
        {weekLoading ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[...Array(4)].map((_, i) => <CardSkeleton key={i} />)}
          </div>
        ) : weekEvents.length === 0 ? (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', margin: 0 }}>
            No events found for this week.
          </p>
        ) : (
          weekGroups.map(([label, events]) => (
            <EventGroup key={label} label={label} events={events} />
          ))
        )}

        {/* ── Next month ── */}
        <SectionHeader title="Next month" />
        {monthLoading ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[...Array(4)].map((_, i) => <CardSkeleton key={i} />)}
          </div>
        ) : monthEvents.length === 0 ? (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', margin: 0 }}>
            No events found for next month.
          </p>
        ) : (
          monthGroups.map(([label, events]) => (
            <EventGroup key={label} label={label} events={events} />
          ))
        )}
      </div>

      {/* ── City modal ── */}
      <AnimatePresence>
        {showCityModal && (
          <CityModal
            current={city}
            onSelect={selectCity}
            onClose={() => setShowCityModal(false)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}
