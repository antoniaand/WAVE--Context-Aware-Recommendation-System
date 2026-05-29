import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { OceanBackground } from '@/components/OceanBackground'
import { ThemeToggle } from '@/components/ThemeToggle'
import { WeatherWidget } from '@/components/WeatherWidget'
import { EventCard } from '@/components/EventCard'
import { LocationPicker, type PickedLocation } from '@/components/LocationPicker'
import { useAuth } from '@/hooks/useAuth'
import { recommendService } from '@/services/recommendService'
import type { EventRecommendation, WeatherContext } from '@/types'

/* ── Constants ────────────────────────────────────────────── */

const LOC_KEY = 'wave_location'
const DEFAULT_LOC: PickedLocation = { lat: 44.44, lng: 26.10, displayName: 'București' }

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
    mon.setDate(d.getDate() - ((d.getDay() + 6) % 7))
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

/* ── HomePage ─────────────────────────────────────────────── */

export function HomePage() {
  const { logout, hasProfile, user } = useAuth()
  const navigate = useNavigate()

  const [location, setLocation] = useState<PickedLocation>(() => {
    try {
      const saved = localStorage.getItem(LOC_KEY)
      return saved ? JSON.parse(saved) : DEFAULT_LOC
    } catch {
      return DEFAULT_LOC
    }
  })
  const [showPicker, setShowPicker] = useState(false)
  const [showBanner, setShowBanner] = useState(true)

  const [weather, setWeather] = useState<WeatherContext | null>(null)
  const [todayEvents, setTodayEvents] = useState<EventRecommendation[]>([])
  const [weekEvents, setWeekEvents]   = useState<EventRecommendation[]>([])
  const [monthEvents, setMonthEvents] = useState<EventRecommendation[]>([])

  const [todayLoading, setTodayLoading] = useState(true)
  const [weekLoading,  setWeekLoading]  = useState(true)
  const [monthLoading, setMonthLoading] = useState(true)

  const fetchAll = useCallback(async (loc: PickedLocation) => {
    setTodayLoading(true)
    setWeekLoading(true)
    setMonthLoading(true)

    const uid = user?.user_id ?? undefined
    const base = { lat: loc.lat, lng: loc.lng, display_name: loc.displayName, user_id: uid }

    const [todayRes, weekRes, monthRes] = await Promise.allSettled([
      recommendService.getRecommendations({ ...base, horizon: 'today', top_n: 5  }),
      recommendService.getRecommendations({ ...base, horizon: 'week',  top_n: 15 }),
      recommendService.getRecommendations({ ...base, horizon: 'month', top_n: 15 }),
    ])

    if (todayRes.status === 'fulfilled') {
      setWeather(todayRes.value.weather ?? null)
      setTodayEvents(todayRes.value.recommendations)
    }
    setTodayLoading(false)

    if (weekRes.status === 'fulfilled')  setWeekEvents(weekRes.value.recommendations)
    setWeekLoading(false)

    if (monthRes.status === 'fulfilled') setMonthEvents(monthRes.value.recommendations)
    setMonthLoading(false)
  }, [user?.user_id])

  useEffect(() => {
    localStorage.setItem(LOC_KEY, JSON.stringify(location))
    fetchAll(location)
  }, [location, fetchAll])

  function selectLocation(loc: PickedLocation) {
    setLocation(loc)
    setShowPicker(false)
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

        {/* ── Weather + location picker row ── */}
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
            onClick={() => setShowPicker(true)}
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
              {location.displayName}
            </span>
          </motion.button>
        </div>

        {/* ── Today in {location} ── */}
        <SectionHeader title={`Today in ${location.displayName}`} />
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

      {/* ── Location picker modal ── */}
      <AnimatePresence>
        {showPicker && (
          <LocationPicker
            onLocationSelect={selectLocation}
            onClose={() => setShowPicker(false)}
            initialLat={location.lat}
            initialLng={location.lng}
          />
        )}
      </AnimatePresence>
    </div>
  )
}
