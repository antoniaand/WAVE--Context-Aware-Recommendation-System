import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { EventRecommendation } from '@/types'

/* ── Helpers ──────────────────────────────────────────────── */

function Badge({ label, color }: { label: string; color: string }) {
  return (
    <span style={{
      display: 'inline-block',
      padding: '2px 9px',
      borderRadius: 99,
      fontSize: '0.7rem',
      fontWeight: 700,
      letterSpacing: '0.04em',
      background: color,
      color: '#fff',
      textTransform: 'uppercase',
    }}>
      {label}
    </span>
  )
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })
  } catch {
    return iso
  }
}

/* ── Detail modal ─────────────────────────────────────────── */

function EventModal({ event, onClose }: { event: EventRecommendation; onClose: () => void }) {
  const matchPct = Math.round(event.attended_prob * 100)
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(3,4,94,0.72)',
        backdropFilter: 'blur(6px)',
        WebkitBackdropFilter: 'blur(6px)',
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
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
          width: '100%',
          maxWidth: 480,
          maxHeight: '85dvh',
          overflowY: 'auto',
          background: 'var(--bg-card)',
          border: '1px solid var(--border-card)',
          borderRadius: 20,
          padding: '24px',
          backdropFilter: 'blur(28px)',
          WebkitBackdropFilter: 'blur(28px)',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
          <div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
              <Badge label={`${matchPct}% match`} color={matchPct >= 70 ? '#0096c7' : matchPct >= 50 ? '#0077b6' : '#023e8a'} />
              {event.is_generated && <Badge label="AI generated" color="rgba(120,80,180,0.85)" />}
              {event.source && event.source !== 'generated' && <Badge label={event.source} color="rgba(0,80,40,0.85)" />}
            </div>
            <h3 style={{ fontFamily: '"Syne", sans-serif', fontSize: '1.125rem', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
              {event.event_name ?? event.event_type}
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.25rem', padding: '4px', lineHeight: 1, flexShrink: 0 }}
          >
            ✕
          </button>
        </div>

        {/* Image */}
        {event.image_url && (
          <img
            src={event.image_url}
            alt={event.event_name ?? event.event_type}
            style={{ width: '100%', height: 160, objectFit: 'cover', borderRadius: 12, marginBottom: 16 }}
          />
        )}

        {/* Details */}
        {[
          ['Type',        event.event_type],
          ['Date',        formatDate(event.event_date)],
          ['Location',    event.location],
          ['Venue',       event.venue],
          ['Environment', event.is_outdoor ? 'Outdoor' : 'Indoor'],
        ].filter(([, v]) => v != null).map(([label, val]) => (
          <div key={label as string} style={{ display: 'flex', gap: 12, padding: '8px 0', borderBottom: '1px solid var(--border-input)', fontSize: '0.875rem' }}>
            <span style={{ color: 'var(--text-muted)', width: 90, flexShrink: 0 }}>{label}</span>
            <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{val as string}</span>
          </div>
        ))}

        {/* Description */}
        {event.description && (
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginTop: 14, marginBottom: 0 }}>
            {event.description}
          </p>
        )}

        {/* Link */}
        {event.url && (
          <a
            href={event.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{ display: 'block', marginTop: 16 }}
          >
            <motion.div
              className="btn-ocean"
              whileHover={{ scale: 1.025 }}
              whileTap={{ scale: 0.97 }}
              transition={{ type: 'spring', stiffness: 420, damping: 22 }}
              style={{ textAlign: 'center', textDecoration: 'none', color: '#fff', fontSize: '0.9rem' }}
            >
              View event →
            </motion.div>
          </a>
        )}
      </motion.div>
    </motion.div>
  )
}

/* ── EventCard ────────────────────────────────────────────── */

export function EventCard({ event }: { event: EventRecommendation }) {
  const [open, setOpen] = useState(false)
  const matchPct = Math.round(event.attended_prob * 100)

  return (
    <>
      <motion.div
        onClick={() => setOpen(true)}
        whileHover={{ scale: 1.015 }}
        whileTap={{ scale: 0.98 }}
        transition={{ type: 'spring', stiffness: 400, damping: 24 }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 14,
          padding: '13px 16px',
          borderRadius: 14,
          border: '1px solid var(--border-input)',
          background: 'var(--bg-input)',
          cursor: 'pointer',
          transition: 'border-color 0.15s',
        }}
      >
        {/* Match % circle */}
        <div style={{
          width: 46,
          height: 46,
          borderRadius: '50%',
          border: `2.5px solid ${matchPct >= 70 ? 'var(--accent)' : matchPct >= 50 ? '#0077b6' : 'var(--border-input)'}`,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          background: matchPct >= 70 ? 'rgba(0,150,199,0.12)' : 'transparent',
        }}>
          <span style={{ fontSize: '0.875rem', fontWeight: 700, color: matchPct >= 70 ? 'var(--accent-light)' : 'var(--text-muted)', lineHeight: 1 }}>
            {matchPct}
          </span>
          <span style={{ fontSize: '0.5rem', color: 'var(--text-muted)', letterSpacing: '0.05em', lineHeight: 1, marginTop: 1 }}>%</span>
        </div>

        {/* Info */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {event.event_name ?? event.event_type}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 2 }}>
            {event.event_type} · {event.location} · {formatDate(event.event_date)}
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 5, flexWrap: 'wrap' }}>
            {event.is_generated && <Badge label="Generated" color="rgba(120,80,180,0.75)" />}
            {event.source && event.source !== 'generated' && <Badge label={event.source} color="rgba(0,100,60,0.8)" />}
          </div>
        </div>

        <span style={{ color: 'var(--text-muted)', fontSize: '1rem', flexShrink: 0 }}>›</span>
      </motion.div>

      <AnimatePresence>
        {open && <EventModal event={event} onClose={() => setOpen(false)} />}
      </AnimatePresence>
    </>
  )
}
