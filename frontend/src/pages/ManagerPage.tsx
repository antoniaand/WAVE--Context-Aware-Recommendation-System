import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { OceanBackground } from '@/components/OceanBackground'
import { ThemeToggle } from '@/components/ThemeToggle'
import { useAuth } from '@/hooks/useAuth'

export function ManagerPage() {
  const navigate = useNavigate()
  const { user, role } = useAuth()

  const [form, setForm] = useState({ name: '', type: '', location: '', date: '', description: '' })

  return (
    <motion.div
      className="auth-page"
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
        >
          <div className="auth-logo" style={{ marginBottom: 8 }}>
            <span className="wave-symbol">≋</span>
            <span className="wave-wordmark">WAVE</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 24 }}>
            <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Signed in as</span>
            <span style={{
              display: 'inline-block',
              background: 'rgba(0,120,60,0.2)',
              color: '#4ade80',
              border: '1px solid rgba(0,150,80,0.35)',
              borderRadius: 99,
              padding: '2px 12px',
              fontSize: '0.75rem',
              fontWeight: 700,
              textTransform: 'uppercase',
            }}>
              {role === 'admin' ? 'Admin' : 'Event Manager'}
            </span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{user?.email}</span>
          </div>

          {/* Post Event stub */}
          <div style={{ marginBottom: 24 }}>
            <p className="field-label" style={{ marginBottom: 16 }}>Post an Event</p>
            <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginBottom: 16 }}>
              This form will submit new events to the Supabase <code style={{ background: 'var(--bg-input)', padding: '1px 6px', borderRadius: 4 }}>events</code> table for inclusion in recommendations.
              Full implementation coming in Stage 3.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {([
                ['Event name',   'name',        'text',   'e.g. Untold Festival'],
                ['Type',         'type',        'text',   'Concert / Festival / …'],
                ['Location',     'location',    'text',   'City'],
                ['Date',         'date',        'date',   ''],
                ['Description',  'description', 'text',   'Short description'],
              ] as const).map(([label, key, inputType, placeholder]) => (
                <div key={key} className="field-group">
                  <label className="field-label">{label}</label>
                  <input
                    type={inputType}
                    placeholder={placeholder}
                    className="field-input"
                    value={form[key]}
                    onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
                    disabled
                  />
                </div>
              ))}
              <button
                type="button"
                className="btn-ocean"
                disabled
                style={{ opacity: 0.5, cursor: 'not-allowed', marginTop: 4 }}
              >
                Submit Event (coming soon)
              </button>
            </div>
          </div>

          <motion.button
            type="button"
            onClick={() => navigate('/home')}
            whileHover={{ scale: 1.025 }}
            whileTap={{ scale: 0.97 }}
            transition={{ type: 'spring', stiffness: 420, damping: 22 }}
            style={{
              width: '100%',
              padding: '11px',
              borderRadius: 10,
              cursor: 'pointer',
              background: 'var(--bg-input)',
              border: '1.5px solid var(--border-input)',
              color: 'var(--text-secondary)',
              fontFamily: 'inherit',
              fontSize: '0.9rem',
            }}
          >
            ← Back to home
          </motion.button>
        </motion.div>
      </div>
    </motion.div>
  )
}
