import { useState, useEffect, useRef, useCallback } from 'react'
import { MapContainer, TileLayer, Marker, useMapEvents, useMap } from 'react-leaflet'
import L from 'leaflet'
import { motion } from 'framer-motion'

// Vite bundles break Leaflet's asset auto-detection — point to CDN copies instead
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconUrl:       'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl:     'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

const NOMINATIM = 'https://nominatim.openstreetmap.org'
const UA = 'WaveApp/1.0 (antoniaandreescu2@gmail.com)'

export interface PickedLocation {
  lat: number
  lng: number
  displayName: string
}

interface LocationPickerProps {
  onLocationSelect: (loc: PickedLocation) => void
  onClose: () => void
  initialLat?: number
  initialLng?: number
}

interface NominatimResult {
  lat: string
  lon: string
  display_name: string
  address?: Record<string, string>
}

function shortLabel(data: NominatimResult): string {
  const a = data.address ?? {}
  return (
    a.city ?? a.town ?? a.village ?? a.municipality ?? a.county ??
    data.display_name.split(',')[0]
  )
}

async function reverseGeocode(lat: number, lng: number): Promise<string> {
  try {
    const r = await fetch(
      `${NOMINATIM}/reverse?lat=${lat}&lon=${lng}&format=json&addressdetails=1`,
      { headers: { 'Accept-Language': 'en', 'User-Agent': UA } },
    )
    if (!r.ok) return `${lat.toFixed(3)}, ${lng.toFixed(3)}`
    return shortLabel(await r.json())
  } catch {
    return `${lat.toFixed(3)}, ${lng.toFixed(3)}`
  }
}

async function forwardGeocode(query: string): Promise<NominatimResult[]> {
  try {
    const r = await fetch(
      `${NOMINATIM}/search?q=${encodeURIComponent(query)}&format=json&limit=5&addressdetails=1`,
      { headers: { 'Accept-Language': 'en', 'User-Agent': UA } },
    )
    if (!r.ok) return []
    return r.json()
  } catch {
    return []
  }
}

// ── Inner components (require Leaflet map context) ─────────────────────────────

function MapClickHandler({ onPick }: { onPick: (lat: number, lng: number) => void }) {
  useMapEvents({ click: e => onPick(e.latlng.lat, e.latlng.lng) })
  return null
}

function FlyTo({ lat, lng }: { lat: number; lng: number }) {
  const map = useMap()
  useEffect(() => { map.flyTo([lat, lng], 12, { duration: 1 }) }, [lat, lng, map])
  return null
}

// Tile layout breaks when the map animates into view — invalidate after mount
function InvalidateOnMount() {
  const map = useMap()
  useEffect(() => {
    const t = setTimeout(() => map.invalidateSize(), 150)
    return () => clearTimeout(t)
  }, [map])
  return null
}

// ── Main component ─────────────────────────────────────────────────────────────

export function LocationPicker({
  onLocationSelect,
  onClose,
  initialLat = 44.44,
  initialLng = 26.10,
}: LocationPickerProps) {
  const [pin, setPin]                   = useState<{ lat: number; lng: number } | null>(null)
  const [displayName, setDisplayName]   = useState<string | null>(null)
  const [geocoding, setGeocoding]       = useState(false)
  const [search, setSearch]             = useState('')
  const [suggestions, setSuggestions]   = useState<NominatimResult[]>([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [flyTarget, setFlyTarget]       = useState<{ lat: number; lng: number } | null>(null)
  const [geoLoading, setGeoLoading]     = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Debounced forward geocode
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    if (search.trim().length < 3) { setSuggestions([]); return }
    debounceRef.current = setTimeout(async () => {
      setSearchLoading(true)
      setSuggestions(await forwardGeocode(search))
      setSearchLoading(false)
    }, 400)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [search])

  const pickCoords = useCallback(async (lat: number, lng: number) => {
    setPin({ lat, lng })
    setDisplayName(null)
    setGeocoding(true)
    setDisplayName(await reverseGeocode(lat, lng))
    setGeocoding(false)
  }, [])

  function selectSuggestion(s: NominatimResult) {
    const lat = parseFloat(s.lat)
    const lng = parseFloat(s.lon)
    setSearch('')
    setSuggestions([])
    setFlyTarget({ lat, lng })
    setPin({ lat, lng })
    setDisplayName(shortLabel(s))
  }

  function handleGeolocate() {
    if (!navigator.geolocation) return
    setGeoLoading(true)
    navigator.geolocation.getCurrentPosition(
      async ({ coords: { latitude: lat, longitude: lng } }) => {
        setGeoLoading(false)
        setFlyTarget({ lat, lng })
        await pickCoords(lat, lng)
      },
      () => setGeoLoading(false),
    )
  }

  function handleConfirm() {
    if (!pin || !displayName || geocoding) return
    onLocationSelect({ lat: pin.lat, lng: pin.lng, displayName })
    onClose()
  }

  const canConfirm = !!pin && !!displayName && !geocoding
  const showSuggestions = suggestions.length > 0 || searchLoading

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
          maxHeight: '90dvh', overflowY: 'auto',
          background: 'var(--bg-card)', border: '1px solid var(--border-card)',
          borderRadius: 20, padding: '20px',
          backdropFilter: 'blur(28px)', WebkitBackdropFilter: 'blur(28px)',
          display: 'flex', flexDirection: 'column', gap: 12,
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{
            fontFamily: '"Syne", sans-serif', fontSize: '1.125rem',
            fontWeight: 700, color: 'var(--text-primary)',
          }}>
            Pick a location
          </span>
          <button
            type="button"
            onClick={onClose}
            style={{
              background: 'none', border: 'none', color: 'var(--text-muted)',
              cursor: 'pointer', fontSize: '1.25rem', padding: '4px', lineHeight: 1,
            }}
          >
            ✕
          </button>
        </div>

        {/* Geolocation */}
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
            opacity: geoLoading ? 0.6 : 1, boxSizing: 'border-box',
          }}
        >
          📍 {geoLoading ? 'Detecting location…' : 'Use my current location'}
        </button>

        {/* Search + inline suggestions */}
        <div>
          <input
            type="text"
            placeholder="Search any city or place…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{
              width: '100%', padding: '10px 14px', boxSizing: 'border-box',
              borderRadius: showSuggestions ? '10px 10px 0 0' : '10px',
              border: '1px solid var(--border-input)',
              background: 'var(--bg-input)', color: 'var(--text-primary)',
              fontSize: '0.875rem', outline: 'none',
            }}
          />
          {showSuggestions && (
            <div style={{
              maxHeight: 160, overflowY: 'auto',
              background: 'var(--bg-card)', border: '1px solid var(--border-card)',
              borderTop: 'none', borderRadius: '0 0 10px 10px',
              boxShadow: '0 4px 16px rgba(0,0,0,0.3)',
            }}>
              {searchLoading && (
                <div style={{ padding: '10px 14px', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                  Searching…
                </div>
              )}
              {suggestions.map((s, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => selectSuggestion(s)}
                  style={{
                    display: 'block', width: '100%', textAlign: 'left',
                    padding: '9px 14px', background: 'transparent',
                    border: 'none', borderTop: i > 0 ? '1px solid var(--border-input)' : 'none',
                    color: 'var(--text-secondary)', fontSize: '0.8rem',
                    cursor: 'pointer', whiteSpace: 'nowrap',
                    overflow: 'hidden', textOverflow: 'ellipsis',
                  }}
                >
                  {s.display_name}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Leaflet map */}
        <div style={{
          borderRadius: 12, overflow: 'hidden',
          border: '1px solid var(--border-input)',
          height: 280, flexShrink: 0,
        }}>
          <MapContainer
            center={[initialLat, initialLng]}
            zoom={4}
            style={{ width: '100%', height: '100%' }}
            scrollWheelZoom
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            />
            <InvalidateOnMount />
            <MapClickHandler onPick={pickCoords} />
            {flyTarget && <FlyTo lat={flyTarget.lat} lng={flyTarget.lng} />}
            {pin && <Marker position={[pin.lat, pin.lng]} />}
          </MapContainer>
        </div>

        {/* Selected location preview */}
        {pin && (
          <div style={{
            padding: '10px 14px', borderRadius: 10,
            background: 'rgba(0,150,199,0.08)',
            border: '1px solid rgba(0,150,199,0.3)',
            fontSize: '0.8rem', color: 'var(--text-secondary)',
          }}>
            {geocoding
              ? <span style={{ color: 'var(--text-muted)' }}>Identifying location…</span>
              : (
                <>
                  <span style={{ color: 'var(--text-muted)' }}>Selected: </span>
                  <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{displayName}</span>
                </>
              )
            }
          </div>
        )}

        {/* Confirm */}
        <button
          type="button"
          onClick={handleConfirm}
          disabled={!canConfirm}
          style={{
            width: '100%', padding: '12px',
            borderRadius: 12, border: 'none',
            background: canConfirm ? 'var(--accent)' : 'var(--bg-input)',
            color: canConfirm ? '#fff' : 'var(--text-muted)',
            fontFamily: '"Syne", sans-serif',
            fontWeight: 700, fontSize: '0.925rem',
            cursor: canConfirm ? 'pointer' : 'not-allowed',
            transition: 'background 0.2s, color 0.2s',
          }}
        >
          Confirm location
        </button>
      </motion.div>
    </motion.div>
  )
}
