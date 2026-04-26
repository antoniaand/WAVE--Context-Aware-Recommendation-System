/* ── Auth ──────────────────────────────────────────────────── */

export interface TokenResponse {
  access_token: string
  token_type: string
  expires_in: number
  user_id: string
}

export interface LoginRequest {
  email: string
  password: string
}

export interface RegisterRequest {
  email: string
  password: string
  profile: UserProfile
}

/* ── User profile (mirrors backend UserProfile Pydantic schema) */
export interface UserProfile {
  gender: 'F' | 'M'
  age_range: '18-24' | '25-34' | '35-44' | '45-54' | '55+'
  attendance_freq: 'Never' | 'Rarely' | 'Occasionally' | 'Often' | 'Very often'
  top_event: 'Concert' | 'Festival' | 'Sports' | 'Theatre' | 'Conference'
  preferred_event_types: string          // comma-separated e.g. "Concert,Festival"
  indoor_outdoor: 0 | 1                 // 0=indoor, 1=outdoor
  // Weather tolerance 1–5
  rain_avoid: number
  cold_tolerance: number
  heat_sensitivity: number
  wind_sensitivity: number
  override_weather: number
  // Scenario interest 0–3
  scenario_concert: number
  scenario_festival: number
  scenario_sports: number
  scenario_theatre: number
  scenario_conference: number
}

export interface UserProfileResponse {
  user_id: string
  email: string
  profile: UserProfile | null
}

/* ── Recommendations ───────────────────────────────────────── */

export interface WeatherContext {
  city: string
  date: string
  temp_C: number | null
  humidity_pct: number | null
  precip_mm: number | null
  wind_speed_kmh: number | null
}

export interface EventRecommendation {
  event_type: string
  location: string
  event_date: string
  attended_prob: number
  climate_zone?: string
  is_outdoor?: number
  weather?: WeatherContext
}

export interface RecommendRequest {
  user_id?: string
  user_profile?: UserProfile
  city?: string
  date?: string
  hour?: number
  top_n?: number
  model?: 'lgbm' | 'xgb' | 'rf_strict'
}

export interface RecommendResponse {
  user_id?: string
  city?: string
  date?: string
  model_used: string
  weather?: WeatherContext
  recommendations: EventRecommendation[]
  total_scored: number
}

export type ModelId = 'lgbm' | 'xgb' | 'rf_strict'

export interface ModelInfo {
  id: ModelId
  name: string
  description: string
  default: boolean
}

/* ── UI helpers ────────────────────────────────────────────── */

export type Theme = 'dark' | 'light'
