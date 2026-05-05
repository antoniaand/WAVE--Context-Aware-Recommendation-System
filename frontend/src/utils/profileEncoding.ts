import type { UserProfile } from '../types'

/* ── Survey answer types ──────────────────────────────────── */

export type GenderAnswer = 'Female' | 'Male' | 'Non-binary' | 'Prefer not to say'
export type AgeAnswer = '<18' | '18-24' | '25-34' | '35-44' | '45-54' | '55-64' | '65+'
export type FreqAnswer = 'Never' | '1-2/year' | 'Monthly' | 'Weekly'
export type EventTypeAnswer = 'Concert' | 'Festival' | 'Sports' | 'Theatre' | 'Conference' | 'Exhibition'
export type ScenarioChoice = 'Would' | 'Probably' | 'Probably not' | 'Would not'

export interface SurveyAnswers {
  gender: GenderAnswer
  age: AgeAnswer
  freq: FreqAnswer
  preferred_types: EventTypeAnswer[]
  indoor_outdoor_slider: 1 | 2 | 3 | 4 | 5
  top_event: EventTypeAnswer
  // Likert 1–5; stored as-is — model was trained on this raw scale
  rain_avoid_likert: 1 | 2 | 3 | 4 | 5
  cold_tolerance_likert: 1 | 2 | 3 | 4 | 5
  heat_sensitivity_likert: 1 | 2 | 3 | 4 | 5
  wind_sensitivity_likert: 1 | 2 | 3 | 4 | 5
  override_weather_likert: 1 | 2 | 3 | 4 | 5
  scenario_concert: ScenarioChoice
  scenario_festival: ScenarioChoice
  scenario_sports: ScenarioChoice
  scenario_theatre: ScenarioChoice
  scenario_conference: ScenarioChoice
}

/* ── Encoding helpers ─────────────────────────────────────── */

// Matches legacy dataset_pipeline.py mapping (simulate_labels.py divides by 3.0)
function encodeScenario(choice: ScenarioChoice): number {
  const map: Record<ScenarioChoice, number> = {
    Would: 3,
    Probably: 2,
    'Probably not': 1,
    'Would not': 0,
  }
  return map[choice]
}

const GENDER_ENCODE: Record<GenderAnswer, 'F' | 'M'> = {
  Female: 'F',
  Male: 'M',
  'Non-binary': 'F',
  'Prefer not to say': 'F',
}

const AGE_ENCODE: Record<AgeAnswer, UserProfile['age_range']> = {
  '<18': '18-24',
  '18-24': '18-24',
  '25-34': '25-34',
  '35-44': '35-44',
  '45-54': '45-54',
  '55-64': '55+',
  '65+': '55+',
}

const FREQ_ENCODE: Record<FreqAnswer, UserProfile['attendance_freq']> = {
  Never: 'Never',
  '1-2/year': 'Rarely',
  Monthly: 'Occasionally',
  Weekly: 'Often',
}

const TOP_EVENT_ENCODE: Record<EventTypeAnswer, UserProfile['top_event']> = {
  Concert: 'Concert',
  Festival: 'Festival',
  Sports: 'Sports',
  Theatre: 'Theatre',
  Conference: 'Conference',
  Exhibition: 'Conference',
}

/* ── encode: SurveyAnswers → UserProfile ─────────────────── */

export function encodeProfile(answers: SurveyAnswers): UserProfile {
  const slider = answers.indoor_outdoor_slider
  const indoor_outdoor: 0 | 1 = slider >= 4 ? 1 : 0

  const preferred_event_types = answers.preferred_types
    .map(t => TOP_EVENT_ENCODE[t])
    .filter((v, i, arr) => arr.indexOf(v) === i)
    .join(',')

  return {
    gender: GENDER_ENCODE[answers.gender],
    age_range: AGE_ENCODE[answers.age],
    attendance_freq: FREQ_ENCODE[answers.freq],
    top_event: TOP_EVENT_ENCODE[answers.top_event],
    preferred_event_types,
    indoor_outdoor,
    // Likert values passed through as-is (1–5); model was trained on this scale
    rain_avoid: answers.rain_avoid_likert,
    cold_tolerance: answers.cold_tolerance_likert,
    heat_sensitivity: answers.heat_sensitivity_likert,
    wind_sensitivity: answers.wind_sensitivity_likert,
    override_weather: answers.override_weather_likert,
    scenario_concert: encodeScenario(answers.scenario_concert),
    scenario_festival: encodeScenario(answers.scenario_festival),
    scenario_sports: encodeScenario(answers.scenario_sports),
    scenario_theatre: encodeScenario(answers.scenario_theatre),
    scenario_conference: encodeScenario(answers.scenario_conference),
  }
}

/* ── Decoding helpers ─────────────────────────────────────── */

function decodeScenario(value: number): ScenarioChoice {
  if (value >= 3) return 'Would'
  if (value >= 2) return 'Probably'
  if (value >= 1) return 'Probably not'
  return 'Would not'
}

const AGE_DECODE: Record<UserProfile['age_range'], AgeAnswer> = {
  '18-24': '18-24',
  '25-34': '25-34',
  '35-44': '35-44',
  '45-54': '45-54',
  '55+': '55-64',
}

const FREQ_DECODE: Record<UserProfile['attendance_freq'], FreqAnswer> = {
  Never: 'Never',
  Rarely: '1-2/year',
  Occasionally: 'Monthly',
  Often: 'Weekly',
  'Very often': 'Weekly',
}

const TOP_EVENT_DECODE: Record<UserProfile['top_event'], EventTypeAnswer> = {
  Concert: 'Concert',
  Festival: 'Festival',
  Sports: 'Sports',
  Theatre: 'Theatre',
  Conference: 'Conference',
}

/* ── decode: UserProfile → SurveyAnswers ─────────────────── */

export function decodeProfile(profile: UserProfile): SurveyAnswers {
  const preferred_types = profile.preferred_event_types
    ? (profile.preferred_event_types.split(',').filter(Boolean) as EventTypeAnswer[])
    : []

  // indoor_outdoor=0 → slider 2 (≤2 encodes to 0); =1 → slider 4 (≥4 encodes to 1)
  const indoor_outdoor_slider: 1 | 2 | 3 | 4 | 5 = profile.indoor_outdoor === 1 ? 4 : 2

  const clampLikert = (v: number): 1 | 2 | 3 | 4 | 5 =>
    (Math.min(5, Math.max(1, Math.round(v))) as 1 | 2 | 3 | 4 | 5)

  return {
    gender: profile.gender === 'M' ? 'Male' : 'Female',
    age: AGE_DECODE[profile.age_range],
    freq: FREQ_DECODE[profile.attendance_freq],
    preferred_types,
    indoor_outdoor_slider,
    top_event: TOP_EVENT_DECODE[profile.top_event],
    // Likert values are stored as-is; just clamp to valid range on decode
    rain_avoid_likert: clampLikert(profile.rain_avoid),
    cold_tolerance_likert: clampLikert(profile.cold_tolerance),
    heat_sensitivity_likert: clampLikert(profile.heat_sensitivity),
    wind_sensitivity_likert: clampLikert(profile.wind_sensitivity),
    override_weather_likert: clampLikert(profile.override_weather),
    scenario_concert: decodeScenario(profile.scenario_concert),
    scenario_festival: decodeScenario(profile.scenario_festival),
    scenario_sports: decodeScenario(profile.scenario_sports),
    scenario_theatre: decodeScenario(profile.scenario_theatre),
    scenario_conference: decodeScenario(profile.scenario_conference),
  }
}
