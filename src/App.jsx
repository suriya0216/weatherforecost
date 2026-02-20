import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import {
  BarChart3,
  Bell,
  CalendarDays,
  ChevronDown,
  CloudRain,
  CloudSun,
  Facebook,
  Instagram,
  LayoutDashboard,
  Loader2,
  LogOut,
  Map,
  Menu,
  Mic,
  Navigation,
  Pause,
  Play,
  Search,
  Settings,
  Square,
  Sun,
  Twitter,
  Volume2,
  Wind,
  X,
  Linkedin,
} from 'lucide-react'
import {
  fetchGeocodeSuggestions,
  fetchNotifications,
  fetchPersonalizedForecast,
  fetchReverseGeocode,
  fetchSidebarSection,
  fetchVoiceExplanation,
} from './api/weatherApi'

const AUTH_USER_KEY = 'todo_weather_user'
const AUTH_SESSION_KEY = 'todo_weather_session'
const UI_SETTINGS_KEY = 'todo_weather_ui_settings'

const DEFAULT_UI_SETTINGS = {
  temperature_unit: 'C',
  wind_unit: 'kmh',
  time_format: '24h',
  date_style: 'short',
  assistant_language: 'en',
  notifications_enabled: true,
  auto_refresh: false,
  compact_calendar: false,
}

const NAV_ITEMS = [
  { key: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { key: 'statistics', label: 'Statistics', icon: BarChart3 },
  { key: 'map', label: 'Map', icon: Map },
  { key: 'calendar', label: 'Calendar', icon: CalendarDays },
  { key: 'setting', label: 'Setting', icon: Settings },
]

const VOICE_LANGUAGES = [
  { code: 'en', label: 'English', speech: 'en-US' },
  { code: 'ta', label: 'Tamil', speech: 'ta-IN' },
  { code: 'hi', label: 'Hindi', speech: 'hi-IN' },
  { code: 'es', label: 'Spanish', speech: 'es-ES' },
  { code: 'fr', label: 'French', speech: 'fr-FR' },
]

const CITY_CARDS = [
  { name: 'Amsterdam', image: 'https://images.unsplash.com/photo-1472396961693-142e6e269027?auto=format&fit=crop&w=800&q=80' },
  { name: 'London', image: 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?auto=format&fit=crop&w=800&q=80' },
  { name: 'Budapest', image: 'https://images.unsplash.com/photo-1541849546-216549ae216d?auto=format&fit=crop&w=800&q=80' },
  { name: 'Paris', image: 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=800&q=80' },
  { name: 'Chicago', image: 'https://images.unsplash.com/photo-1494522855154-9297ac14b55f?auto=format&fit=crop&w=800&q=80' },
]

const LANDING_FEATURES = [
  {
    key: 'global',
    title: 'Real-Time Global Forecast',
    copy: 'Search any city worldwide and get instant temperature, humidity and wind speed updates.',
    icon: CloudSun,
    points: ['Search any city', 'Instant temperature', 'Live humidity and wind'],
  },
  {
    key: 'intelligence',
    title: 'AI Weather Intelligence',
    copy: 'Get predictive analysis, storm and rain alerts, and smart weather explanations.',
    icon: Wind,
    points: ['Predictive analysis', 'Storm and rain alerts', 'Smart reasoning'],
  },
  {
    key: 'prediction',
    title: '7-Day and 15-Day Prediction',
    copy: 'Track temperature trends, rain probability and variation charts for upcoming days.',
    icon: CalendarDays,
    points: ['Trend graphs', 'Rain probability', 'Temperature charts'],
  },
  {
    key: 'maps',
    title: 'Interactive Weather Maps',
    copy: 'Explore rain radar, wind direction and heat map visualization in one place.',
    icon: Map,
    points: ['Rain radar', 'Wind direction', 'Heat map view'],
  },
]

const LANDING_NAV_LINKS = [
  { key: 'home', label: 'Home', target: 'landing-top' },
  { key: 'live', label: 'Live Forecast', target: 'landing-live-forecast' },
  { key: 'week', label: '7-Day Prediction', target: 'landing-features' },
  { key: 'maps', label: 'Weather Maps', target: 'landing-features' },
  { key: 'climate', label: 'Climate Insights', target: 'landing-why' },
  { key: 'voice', label: 'AI Weather Voice', target: 'landing-why' },
  { key: 'about', label: 'About', target: 'landing-about' },
  { key: 'contact', label: 'Contact', target: 'landing-contact' },
]

const LANDING_WHY_POINTS = [
  { key: 'accuracy', label: 'High Accuracy Forecasting', icon: BarChart3 },
  { key: 'reasoning', label: 'AI-Powered Weather Reasoning', icon: Wind },
  { key: 'voice', label: 'Multilingual Voice Assistant', icon: Volume2 },
  { key: 'alerts', label: 'Location-Based Alerts', icon: Navigation },
  { key: 'risk', label: 'Climate Risk Detection', icon: Bell },
  { key: 'mobile', label: 'Mobile Friendly Design', icon: LayoutDashboard },
]

const LANDING_EXPERIENCES = [
  {
    key: 'rain',
    title: 'Rain Theme',
    copy: 'Soft rain particles and cool mist gradient when rain probability increases.',
    icon: CloudRain,
  },
  {
    key: 'wind',
    title: 'Wind Theme',
    copy: 'Directional particle streams and animated gust lines during windy periods.',
    icon: Wind,
  },
  {
    key: 'sun',
    title: 'Sunny Theme',
    copy: 'Warm sunlight bloom and crisp highlights for clear sky predictions.',
    icon: Sun,
  },
  {
    key: 'storm',
    title: 'Storm Theme',
    copy: 'Low-light atmosphere with brief lightning accents for storm alerts.',
    icon: CloudSun,
  },
]

const LANDING_QUICK_LINKS = ['Home', 'Live Forecast', '7-Day Prediction', 'Weather Maps', 'Climate Insights']
const LANDING_WEATHER_RESOURCES = ['Rain Radar', 'Wind Tracker', 'Heat Map', 'UV Index', 'Storm Alerts']
const LANDING_CONTACT_INFO = ['support@todoweather.ai', '+1 (202) 555-0186', 'New York, United States']

const LANDING_SOCIALS = [
  { key: 'facebook', label: 'Facebook', icon: Facebook },
  { key: 'twitter', label: 'Twitter', icon: Twitter },
  { key: 'linkedin', label: 'LinkedIn', icon: Linkedin },
  { key: 'instagram', label: 'Instagram', icon: Instagram },
]

const LANDING_AUTO_SECTION_IDS = [
  'landing-top',
  'landing-features',
  'landing-why',
  'landing-live-forecast',
  'landing-experience',
  'landing-contact',
]

const DEFAULT_ROUTINE = {
  wake_time: '07:00',
  commute_time: '08:30',
  workout_time: '18:00',
  sleep_time: '22:30',
}

const DEFAULT_PREFS = {
  outdoor_commute: true,
  uv_sensitive: true,
  air_quality_sensitive: false,
}

const FALLBACK = {
  location: {
    name: 'New York',
    country: 'United States',
    latitude: 40.7128,
    longitude: -74.006,
    timezone: 'America/New_York',
  },
  current: {
    temperature_c: 4,
    humidity_percent: 70,
    wind_kph: 4.5,
    weather: 'Partly cloudy',
  },
  metrics: {
    rain_peak_probability: 58,
    uv_peak: 5.4,
  },
  forecast_quality: {
    confidence_score: 72,
  },
  alerts: [],
  hourly: [
    { time: '2026-02-20T09:00:00', temperature_c: -6, wind_kph: 12 },
    { time: '2026-02-20T10:00:00', temperature_c: -10, wind_kph: 11 },
    { time: '2026-02-20T11:00:00', temperature_c: -9, wind_kph: 9 },
    { time: '2026-02-20T12:00:00', temperature_c: -11, wind_kph: 10 },
    { time: '2026-02-20T13:00:00', temperature_c: -2, wind_kph: 13 },
    { time: '2026-02-20T14:00:00', temperature_c: 3, wind_kph: 14 },
    { time: '2026-02-20T15:00:00', temperature_c: 9, wind_kph: 18 },
    { time: '2026-02-20T16:00:00', temperature_c: 1, wind_kph: 12 },
    { time: '2026-02-20T17:00:00', temperature_c: 6, wind_kph: 13 },
    { time: '2026-02-20T18:00:00', temperature_c: 2, wind_kph: 11 },
  ],
  daily: [
    { date: '2026-02-21', weather: 'Light rain', temp_max_c: 4, temp_min_c: -5 },
    { date: '2026-02-22', weather: 'Partly cloudy', temp_max_c: 4, temp_min_c: -5 },
    { date: '2026-02-23', weather: 'Clear sky', temp_max_c: 4, temp_min_c: -5 },
    { date: '2026-02-24', weather: 'Partly cloudy', temp_max_c: 4, temp_min_c: -5 },
    { date: '2026-02-25', weather: 'Clear sky', temp_max_c: 4, temp_min_c: -5 },
    { date: '2026-02-26', weather: 'Light rain', temp_max_c: 4, temp_min_c: -5 },
    { date: '2026-02-27', weather: 'Partly cloudy', temp_max_c: 4, temp_min_c: -5 },
    { date: '2026-02-28', weather: 'Light rain', temp_max_c: 4, temp_min_c: -5 },
  ],
}

function weatherIconFromLabel(text) {
  const label = String(text || '').toLowerCase()
  if (label.includes('rain') || label.includes('drizzle') || label.includes('storm')) return CloudRain
  if (label.includes('clear') || label.includes('sun')) return Sun
  return CloudSun
}

function mapSuggestionLabel(item) {
  return [item.name, item.admin1, item.country].filter(Boolean).join(', ')
}

function formatDate(value, style = 'short') {
  if (!value) return '--'
  const stamp = new Date(value)
  if (Number.isNaN(stamp.getTime())) return String(value)
  if (style === 'long') {
    return stamp.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' })
  }
  return stamp.toLocaleDateString([], { weekday: 'short', month: 'short', day: '2-digit' })
}

function formatNotificationTime(value, hour12 = false) {
  if (!value) return '--'
  const stamp = new Date(value)
  if (Number.isNaN(stamp.getTime())) return String(value)
  return stamp.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12 })
}

function toDateKey(value) {
  const stamp = value instanceof Date ? value : new Date(value)
  if (Number.isNaN(stamp.getTime())) return ''
  const year = stamp.getFullYear()
  const month = String(stamp.getMonth() + 1).padStart(2, '0')
  const day = String(stamp.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function getMonthTitle(dateValue) {
  const stamp = new Date(dateValue)
  return stamp.toLocaleDateString([], { month: 'long', year: 'numeric' })
}

function buildMonthGrid(baseDate) {
  const monthStart = new Date(baseDate.getFullYear(), baseDate.getMonth(), 1)
  const gridStart = new Date(monthStart)
  gridStart.setDate(monthStart.getDate() - monthStart.getDay())

  const days = []
  for (let idx = 0; idx < 42; idx += 1) {
    const day = new Date(gridStart)
    day.setDate(gridStart.getDate() + idx)
    days.push(day)
  }
  return days
}

function displayRounded(value, suffix = '') {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--'
  return `${Math.round(Number(value))}${suffix}`
}

function readStore(key) {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

function writeStore(key, value) {
  localStorage.setItem(key, JSON.stringify(value))
}

function normalizeUiSettings(value) {
  const input = value && typeof value === 'object' ? value : {}
  const allowedVoiceLanguages = new Set(VOICE_LANGUAGES.map((item) => item.code))
  return {
    temperature_unit: input.temperature_unit === 'F' ? 'F' : 'C',
    wind_unit: input.wind_unit === 'mph' ? 'mph' : 'kmh',
    time_format: input.time_format === '12h' ? '12h' : '24h',
    date_style: input.date_style === 'long' ? 'long' : 'short',
    assistant_language: allowedVoiceLanguages.has(input.assistant_language) ? input.assistant_language : 'en',
    notifications_enabled: input.notifications_enabled !== false,
    auto_refresh: Boolean(input.auto_refresh),
    compact_calendar: Boolean(input.compact_calendar),
  }
}

function App() {
  const [authMode, setAuthMode] = useState('login')
  const [authError, setAuthError] = useState('')
  const [showAuthPanel, setShowAuthPanel] = useState(false)
  const [authLaunching, setAuthLaunching] = useState(false)
  const [launchStatus, setLaunchStatus] = useState('')
  const [landingMenuOpen, setLandingMenuOpen] = useState(false)
  const [landingScrolled, setLandingScrolled] = useState(false)
  const [landingMood, setLandingMood] = useState('sunny')
  const [authForm, setAuthForm] = useState({ name: '', email: '', password: '', confirmPassword: '' })
  const [currentUser, setCurrentUser] = useState(() => readStore(AUTH_SESSION_KEY))
  const [dashboardIntro, setDashboardIntro] = useState(false)
  const [welcomeMessage, setWelcomeMessage] = useState('')
  const [uiSettings, setUiSettings] = useState(() => normalizeUiSettings(readStore(UI_SETTINGS_KEY) || DEFAULT_UI_SETTINGS))

  const [activeSection, setActiveSection] = useState('dashboard')
  const [sectionPayloads, setSectionPayloads] = useState({})
  const [sectionLoading, setSectionLoading] = useState(false)
  const [sectionError, setSectionError] = useState('')

  const [query, setQuery] = useState('London')
  const [selectedLocation, setSelectedLocation] = useState(null)
  const [suggestions, setSuggestions] = useState([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [loadingSuggestions, setLoadingSuggestions] = useState(false)
  const [loadingForecast, setLoadingForecast] = useState(false)
  const [locating, setLocating] = useState(false)
  const [forecast, setForecast] = useState(null)
  const [error, setError] = useState('')
  const [calendarFilter, setCalendarFilter] = useState('all')
  const [calendarMonth, setCalendarMonth] = useState(() => new Date())
  const [calendarSelectedKey, setCalendarSelectedKey] = useState('')
  const [settingsNotice, setSettingsNotice] = useState('')
  const [notificationsOpen, setNotificationsOpen] = useState(false)
  const [notificationsLoading, setNotificationsLoading] = useState(false)
  const [notificationsError, setNotificationsError] = useState('')
  const [notificationsPayload, setNotificationsPayload] = useState({ items: [], generated_at_utc: '' })
  const [voiceLoading, setVoiceLoading] = useState(false)
  const [voiceError, setVoiceError] = useState('')
  const [voiceNotice, setVoiceNotice] = useState('')
  const [voiceData, setVoiceData] = useState(null)
  const [voiceSpeaking, setVoiceSpeaking] = useState(false)
  const [voicePaused, setVoicePaused] = useState(false)
  const searchRef = useRef(null)
  const notificationRef = useRef(null)
  const speechUtteranceRef = useRef(null)
  const mapRef = useRef(null)
  const mapContainerRef = useRef(null)
  const mapMarkerLayerRef = useRef(null)
  const launchTimersRef = useRef([])
  const landingAutoScrollIntervalRef = useRef(null)
  const landingAutoScrollResumeRef = useRef(null)
  const landingAutoScrollIndexRef = useRef(0)

  const clearLaunchTimers = useCallback(() => {
    launchTimersRef.current.forEach((timerId) => clearTimeout(timerId))
    launchTimersRef.current = []
  }, [])

  const launchIntoDashboard = useCallback((sessionUser) => {
    clearLaunchTimers()
    setAuthLaunching(true)
    setLaunchStatus('Verifying account...')
    setAuthError('')
    setShowAuthPanel(false)
    setLandingMenuOpen(false)

    launchTimersRef.current.push(setTimeout(() => {
      setLaunchStatus('Preparing weather intelligence...')
    }, 520))

    launchTimersRef.current.push(setTimeout(() => {
      setLaunchStatus('Launching dashboard...')
    }, 980))

    launchTimersRef.current.push(setTimeout(() => {
      setCurrentUser(sessionUser)
      setDashboardIntro(true)
      setAuthLaunching(false)
      setLaunchStatus('')
      setWelcomeMessage('Welcome to Todo Weather Forecast')
    }, 1450))

    launchTimersRef.current.push(setTimeout(() => {
      setDashboardIntro(false)
    }, 2500))
  }, [clearLaunchTimers])

  const stopLandingAutoScroll = useCallback(() => {
    if (landingAutoScrollIntervalRef.current) {
      clearInterval(landingAutoScrollIntervalRef.current)
      landingAutoScrollIntervalRef.current = null
    }
  }, [])

  const clearLandingAutoResume = useCallback(() => {
    if (landingAutoScrollResumeRef.current) {
      clearTimeout(landingAutoScrollResumeRef.current)
      landingAutoScrollResumeRef.current = null
    }
  }, [])

  const resolveNextLandingIndex = useCallback(() => {
    const marker = window.scrollY + 140
    const activeIndex = LANDING_AUTO_SECTION_IDS.findIndex((sectionId) => {
      const element = document.getElementById(sectionId)
      if (!element) return false
      const top = element.offsetTop
      const bottom = top + Math.max(element.offsetHeight, window.innerHeight * 0.4)
      return marker >= top && marker < bottom
    })
    if (activeIndex === -1) return 1
    return (activeIndex + 1) % LANDING_AUTO_SECTION_IDS.length
  }, [])

  const runLandingAutoStep = useCallback(() => {
    if (currentUser || showAuthPanel || authLaunching) return
    if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return
    const targetId = LANDING_AUTO_SECTION_IDS[landingAutoScrollIndexRef.current]
    const target = targetId ? document.getElementById(targetId) : null
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
    landingAutoScrollIndexRef.current = (landingAutoScrollIndexRef.current + 1) % LANDING_AUTO_SECTION_IDS.length
  }, [authLaunching, currentUser, showAuthPanel])

  const startLandingAutoScroll = useCallback(() => {
    if (currentUser || showAuthPanel || authLaunching) return
    stopLandingAutoScroll()
    landingAutoScrollIndexRef.current = resolveNextLandingIndex()
    landingAutoScrollIntervalRef.current = setInterval(() => {
      runLandingAutoStep()
    }, 5600)
  }, [authLaunching, currentUser, resolveNextLandingIndex, runLandingAutoStep, showAuthPanel, stopLandingAutoScroll])

  const pauseLandingAutoScroll = useCallback((resumeDelayMs = 9000) => {
    stopLandingAutoScroll()
    clearLandingAutoResume()
    landingAutoScrollResumeRef.current = setTimeout(() => {
      if (!currentUser && !showAuthPanel && !authLaunching) {
        startLandingAutoScroll()
      }
    }, resumeDelayMs)
  }, [authLaunching, clearLandingAutoResume, currentUser, showAuthPanel, startLandingAutoScroll, stopLandingAutoScroll])

  const safeForecast = forecast || FALLBACK
  const current = safeForecast.current || FALLBACK.current
  const metrics = safeForecast.metrics || FALLBACK.metrics
  const quality = safeForecast.forecast_quality || FALLBACK.forecast_quality
  const hourly = (safeForecast.hourly || FALLBACK.hourly).slice(0, 10)
  const daily = (safeForecast.daily || FALLBACK.daily).slice(0, 8)
  const locationName = safeForecast.location?.name || query
  const currentTimezone = safeForecast.location?.timezone || 'auto'
  const WeatherIcon = weatherIconFromLabel(current.weather)
  const slidePayload = sectionPayloads[activeSection]
  const activeNav = NAV_ITEMS.find((item) => item.key === activeSection)
  const tempUnit = uiSettings.temperature_unit === 'F' ? 'F' : 'C'
  const windUnit = uiSettings.wind_unit === 'mph' ? 'mph' : 'kmh'
  const useHour12 = uiSettings.time_format === '12h'
  const dateStyle = uiSettings.date_style === 'long' ? 'long' : 'short'
  const notificationItems = Array.isArray(notificationsPayload.items) ? notificationsPayload.items : []
  const unreadNotifications = notificationItems.filter((item) => ['high', 'medium'].includes(String(item.severity || '').toLowerCase())).length
  const activeVoiceLanguage = VOICE_LANGUAGES.find((item) => item.code === uiSettings.assistant_language) || VOICE_LANGUAGES[0]
  const voiceSections = Array.isArray(voiceData?.sections) ? voiceData.sections : []
  const voiceTranscript = String(voiceData?.transcript || '').trim()
  const selectedSpeechLanguage = String(voiceData?.speech_language || activeVoiceLanguage.speech || 'en-US')
  const selectedSpeechCode = String(voiceData?.language || activeVoiceLanguage.code || 'en').toLowerCase()
  const landingPreviewHours = hourly.slice(0, 8)
  const landingPreviewTemps = landingPreviewHours.map((item) => Number(item.temperature_c || 0))
  const landingTempMin = Math.min(...landingPreviewTemps, 0)
  const landingTempMax = Math.max(...landingPreviewTemps, 1)
  const landingTempSpread = Math.max(1, landingTempMax - landingTempMin)
  const landingThemeClass = useMemo(() => {
    const condition = String(current.weather || '').toLowerCase()
    if (condition.includes('storm') || condition.includes('thunder')) return 'theme-storm'
    if (condition.includes('rain') || condition.includes('drizzle')) return 'theme-rain'
    if (condition.includes('clear') || condition.includes('sun')) return 'theme-sun'
    return 'theme-cloud'
  }, [current.weather])
  const formatTemp = useCallback((value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '--'
    const numeric = Number(value)
    if (tempUnit === 'F') return `${Math.round((numeric * 9) / 5 + 32)}F`
    return `${Math.round(numeric)}C`
  }, [tempUnit])

  const formatWind = useCallback((value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '--'
    const numeric = Number(value)
    if (windUnit === 'mph') return `${Math.round(numeric * 0.621371)} mph`
    return `${Math.round(numeric)} km/h`
  }, [windUnit])

  const formatHourLabel = useCallback((value) => {
    if (!value) return '--:--'
    const stamp = new Date(value)
    if (Number.isNaN(stamp.getTime())) return String(value).slice(-5)
    return stamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: useHour12 })
  }, [useHour12])

  const formatDateLabel = useCallback((value) => formatDate(value, dateStyle), [dateStyle])

  const formatClockLabel = useCallback((value = new Date()) => {
    const stamp = value instanceof Date ? value : new Date(value)
    if (Number.isNaN(stamp.getTime())) return '--:--'
    return stamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: useHour12 })
  }, [useHour12])

  const formatMonthTitle = useCallback((value) => getMonthTitle(value), [])
  const formatNotificationLabel = useCallback((value) => formatNotificationTime(value, useHour12), [useHour12])

  const chartSeries = useMemo(() => {
    const temps = hourly.map((item) => Number(item.temperature_c || 0))
    const min = Math.min(...temps)
    const max = Math.max(...temps)
    const spread = Math.max(1, max - min)
    const points = temps.map((temp, idx) => ({
      x: 36 + idx * 74,
      y: 220 - ((temp - min) / spread) * 160,
    }))
    const split = Math.max(4, Math.floor(points.length / 2))
    return {
      blue: points.slice(0, split).map((p) => `${p.x},${p.y}`).join(' '),
      red: points.slice(Math.max(0, split - 1)).map((p) => `${p.x},${p.y}`).join(' '),
      dots: points,
    }
  }, [hourly])

  const calendarSchedule = useMemo(() => {
    if (activeSection !== 'calendar') return []
    return Array.isArray(slidePayload?.schedule) ? slidePayload.schedule : []
  }, [activeSection, slidePayload])

  const calendarScheduleByKey = useMemo(() => {
    const mapping = {}
    calendarSchedule.forEach((entry) => {
      const key = toDateKey(entry.date)
      if (key) mapping[key] = entry
    })
    return mapping
  }, [calendarSchedule])

  const calendarGridDays = useMemo(() => buildMonthGrid(calendarMonth), [calendarMonth])

  const requestForecast = useCallback(async ({ locationQuery, location }) => {
    setLoadingForecast(true)
    setError('')
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel()
    }
    setVoiceSpeaking(false)
    setVoicePaused(false)
    setVoiceData(null)
    setVoiceError('')
    setVoiceNotice('')
    try {
      const payload = {
        routine: DEFAULT_ROUTINE,
        preferences: DEFAULT_PREFS,
      }
      if (location) payload.location = location
      else payload.location_query = locationQuery
      const result = await fetchPersonalizedForecast(payload)
      setForecast(result)
      setSectionPayloads({})
      setSectionError('')
    } catch (requestError) {
      setError(requestError?.message || 'Unable to load personalized forecast')
    } finally {
      setLoadingForecast(false)
    }
  }, [])

  const loadSection = useCallback(async (sectionKey, force = false) => {
    if (sectionKey === 'dashboard') return
    if (!force && sectionPayloads[sectionKey]) return

    const locationFromForecast = safeForecast.location || {}
    const options = { locationQuery: query.trim() || locationFromForecast.name || 'London' }
    if (selectedLocation?.latitude !== undefined && selectedLocation?.longitude !== undefined) {
      options.location = selectedLocation
    } else if (locationFromForecast.latitude !== undefined && locationFromForecast.longitude !== undefined) {
      options.location = {
        name: locationFromForecast.name,
        latitude: locationFromForecast.latitude,
        longitude: locationFromForecast.longitude,
        timezone: locationFromForecast.timezone || 'auto',
      }
    }

    setSectionLoading(true)
    setSectionError('')
    try {
      const payload = await fetchSidebarSection(sectionKey, options)
      setSectionPayloads((prev) => ({ ...prev, [sectionKey]: payload }))
    } catch (slideError) {
      setSectionError(slideError?.message || 'Unable to load section.')
    } finally {
      setSectionLoading(false)
    }
  }, [query, safeForecast.location, sectionPayloads, selectedLocation])

  const loadNotifications = useCallback(async (force = false) => {
    if (!currentUser) return
    if (!uiSettings.notifications_enabled) {
      setNotificationsPayload({ items: [], generated_at_utc: '' })
      setNotificationsError('Notifications are disabled in settings.')
      return
    }
    if (!force && notificationsLoading) return

    const locationFromForecast = safeForecast.location || {}
    const options = { locationQuery: query.trim() || locationFromForecast.name || 'London', limit: 8 }
    if (selectedLocation?.latitude !== undefined && selectedLocation?.longitude !== undefined) {
      options.location = selectedLocation
    } else if (locationFromForecast.latitude !== undefined && locationFromForecast.longitude !== undefined) {
      options.location = {
        name: locationFromForecast.name,
        latitude: locationFromForecast.latitude,
        longitude: locationFromForecast.longitude,
        timezone: locationFromForecast.timezone || 'auto',
      }
    }

    setNotificationsLoading(true)
    setNotificationsError('')
    try {
      const payload = await fetchNotifications(options)
      setNotificationsPayload({
        items: Array.isArray(payload?.items) ? payload.items : [],
        generated_at_utc: payload?.generated_at_utc || '',
      })
    } catch (notificationRequestError) {
      setNotificationsError(notificationRequestError?.message || 'Unable to load notifications.')
    } finally {
      setNotificationsLoading(false)
    }
  }, [
    currentUser,
    notificationsLoading,
    query,
    safeForecast.location,
    selectedLocation,
    uiSettings.notifications_enabled,
  ])

  const onToggleNotifications = useCallback(() => {
    setNotificationsOpen((prev) => {
      const next = !prev
      if (next) {
        void loadNotifications(true)
      }
      return next
    })
  }, [loadNotifications])

  const waitForSpeechVoices = useCallback(async (synth) => {
    const existing = synth.getVoices()
    if (existing.length) return existing

    return new Promise((resolve) => {
      let settled = false
      const finish = () => {
        if (settled) return
        settled = true
        resolve(synth.getVoices())
      }

      const timeout = window.setTimeout(() => {
        cleanup()
        finish()
      }, 1200)

      const onVoicesChanged = () => {
        cleanup()
        finish()
      }

      const cleanup = () => {
        window.clearTimeout(timeout)
        if (typeof synth.removeEventListener === 'function') synth.removeEventListener('voiceschanged', onVoicesChanged)
        if (synth.onvoiceschanged === onVoicesChanged) synth.onvoiceschanged = null
      }

      if (typeof synth.addEventListener === 'function') synth.addEventListener('voiceschanged', onVoicesChanged)
      else synth.onvoiceschanged = onVoicesChanged
      synth.getVoices()
    })
  }, [])

  const resolveSpeechRate = useCallback((languageCode) => {
    if (languageCode === 'ta' || languageCode === 'hi') return 0.76
    if (languageCode === 'es' || languageCode === 'fr') return 0.82
    return 0.86
  }, [])

  const pickBestVoice = useCallback((voices, speechLanguage, languageCode) => {
    const loweredSpeech = String(speechLanguage || '').toLowerCase()
    const loweredCode = String(languageCode || '').toLowerCase()
    const normalized = loweredSpeech.split('-')[0]
    return (
      voices.find((voice) => String(voice.lang || '').toLowerCase() === loweredSpeech)
      || voices.find((voice) => String(voice.lang || '').toLowerCase().startsWith(`${normalized}-`))
      || voices.find((voice) => String(voice.lang || '').toLowerCase().startsWith(loweredCode))
      || null
    )
  }, [])

  const loadVoiceExplanation = useCallback(async (force = true) => {
    if (!currentUser) return
    if (!force && voiceData?.language === activeVoiceLanguage.code) return

    const locationFromForecast = safeForecast.location || {}
    const options = {
      locationQuery: query.trim() || locationFromForecast.name || 'London',
      language: activeVoiceLanguage.code,
      horizonDays: 5,
    }
    if (selectedLocation?.latitude !== undefined && selectedLocation?.longitude !== undefined) {
      options.location = selectedLocation
    } else if (locationFromForecast.latitude !== undefined && locationFromForecast.longitude !== undefined) {
      options.location = {
        name: locationFromForecast.name,
        latitude: locationFromForecast.latitude,
        longitude: locationFromForecast.longitude,
        timezone: locationFromForecast.timezone || 'auto',
      }
    }

    setVoiceLoading(true)
    setVoiceError('')
    setVoiceNotice('')
    try {
      const payload = await fetchVoiceExplanation(options)
      setVoiceData(payload)
    } catch (voiceRequestError) {
      setVoiceError(voiceRequestError?.message || 'Unable to generate voice weather explanation.')
    } finally {
      setVoiceLoading(false)
    }
  }, [
    activeVoiceLanguage.code,
    currentUser,
    query,
    safeForecast.location,
    selectedLocation,
    voiceData?.language,
  ])

  const stopVoicePlayback = useCallback(() => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return
    window.speechSynthesis.cancel()
    speechUtteranceRef.current = null
    setVoiceSpeaking(false)
    setVoicePaused(false)
    setVoiceNotice('')
  }, [])

  const pauseVoicePlayback = useCallback(() => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return
    if (!window.speechSynthesis.speaking) return
    window.speechSynthesis.pause()
    setVoicePaused(true)
  }, [])

  const playVoicePlayback = useCallback(async () => {
    if (typeof window === 'undefined' || !window.speechSynthesis) {
      setVoiceError('Voice playback is not supported in this browser.')
      return
    }
    if (!voiceTranscript) {
      setVoiceError('Generate voice explanation first.')
      return
    }
    setVoiceError('')
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume()
      setVoicePaused(false)
      setVoiceSpeaking(true)
      return
    }

    const utterance = new window.SpeechSynthesisUtterance(voiceTranscript)
    utterance.lang = selectedSpeechLanguage
    utterance.rate = resolveSpeechRate(selectedSpeechCode)
    utterance.pitch = 1

    const allVoices = await waitForSpeechVoices(window.speechSynthesis)
    const selectedVoice = pickBestVoice(allVoices, selectedSpeechLanguage, selectedSpeechCode)
    if (selectedVoice) {
      utterance.voice = selectedVoice
      setVoiceNotice(`Voice: ${selectedVoice.name}`)
    } else {
      setVoiceNotice(`Using default voice for ${selectedSpeechLanguage}. Install this language voice pack for clearer speech.`)
    }

    utterance.onstart = () => {
      setVoiceSpeaking(true)
      setVoicePaused(false)
    }
    utterance.onpause = () => setVoicePaused(true)
    utterance.onresume = () => {
      setVoicePaused(false)
      setVoiceSpeaking(true)
    }
    utterance.onend = () => {
      setVoiceSpeaking(false)
      setVoicePaused(false)
      speechUtteranceRef.current = null
    }
    utterance.onerror = () => {
      setVoiceSpeaking(false)
      setVoicePaused(false)
      speechUtteranceRef.current = null
      setVoiceError('Voice playback failed in this browser.')
    }

    speechUtteranceRef.current = utterance
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(utterance)
  }, [
    pickBestVoice,
    resolveSpeechRate,
    selectedSpeechCode,
    selectedSpeechLanguage,
    voiceTranscript,
    waitForSpeechVoices,
  ])

  useEffect(() => {
    document.title = 'Todo Weather'
  }, [])

  useEffect(() => {
    writeStore(UI_SETTINGS_KEY, uiSettings)
  }, [uiSettings])

  useEffect(() => {
    const outsideClick = (event) => {
      if (!searchRef.current?.contains(event.target)) setShowSuggestions(false)
      if (!notificationRef.current?.contains(event.target)) setNotificationsOpen(false)
    }
    document.addEventListener('mousedown', outsideClick)
    return () => document.removeEventListener('mousedown', outsideClick)
  }, [])

  useEffect(() => {
    if (!uiSettings.notifications_enabled) {
      setNotificationsOpen(false)
    }
  }, [uiSettings.notifications_enabled])

  useEffect(() => {
    return () => {
      if (typeof window !== 'undefined' && window.speechSynthesis) {
        window.speechSynthesis.cancel()
      }
    }
  }, [])

  useEffect(() => {
    if (!voiceData) return
    stopVoicePlayback()
  }, [activeVoiceLanguage.code, stopVoicePlayback, voiceData])

  useEffect(() => {
    if (!currentUser) return
    void requestForecast({ locationQuery: 'London' })
  }, [currentUser, requestForecast])

  useEffect(() => {
    if (!currentUser || activeSection === 'dashboard') return
    if (sectionPayloads[activeSection]) return
    void loadSection(activeSection)
  }, [activeSection, currentUser, loadSection, sectionPayloads])

  useEffect(() => {
    if (!currentUser) return
    const debounce = setTimeout(async () => {
      const trimmed = query.trim()
      if (trimmed.length < 2 || selectedLocation?.name === trimmed) {
        setSuggestions([])
        return
      }
      setLoadingSuggestions(true)
      try {
        const response = await fetchGeocodeSuggestions(trimmed)
        setSuggestions(response?.results || [])
      } catch {
        setSuggestions([])
      } finally {
        setLoadingSuggestions(false)
      }
    }, 300)
    return () => clearTimeout(debounce)
  }, [currentUser, query, selectedLocation])

  useEffect(() => {
    if (currentUser) {
      setLandingMenuOpen(false)
      setLandingScrolled(false)
      return
    }
    const onScroll = () => setLandingScrolled(window.scrollY > 18)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [currentUser])

  useEffect(() => {
    if (currentUser) {
      stopLandingAutoScroll()
      clearLandingAutoResume()
      return
    }
    if (showAuthPanel || authLaunching) {
      pauseLandingAutoScroll(12000)
      return
    }

    startLandingAutoScroll()
    const pauseForInteraction = () => pauseLandingAutoScroll()

    window.addEventListener('wheel', pauseForInteraction, { passive: true })
    window.addEventListener('touchstart', pauseForInteraction, { passive: true })
    window.addEventListener('mousedown', pauseForInteraction)
    window.addEventListener('keydown', pauseForInteraction)
    document.addEventListener('visibilitychange', pauseForInteraction)

    return () => {
      window.removeEventListener('wheel', pauseForInteraction)
      window.removeEventListener('touchstart', pauseForInteraction)
      window.removeEventListener('mousedown', pauseForInteraction)
      window.removeEventListener('keydown', pauseForInteraction)
      document.removeEventListener('visibilitychange', pauseForInteraction)
      stopLandingAutoScroll()
      clearLandingAutoResume()
    }
  }, [
    authLaunching,
    clearLandingAutoResume,
    currentUser,
    pauseLandingAutoScroll,
    showAuthPanel,
    startLandingAutoScroll,
    stopLandingAutoScroll,
  ])

  useEffect(() => {
    if (!welcomeMessage) return
    const timer = setTimeout(() => setWelcomeMessage(''), 3200)
    return () => clearTimeout(timer)
  }, [welcomeMessage])

  const openAuthPanel = useCallback(() => {
    if (authLaunching) return
    pauseLandingAutoScroll(14000)
    setLandingMood('sunny')
    setAuthMode('login')
    setLandingMenuOpen(false)
    setAuthError('')
    setShowAuthPanel(true)
  }, [authLaunching, pauseLandingAutoScroll])

  const openRegisterPanel = useCallback(() => {
    if (authLaunching) return
    pauseLandingAutoScroll(14000)
    setLandingMood('rainy')
    setAuthMode('register')
    setLandingMenuOpen(false)
    setAuthError('')
    setShowAuthPanel(true)
  }, [authLaunching, pauseLandingAutoScroll])

  const onLandingNavSelect = useCallback((target) => {
    pauseLandingAutoScroll(9000)
    setLandingMenuOpen(false)
    if (!target) return
    document.getElementById(target)?.scrollIntoView({ behavior: 'smooth' })
  }, [pauseLandingAutoScroll])

  const onAuthSubmit = (event) => {
    event.preventDefault()
    if (authLaunching) return
    setAuthError('')
    const name = authForm.name.trim()
    const email = authForm.email.trim().toLowerCase()
    const password = authForm.password
    const confirmPassword = authForm.confirmPassword

    if (!email || !password) return setAuthError('Enter email and password.')
    if (authMode === 'register') {
      if (!name) return setAuthError('Enter your name.')
      if (password.length < 6) return setAuthError('Password must be at least 6 characters.')
      if (password !== confirmPassword) return setAuthError('Password and confirm password do not match.')

      writeStore(AUTH_USER_KEY, { name, email, password })
      writeStore(AUTH_SESSION_KEY, { name, email })
      launchIntoDashboard({ name, email })
      return
    }

    const storedUser = readStore(AUTH_USER_KEY)
    if (!storedUser) return setAuthError('No account found. Please register first.')
    if (storedUser.email !== email || storedUser.password !== password) {
      return setAuthError('Invalid login credentials.')
    }
    writeStore(AUTH_SESSION_KEY, { name: storedUser.name, email: storedUser.email })
    launchIntoDashboard({ name: storedUser.name, email: storedUser.email })
  }

  const onLogout = () => {
    stopVoicePlayback()
    clearLaunchTimers()
    stopLandingAutoScroll()
    clearLandingAutoResume()
    localStorage.removeItem(AUTH_SESSION_KEY)
    setCurrentUser(null)
    setDashboardIntro(false)
    setAuthLaunching(false)
    setLaunchStatus('')
    setForecast(null)
    setSectionPayloads({})
    setActiveSection('dashboard')
    setNotificationsOpen(false)
    setNotificationsPayload({ items: [], generated_at_utc: '' })
    setNotificationsError('')
    setVoiceData(null)
    setVoiceError('')
    setVoiceNotice('')
    setVoiceLoading(false)
    setWelcomeMessage('')
    setLandingMenuOpen(false)
    setLandingScrolled(false)
    setLandingMood('sunny')
    setError('')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const runSearch = useCallback(async () => {
    const trimmed = query.trim()
    if (!trimmed) return
    if (selectedLocation && selectedLocation.name === trimmed) return requestForecast({ location: selectedLocation })
    setSelectedLocation(null)
    await requestForecast({ locationQuery: trimmed })
  }, [query, requestForecast, selectedLocation])

  useEffect(() => {
    if (!currentUser || !uiSettings.auto_refresh) return
    const timer = setInterval(() => {
      if (activeSection === 'dashboard') {
        void runSearch()
      } else {
        void loadSection(activeSection, true)
      }
    }, 60000)
    return () => clearInterval(timer)
  }, [activeSection, currentUser, loadSection, runSearch, uiSettings.auto_refresh])

  useEffect(() => {
    if (activeSection !== 'calendar') return
    if (!calendarSchedule.length) return

    if (!calendarSelectedKey || !calendarScheduleByKey[calendarSelectedKey]) {
      const firstDate = new Date(calendarSchedule[0].date)
      if (!Number.isNaN(firstDate.getTime())) {
        setCalendarSelectedKey(toDateKey(firstDate))
        setCalendarMonth(new Date(firstDate.getFullYear(), firstDate.getMonth(), 1))
      }
    }
  }, [activeSection, calendarSchedule, calendarScheduleByKey, calendarSelectedKey])

  useEffect(() => {
    return () => {
      if (mapRef.current) {
        mapRef.current.remove()
        mapRef.current = null
      }
    }
  }, [])

  useEffect(() => () => clearLaunchTimers(), [clearLaunchTimers])
  useEffect(() => () => { stopLandingAutoScroll(); clearLandingAutoResume() }, [clearLandingAutoResume, stopLandingAutoScroll])

  const onUseMyLocation = async () => {
    if (!navigator.geolocation) return setError('Geolocation is not supported in this browser.')
    setLocating(true)
    setError('')
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        try {
          const latitude = position.coords.latitude
          const longitude = position.coords.longitude
          const reverse = await fetchReverseGeocode(latitude, longitude)
          const result = reverse?.result
          const label = result ? mapSuggestionLabel(result) : `My Location (${latitude.toFixed(2)}, ${longitude.toFixed(2)})`
          const picked = { name: label, latitude, longitude, timezone: result?.timezone || 'auto' }
          setQuery(label)
          setSelectedLocation(picked)
          setShowSuggestions(false)
          await requestForecast({ location: picked })
        } catch (locationError) {
          setError(locationError?.message || 'Unable to detect your location.')
        } finally {
          setLocating(false)
        }
      },
      () => {
        setLocating(false)
        setError('Location access denied. Please allow location permission.')
      },
      { enableHighAccuracy: true, timeout: 10000 },
    )
  }

  const onMapPointSelect = useCallback(async (point) => {
    const label = mapSuggestionLabel(point)
    const picked = {
      name: label,
      latitude: point.latitude,
      longitude: point.longitude,
      timezone: currentTimezone,
    }
    setQuery(label)
    setSelectedLocation(picked)
    setShowSuggestions(false)
    await requestForecast({ location: picked })
    setActiveSection('dashboard')
  }, [currentTimezone, requestForecast])

  useEffect(() => {
    if (activeSection !== 'map') return
    const center = slidePayload?.center
    if (!center || center.latitude === undefined || center.longitude === undefined) return
    if (!mapContainerRef.current) return

    const centerCoords = [Number(center.latitude), Number(center.longitude)]
    if (Number.isNaN(centerCoords[0]) || Number.isNaN(centerCoords[1])) return

    if (!mapRef.current) {
      const mapInstance = L.map(mapContainerRef.current, { zoomControl: true })
      mapInstance.setView(centerCoords, 7)
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
      }).addTo(mapInstance)
      mapRef.current = mapInstance
    }

    const mapInstance = mapRef.current
    mapInstance.setView(centerCoords, Math.max(mapInstance.getZoom(), 7))

    if (mapMarkerLayerRef.current) {
      mapInstance.removeLayer(mapMarkerLayerRef.current)
    }
    const markerLayer = L.layerGroup().addTo(mapInstance)
    mapMarkerLayerRef.current = markerLayer

    L.circleMarker(centerCoords, {
      radius: 9,
      color: '#495bd1',
      fillColor: '#6475ef',
      fillOpacity: 0.8,
      weight: 2,
    })
      .bindPopup(`Selected: ${locationName}`)
      .addTo(markerLayer)

    const points = Array.isArray(slidePayload?.points) ? slidePayload.points : []
    const bounds = [centerCoords]
    points.forEach((point) => {
      if (point.latitude === undefined || point.longitude === undefined) return
      const lat = Number(point.latitude)
      const lon = Number(point.longitude)
      if (Number.isNaN(lat) || Number.isNaN(lon)) return
      bounds.push([lat, lon])
      L.circleMarker([lat, lon], {
        radius: 6,
        color: '#2d3f82',
        fillColor: '#7a87d9',
        fillOpacity: 0.75,
        weight: 1.4,
      })
        .bindTooltip(point.name || 'Point')
        .on('click', () => {
          void onMapPointSelect(point)
        })
        .addTo(markerLayer)
    })

    if (bounds.length > 1) {
      mapInstance.fitBounds(bounds, { padding: [20, 20] })
    }
    window.setTimeout(() => mapInstance.invalidateSize(), 120)
  }, [activeSection, locationName, onMapPointSelect, slidePayload])

  const saveSetting = (nextSettings) => {
    setUiSettings(normalizeUiSettings(nextSettings))
    setSettingsNotice('Settings saved.')
    window.setTimeout(() => setSettingsNotice(''), 1800)
  }

  const renderStatisticsFeatures = () => {
    const overview = slidePayload?.overview || {}
    const trend = Array.isArray(slidePayload?.trend) ? slidePayload.trend.slice(0, 10) : []
    const alerts = Array.isArray(slidePayload?.alerts) ? slidePayload.alerts : []
    const temps = trend.map((item) => Number(item.temperature_c || 0))
    const min = temps.length ? Math.min(...temps) : 0
    const max = temps.length ? Math.max(...temps) : 1
    const spread = Math.max(1, max - min)

    return (
      <div className="wx-feature-grid">
        <article className="wx-feature-card">
          <h3>Statistics Overview</h3>
          <div className="wx-overview-grid">
            <p><span>Average Temp</span><strong>{formatTemp(overview.average_temp_c)}</strong></p>
            <p><span>Average Wind</span><strong>{formatWind(overview.average_wind_kph)}</strong></p>
            <p><span>Max Rain</span><strong>{displayRounded(overview.max_rain_probability, '%')}</strong></p>
            <p><span>Average AQI</span><strong>{displayRounded(overview.average_aqi_us)}</strong></p>
            <p><span>Severe Alerts</span><strong>{displayRounded(overview.severe_alert_count)}</strong></p>
          </div>
        </article>

        <article className="wx-feature-card">
          <h3>Hourly Trend</h3>
          {trend.length ? (
            <div className="wx-mini-bars">
              {trend.map((item) => {
                const value = Number(item.temperature_c || min)
                const height = 12 + ((value - min) / spread) * 65
                return (
                  <div key={`trend-${item.time}`} className="wx-mini-bar-col">
                    <div className="wx-mini-bar" style={{ height: `${height}px` }} />
                    <small>{item.time}</small>
                  </div>
                )
              })}
            </div>
          ) : <p className="wx-slide-empty">No trend points available.</p>}
        </article>

        <article className="wx-feature-card wx-feature-card--full">
          <h3>Alerts and Trend Table</h3>
          <div className="wx-feature-inline">
            <div className="wx-alert-list">
              {alerts.length ? alerts.map((alert, idx) => (
                <p key={`alert-${idx}`}>
                  <span>{alert.event || 'Weather Alert'}</span>
                  <strong>{String(alert.severity || 'unknown').toUpperCase()}</strong>
                </p>
              )) : <p className="wx-slide-empty">No active alerts.</p>}
            </div>
            <div className="wx-table-wrap">
              <table className="wx-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Temp</th>
                    <th>Rain</th>
                    <th>Wind</th>
                    <th>AQI</th>
                  </tr>
                </thead>
                <tbody>
                  {trend.map((item, idx) => (
                    <tr key={`stat-row-${idx}`}>
                      <td>{item.time}</td>
                      <td>{formatTemp(item.temperature_c)}</td>
                      <td>{displayRounded(item.rain_probability, '%')}</td>
                      <td>{formatWind(item.wind_kph)}</td>
                      <td>{displayRounded(item.aqi_us)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </article>
      </div>
    )
  }

  const renderMapFeatures = () => {
    const center = slidePayload?.center || {}
    const points = Array.isArray(slidePayload?.points) ? slidePayload.points : []
    const currentWeather = slidePayload?.current_weather || {}

    return (
      <div className="wx-feature-grid">
        <article className="wx-feature-card wx-feature-card--full">
          <div className="wx-map-head">
            <h3>Live Map</h3>
            <p>{`Center: ${displayRounded(center.latitude)}, ${displayRounded(center.longitude)}`}</p>
          </div>
          <div ref={mapContainerRef} className="wx-live-map" />
          <p className="wx-feature-note">Tip: click nearby marker to switch forecast location.</p>
        </article>

        <article className="wx-feature-card">
          <h3>Map Center</h3>
          <div className="wx-overview-grid">
            <p><span>Latitude</span><strong>{displayRounded(center.latitude)}</strong></p>
            <p><span>Longitude</span><strong>{displayRounded(center.longitude)}</strong></p>
            <p><span>Timezone</span><strong>{center.timezone || '--'}</strong></p>
            <p><span>Temperature</span><strong>{formatTemp(currentWeather.temperature_c)}</strong></p>
            <p><span>Wind</span><strong>{formatWind(currentWeather.wind_kph)}</strong></p>
          </div>
          <p className="wx-feature-note">{slidePayload?.note || ''}</p>
        </article>

        <article className="wx-feature-card">
          <h3>Nearby Places</h3>
          {points.length ? (
            <div className="wx-points-grid">
              {points.map((point) => (
                <button
                  key={`${point.name}-${point.latitude}-${point.longitude}`}
                  type="button"
                  className="wx-point-card"
                  onClick={() => void onMapPointSelect(point)}
                >
                  <span>{point.name}</span>
                  <small>{[point.admin1, point.country].filter(Boolean).join(', ')}</small>
                  <strong>{`${displayRounded(point.latitude)}, ${displayRounded(point.longitude)}`}</strong>
                  <em>Use this location</em>
                </button>
              ))}
            </div>
          ) : (
            <p className="wx-slide-empty">No nearby points found.</p>
          )}
        </article>
      </div>
    )
  }

  const renderCalendarFeatures = () => {
    const schedule = Array.isArray(slidePayload?.schedule) ? slidePayload.schedule : []
    const filtered = schedule.filter((item) => calendarFilter === 'all' || String(item.risk || '').toLowerCase() === calendarFilter)
    const bestDay = schedule.find((item) => String(item.risk || '').toLowerCase() === 'low') || schedule[0]
    const selectedDateKey = calendarSelectedKey || toDateKey(new Date())
    const selectedEntry = calendarScheduleByKey[selectedDateKey]
    const weekdayLabels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

    return (
      <div className="wx-feature-grid">
        <article className="wx-feature-card">
          <h3>Calendar Summary</h3>
          <div className="wx-overview-grid">
            <p><span>Total Days</span><strong>{displayRounded(schedule.length)}</strong></p>
            <p><span>Low Risk Days</span><strong>{displayRounded(schedule.filter((d) => d.risk === 'low').length)}</strong></p>
            <p><span>Moderate Days</span><strong>{displayRounded(schedule.filter((d) => d.risk === 'moderate').length)}</strong></p>
            <p><span>High Risk Days</span><strong>{displayRounded(schedule.filter((d) => d.risk === 'high').length)}</strong></p>
          </div>
          <p className="wx-feature-note">Best day: {bestDay ? `${formatDateLabel(bestDay.date)} (${bestDay.risk})` : '--'}</p>
        </article>

        <article className="wx-feature-card wx-feature-card--full">
          <div className="wx-calendar-head wx-calendar-head--month">
            <button
              type="button"
              className="wx-month-nav"
              onClick={() => setCalendarMonth((prev) => new Date(prev.getFullYear(), prev.getMonth() - 1, 1))}
            >
              {'<'}
            </button>
            <h3>{formatMonthTitle(calendarMonth)}</h3>
            <button
              type="button"
              className="wx-month-nav"
              onClick={() => setCalendarMonth((prev) => new Date(prev.getFullYear(), prev.getMonth() + 1, 1))}
            >
              {'>'}
            </button>
          </div>
          <div className="wx-month-grid-head">
            {weekdayLabels.map((label) => <span key={label}>{label}</span>)}
          </div>
          <div className="wx-month-grid">
            {calendarGridDays.map((day) => {
              const dayKey = toDateKey(day)
              const entry = calendarScheduleByKey[dayKey]
              const inCurrentMonth = day.getMonth() === calendarMonth.getMonth()
              return (
                <button
                  key={dayKey}
                  type="button"
                  className={`wx-month-cell${inCurrentMonth ? '' : ' dim'}${dayKey === selectedDateKey ? ' selected' : ''}${entry ? ' has-event' : ''}`}
                  onClick={() => setCalendarSelectedKey(dayKey)}
                >
                  <strong>{day.getDate()}</strong>
                  {entry ? <small className={`wx-risk-tag ${String(entry.risk || '').toLowerCase()}`}>{entry.risk}</small> : null}
                </button>
              )
            })}
          </div>
        </article>

        <article className="wx-feature-card wx-feature-card--full">
          <div className="wx-calendar-head">
            <h3>Planner Events</h3>
            <div className="wx-pill-row">
              <button type="button" className={calendarFilter === 'all' ? 'active' : ''} onClick={() => setCalendarFilter('all')}>All</button>
              <button type="button" className={calendarFilter === 'low' ? 'active' : ''} onClick={() => setCalendarFilter('low')}>Low</button>
              <button type="button" className={calendarFilter === 'moderate' ? 'active' : ''} onClick={() => setCalendarFilter('moderate')}>Moderate</button>
              <button type="button" className={calendarFilter === 'high' ? 'active' : ''} onClick={() => setCalendarFilter('high')}>High</button>
            </div>
          </div>
          {selectedEntry ? (
            <div className="wx-selected-day">
              <p>{formatDateLabel(selectedEntry.date)}</p>
              <span>{selectedEntry.weather}</span>
              <strong>{`${formatTemp(selectedEntry.temp_min_c)} to ${formatTemp(selectedEntry.temp_max_c)}`}</strong>
              <small>{`Rain ${displayRounded(selectedEntry.rain_probability_max, '%')}`}</small>
              <em className={`wx-risk-tag ${String(selectedEntry.risk || '').toLowerCase()}`}>{selectedEntry.risk}</em>
              <i>{selectedEntry.action}</i>
            </div>
          ) : (
            <p className="wx-slide-empty">Select a day from calendar to view details.</p>
          )}
          {filtered.length ? (
            <div className={`wx-calendar-grid${uiSettings.compact_calendar ? ' compact' : ''}`}>
              {filtered.map((day) => (
                <article key={day.date} className="wx-calendar-card">
                  <p>{formatDateLabel(day.date)}</p>
                  <span>{day.weather}</span>
                  <strong>{`${formatTemp(day.temp_min_c)} to ${formatTemp(day.temp_max_c)}`}</strong>
                  <small>{`Rain ${displayRounded(day.rain_probability_max, '%')}`}</small>
                  <em className={`wx-risk-tag ${String(day.risk || '').toLowerCase()}`}>{day.risk}</em>
                </article>
              ))}
            </div>
          ) : (
            <p className="wx-slide-empty">No days in selected filter.</p>
          )}
        </article>
      </div>
    )
  }

  const renderSettingFeatures = () => {
    const backendSettings = Array.isArray(slidePayload?.settings) ? slidePayload.settings : []
    const recommendations = Array.isArray(slidePayload?.recommendations) ? slidePayload.recommendations : []
    return (
      <div className="wx-feature-grid">
        <article className="wx-feature-card">
          <h3>UI Settings</h3>
          <div className="wx-settings-group">
            <label>
              Temperature Unit
              <select
                value={uiSettings.temperature_unit}
                onChange={(event) => saveSetting({ ...uiSettings, temperature_unit: event.target.value })}
              >
                <option value="C">Celsius (C)</option>
                <option value="F">Fahrenheit (F)</option>
              </select>
            </label>
            <label>
              Wind Speed Unit
              <select
                value={uiSettings.wind_unit}
                onChange={(event) => saveSetting({ ...uiSettings, wind_unit: event.target.value })}
              >
                <option value="kmh">Kilometers per hour (km/h)</option>
                <option value="mph">Miles per hour (mph)</option>
              </select>
            </label>
            <label>
              Time Format
              <select
                value={uiSettings.time_format}
                onChange={(event) => saveSetting({ ...uiSettings, time_format: event.target.value })}
              >
                <option value="24h">24-hour</option>
                <option value="12h">12-hour</option>
              </select>
            </label>
            <label>
              Date Style
              <select
                value={uiSettings.date_style}
                onChange={(event) => saveSetting({ ...uiSettings, date_style: event.target.value })}
              >
                <option value="short">Short</option>
                <option value="long">Long</option>
              </select>
            </label>
            <label>
              Voice Assistant Language
              <select
                value={uiSettings.assistant_language}
                onChange={(event) => saveSetting({ ...uiSettings, assistant_language: event.target.value })}
              >
                {VOICE_LANGUAGES.map((item) => (
                  <option key={item.code} value={item.code}>{item.label}</option>
                ))}
              </select>
            </label>
            <label className="wx-check">
              <input
                type="checkbox"
                checked={uiSettings.notifications_enabled}
                onChange={(event) => saveSetting({ ...uiSettings, notifications_enabled: event.target.checked })}
              />
              Enable Weather Notifications
            </label>
            <label className="wx-check">
              <input
                type="checkbox"
                checked={uiSettings.auto_refresh}
                onChange={(event) => saveSetting({ ...uiSettings, auto_refresh: event.target.checked })}
              />
              Auto Refresh Every 60 Seconds
            </label>
            <label className="wx-check">
              <input
                type="checkbox"
                checked={uiSettings.compact_calendar}
                onChange={(event) => saveSetting({ ...uiSettings, compact_calendar: event.target.checked })}
              />
              Compact Calendar Cards
            </label>
          </div>
          <p className="wx-feature-note">These settings apply to Dashboard, Statistics, Map, Calendar, notifications and voice assistant.</p>
          {settingsNotice ? <p className="wx-feature-note success">{settingsNotice}</p> : null}
        </article>

        <article className="wx-feature-card">
          <h3>Backend Settings</h3>
          {backendSettings.length ? (
            <div className="wx-list">
              {backendSettings.map((item) => (
                <p key={item.key}>
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </p>
              ))}
            </div>
          ) : <p className="wx-slide-empty">No backend settings found.</p>}
        </article>

        <article className="wx-feature-card wx-feature-card--full">
          <h3>Recommendations</h3>
          {recommendations.length ? (
            <ul className="wx-bullets">
              {recommendations.map((item, idx) => (
                <li key={`rec-${idx}`}>{item}</li>
              ))}
            </ul>
          ) : <p className="wx-slide-empty">No recommendations available.</p>}
        </article>
      </div>
    )
  }

  const renderVoiceAssistantCard = () => (
    <section className="wx-voice-card">
      <div className="wx-voice-head">
        <div className="wx-voice-title-wrap">
          <span className={`wx-voice-mic${voiceLoading || voiceSpeaking ? ' active' : ''}`}>
            <Mic size={15} />
          </span>
          <div>
            <h3><Volume2 size={15} />SkyVoice AI Assistant</h3>
            <p>Ask Why to generate weather reason, tomorrow outlook and future trend.</p>
          </div>
        </div>
        <div className={`wx-voice-wave${voiceSpeaking ? ' live' : ''}`}>
          {Array.from({ length: 8 }).map((_, idx) => <span key={`wave-${idx}`} />)}
        </div>
      </div>

      <div className="wx-voice-actions">
        <label>
          Language
          <select
            value={uiSettings.assistant_language}
            onChange={(event) => saveSetting({ ...uiSettings, assistant_language: event.target.value })}
          >
            {VOICE_LANGUAGES.map((item) => (
              <option key={item.code} value={item.code}>{item.label}</option>
            ))}
          </select>
        </label>
        <button type="button" className="wx-ask-btn" onClick={() => void loadVoiceExplanation(true)} disabled={voiceLoading}>
          {voiceLoading ? <><Loader2 size={14} className="spin" />Thinking...</> : 'Ask Why?'}
        </button>
        <div className="wx-voice-player">
          <button type="button" onClick={() => void playVoicePlayback()} disabled={!voiceTranscript || voiceLoading}>
            <Play size={14} />Play
          </button>
          <button type="button" onClick={pauseVoicePlayback} disabled={!voiceSpeaking || voicePaused}>
            <Pause size={14} />Pause
          </button>
          <button type="button" onClick={stopVoicePlayback} disabled={!voiceSpeaking && !voicePaused}>
            <Square size={14} />Stop
          </button>
        </div>
      </div>

      {voiceNotice ? <p className="wx-voice-note">{voiceNotice}</p> : null}
      {voiceError ? <p className="wx-error">{voiceError}</p> : null}
      {!voiceError && voiceLoading ? <p className="wx-slide-loading"><Loader2 size={14} className="spin" />Generating explanation...</p> : null}

      {!voiceError && !voiceLoading ? (
        voiceTranscript ? (
          <>
            <div className="wx-voice-transcript">
              <h4>Transcript</h4>
              <p>{voiceTranscript}</p>
            </div>
            <div className="wx-voice-sections">
              {voiceSections.map((item) => (
                <article key={item.key}>
                  <strong>{item.label}</strong>
                  <p>{item.text}</p>
                </article>
              ))}
            </div>
          </>
        ) : <p className="wx-slide-empty">No voice explanation yet. Click Ask Why.</p>
      ) : null}
    </section>
  )

  if (!currentUser) {
    return (
      <div className={`landing-page landing-theme-${landingMood}`}>
        <div className="landing-bg landing-bg--base" />
        <div className="landing-bg landing-bg--rays" />
        <div className="landing-bg landing-bg--clouds" />
        <div className="landing-bg landing-bg--flash" />
        <div className="landing-particles" aria-hidden="true">
          {Array.from({ length: 16 }).map((_, idx) => (
            <span
              key={`particle-${idx}`}
              style={{ left: `${(idx * 11) % 100}%`, animationDelay: `${(idx % 8) * 0.5}s` }}
            />
          ))}
        </div>

        <header className={`landing-nav${landingScrolled ? ' scrolled' : ''}`}>
          <div className="landing-nav-left">
            <p className="landing-brand"><CloudSun size={18} />Todo Weather Forecast</p>
          </div>

          <button
            type="button"
            className="landing-nav-toggle"
            aria-label="Toggle navigation"
            aria-expanded={landingMenuOpen}
            onClick={() => setLandingMenuOpen((prev) => !prev)}
          >
            {landingMenuOpen ? <X size={18} /> : <Menu size={18} />}
          </button>

          <nav className={`landing-nav-center${landingMenuOpen ? ' open' : ''}`}>
            {LANDING_NAV_LINKS.map((link) => (
              <button
                key={link.key}
                type="button"
                className="landing-nav-link"
                onClick={() => onLandingNavSelect(link.target)}
              >
                {link.label}
              </button>
            ))}
            <div className="landing-nav-mobile-actions">
              <button type="button" className="landing-nav-login" disabled={authLaunching} onClick={openAuthPanel}>Login</button>
              <button type="button" className="landing-nav-signup" disabled={authLaunching} onClick={openRegisterPanel}>Sign Up</button>
            </div>
          </nav>

          <div className="landing-nav-right">
            <button type="button" className="landing-nav-login" disabled={authLaunching} onClick={openAuthPanel}>Login</button>
            <button type="button" className="landing-nav-signup" disabled={authLaunching} onClick={openRegisterPanel}>Sign Up</button>
          </div>
        </header>

        <section className="landing-hero" id="landing-top">
          <p className="landing-kicker">Smart, Accurate and Real-Time Weather Intelligence Powered by AI</p>
          <h1>Todo Weather Forecast</h1>
          <p>
            Track live weather conditions, explore 7-day predictions, analyze climate patterns, and receive
            intelligent weather insights for any location worldwide.
          </p>
          <div className="landing-cta">
            <button
              type="button"
              className="landing-cta-primary"
              disabled={authLaunching}
              onClick={openAuthPanel}
            >
              GET STARTED
            </button>
            <button
              type="button"
              className="landing-cta-ghost"
              onClick={() => onLandingNavSelect('landing-live-forecast')}
            >
              Explore Live Weather
            </button>
          </div>
        </section>

        <button
          type="button"
          className="landing-scroll-indicator"
          onClick={() => onLandingNavSelect('landing-features')}
        >
          <span>Scroll Down</span>
          <ChevronDown size={16} />
        </button>

        <section className="landing-features" id="landing-features">
          <div className="landing-section-head">
            <p>Weather Features</p>
            <h2>Premium weather intelligence with live updates, prediction modeling and map analytics</h2>
          </div>
          <div className="landing-feature-grid">
            {LANDING_FEATURES.map((item) => {
              const Icon = item.icon
              return (
                <article key={item.key} className="landing-feature-card">
                  <div className="landing-feature-icon"><Icon size={18} /></div>
                  <h3>{item.title}</h3>
                  <p>{item.copy}</p>
                  {item.points?.length ? (
                    <ul className="landing-feature-points">
                      {item.points.map((point) => <li key={`${item.key}-${point}`}>{point}</li>)}
                    </ul>
                  ) : null}
                </article>
              )
            })}
          </div>
        </section>

        <section className="landing-why" id="landing-why">
          <div className="landing-section-head">
            <p>Why Choose Todo Weather Forecast</p>
            <h2>Designed for intelligent forecasting, alert readiness and smooth mobile-first experience</h2>
          </div>
          <div className="landing-why-grid">
            {LANDING_WHY_POINTS.map((item) => {
              const Icon = item.icon
              return (
                <article key={item.key} className="landing-why-item">
                  <Icon size={16} />
                  <p>{item.label}</p>
                </article>
              )
            })}
          </div>
        </section>

        <section className="landing-preview" id="landing-live-forecast">
          <div className="landing-section-head">
            <p>Live Weather Preview</p>
            <h2>Preview the dashboard with live location search, graph trend and seven-day outlook</h2>
          </div>
          <div className={`landing-preview-shell ${landingThemeClass}`}>
            <article className="landing-preview-main">
              <div className="landing-preview-search">
                <Search size={15} />
                <input value={locationName || query} readOnly aria-label="Weather location preview" />
                <span>Live</span>
              </div>
              <p className="landing-preview-tag">Todo Weather Forecast</p>
              <h3>{locationName}</h3>
              <div className="landing-preview-temp">
                <WeatherIcon size={46} />
                <strong>{formatTemp(current.temperature_c)}</strong>
              </div>
              <p className="landing-preview-desc">{current.weather || 'Partly cloudy'}</p>
              <div className="landing-preview-stats">
                <p><span>Humidity</span><strong>{displayRounded(current.humidity_percent, '%')}</strong></p>
                <p><span>Wind</span><strong>{formatWind(current.wind_kph)}</strong></p>
                <p><span>UV Index</span><strong>{displayRounded(metrics.uv_peak)}</strong></p>
                <p><span>Pressure</span><strong>{displayRounded(safeForecast.current?.surface_pressure_hpa, ' hPa')}</strong></p>
              </div>
              <button type="button" className="landing-preview-cta" disabled={authLaunching} onClick={openAuthPanel}>
                GET STARTED
              </button>
            </article>

            <article className="landing-preview-side">
              <h3>Forecast Graph</h3>
              <div className="landing-hour-bars">
                {landingPreviewHours.map((item) => {
                  const value = Number(item.temperature_c || 0)
                  const ratio = (value - landingTempMin) / landingTempSpread
                  const height = 24 + Math.round(ratio * 54)
                  return (
                    <div key={item.time} className="landing-hour-col">
                      <span className="landing-hour-bar" style={{ height: `${height}px` }} />
                      <small>{formatHourLabel(item.time)}</small>
                      <strong>{formatTemp(item.temperature_c)}</strong>
                    </div>
                  )
                })}
              </div>

              <div className="landing-preview-days">
                {daily.slice(0, 7).map((day) => {
                  const DayIcon = weatherIconFromLabel(day.weather)
                  return (
                    <article key={`landing-day-${day.date}`} className="landing-preview-day">
                      <DayIcon size={15} />
                      <p>{formatDateLabel(day.date)}</p>
                      <strong>{`${formatTemp(day.temp_min_c)} / ${formatTemp(day.temp_max_c)}`}</strong>
                    </article>
                  )
                })}
              </div>
            </article>
          </div>
        </section>

        <section className="landing-experience" id="landing-experience">
          <div className="landing-section-head">
            <p>Weather Experience</p>
            <h2>Dynamic themes that react to sunny, rainy, windy and storm weather conditions</h2>
          </div>
          <div className="landing-experience-grid">
            {LANDING_EXPERIENCES.map((item) => {
              const Icon = item.icon
              return (
                <article key={item.key} className={`landing-exp-card ${item.key}`}>
                  <div className="landing-exp-icon"><Icon size={16} /></div>
                  <h3>{item.title}</h3>
                  <p>{item.copy}</p>
                </article>
              )
            })}
          </div>
        </section>

        <section className="landing-final-cta">
          <h2>Stay Ahead of the Weather</h2>
          <p>
            Get intelligent weather insights, real-time alerts, and predictive forecasts anytime, anywhere.
          </p>
          <button type="button" className="landing-final-btn" disabled={authLaunching} onClick={openAuthPanel}>
            GET STARTED
          </button>
        </section>

        <footer className="landing-footer" id="landing-contact">
          <div className="landing-footer-grid">
            <article id="landing-about">
              <h3>About Todo Weather Forecast</h3>
              <p>
                Todo Weather Forecast delivers AI-powered weather intelligence with live alerts,
                predictive insights and clean forecasting experiences for daily planning.
              </p>
            </article>

            <article>
              <h4>Quick Links</h4>
              <ul>
                {LANDING_QUICK_LINKS.map((item) => <li key={item}>{item}</li>)}
              </ul>
            </article>

            <article>
              <h4>Weather Resources</h4>
              <ul>
                {LANDING_WEATHER_RESOURCES.map((item) => <li key={item}>{item}</li>)}
              </ul>
            </article>

            <article>
              <h4>Contact Information</h4>
              <ul>
                {LANDING_CONTACT_INFO.map((item) => <li key={item}>{item}</li>)}
              </ul>
            </article>
          </div>

          <div className="landing-social-row">
            {LANDING_SOCIALS.map((item) => {
              const Icon = item.icon
              return (
                <button key={item.key} type="button" aria-label={item.label}>
                  <Icon size={15} />
                </button>
              )
            })}
          </div>

          <p className="landing-footer-copy">Copyright 2026 Todo Weather Forecast. All rights reserved.</p>
        </footer>

        {showAuthPanel ? (
          <div className="landing-auth-overlay" onClick={() => { if (!authLaunching) setShowAuthPanel(false) }}>
            <div className="landing-auth-shell auth-card" onClick={(event) => event.stopPropagation()}>
              <button type="button" className="landing-auth-close" onClick={() => setShowAuthPanel(false)} disabled={authLaunching}>x</button>
              <p className="auth-brand">TODO WEATHER</p>
              <h1>Login and Register</h1>
              <p className="auth-sub">Please login first. If you are new, create your account.</p>
              <div className="auth-toggle">
                <button type="button" className={authMode === 'login' ? 'active' : ''} onClick={() => { setLandingMood('sunny'); setAuthMode('login') }} disabled={authLaunching}>Login</button>
                <button type="button" className={authMode === 'register' ? 'active' : ''} onClick={() => { setLandingMood('rainy'); setAuthMode('register') }} disabled={authLaunching}>Register</button>
              </div>
              <form className="auth-form" onSubmit={onAuthSubmit} aria-busy={authLaunching}>
                {authMode === 'register' ? (
                  <label>Full Name<input type="text" value={authForm.name} onChange={(event) => setAuthForm((prev) => ({ ...prev, name: event.target.value }))} placeholder="Enter your full name" disabled={authLaunching} /></label>
                ) : null}
                <label>Email<input type="email" value={authForm.email} onChange={(event) => setAuthForm((prev) => ({ ...prev, email: event.target.value }))} placeholder="Enter your email" disabled={authLaunching} /></label>
                <label>Password<input type="password" value={authForm.password} onChange={(event) => setAuthForm((prev) => ({ ...prev, password: event.target.value }))} placeholder="Enter your password" disabled={authLaunching} /></label>
                {authMode === 'register' ? (
                  <label>Confirm Password<input type="password" value={authForm.confirmPassword} onChange={(event) => setAuthForm((prev) => ({ ...prev, confirmPassword: event.target.value }))} placeholder="Confirm your password" disabled={authLaunching} /></label>
                ) : null}
                {authError ? <p className="auth-error">{authError}</p> : null}
                <button type="submit" className="auth-submit" disabled={authLaunching}>{authLaunching ? 'Launching...' : (authMode === 'login' ? 'Login' : 'Register')}</button>
              </form>
            </div>
          </div>
        ) : null}
        {authLaunching ? (
          <div className="wx-launch-overlay" role="status" aria-live="polite">
            <div className="wx-launch-card">
              <span className="wx-launch-orbit" aria-hidden="true">
                <Loader2 size={20} className="spin" />
              </span>
              <p className="wx-launch-title">Launching Todo Weather Forecast</p>
              <p className="wx-launch-text">{launchStatus || 'Preparing your dashboard...'}</p>
              <div className="wx-launch-progress" aria-hidden="true"><span /></div>
            </div>
          </div>
        ) : null}
      </div>
    )
  }

  return (
    <div className="wx-page">
      <div className="wx-page-bg wx-page-bg--base" />
      <div className="wx-page-bg wx-page-bg--rays" />
      <div className="wx-page-bg wx-page-bg--clouds" />
      <div className="wx-page-particles" aria-hidden="true">
        {Array.from({ length: 14 }).map((_, idx) => (
          <span
            key={`wx-particle-${idx}`}
            style={{ left: `${(idx * 13) % 100}%`, animationDelay: `${(idx % 7) * 0.55}s` }}
          />
        ))}
      </div>
      <div className={`wx-board${dashboardIntro ? ' wx-board--intro' : ''}`}>
        <aside className="wx-sidebar">
          <div>
            <div className="wx-brand"><CloudSun size={18} /><span>TODO WEATHER</span></div>
            <nav className="wx-nav">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon
                return (
                  <button key={item.key} type="button" className={`wx-nav-item${activeSection === item.key ? ' active' : ''}`} onClick={() => { setActiveSection(item.key); setSectionError('') }}>
                    <Icon size={15} /><span>{item.label}</span>
                  </button>
                )
              })}
            </nav>
          </div>

          <article className="wx-today-card">
            <p className="wx-today-head"><CloudRain size={13} />Today</p>
            <p className="wx-today-time">{formatClockLabel(new Date())}</p>
            <h2>{formatTemp(current.temperature_c)}</h2>
            <p className="wx-today-city">{locationName}</p>
            <p className="wx-today-sub">{safeForecast.location?.country || '--'}</p>
            <div className="wx-today-stats">
              <p><span>Humidity:</span><strong>{displayRounded(current.humidity_percent, '%')}</strong></p>
              <p><span>Wind:</span><strong>{formatWind(current.wind_kph)}</strong></p>
            </div>
            <div className="wx-backend-result">
              <p className="wx-backend-title">Backend Result</p>
              <p><span>Rain Peak:</span><strong>{displayRounded(metrics.rain_peak_probability, '%')}</strong></p>
              <p><span>UV Peak:</span><strong>{displayRounded(metrics.uv_peak)}</strong></p>
              <p><span>Confidence:</span><strong>{displayRounded(quality.confidence_score, '%')}</strong></p>
              <p><span>Alerts:</span><strong>{displayRounded(safeForecast.alerts?.length || 0)}</strong></p>
            </div>
          </article>
        </aside>

        <main className="wx-main">
          <header className="wx-topbar">
            <form className="wx-search" onSubmit={(event) => { event.preventDefault(); setShowSuggestions(false); void runSearch() }} ref={searchRef}>
              <Search size={15} className="wx-search-icon" />
              <input
                value={query}
                onFocus={() => setShowSuggestions(true)}
                onChange={(event) => { setQuery(event.target.value); setSelectedLocation(null); setShowSuggestions(true) }}
                placeholder="Search district, city, country"
                aria-label="Search location"
              />
              {showSuggestions && (query.trim().length >= 2 || loadingSuggestions) ? (
                <div className="wx-suggestions">
                  {loadingSuggestions ? (
                    <p className="wx-suggestions-empty"><Loader2 size={13} className="spin" />Finding locations...</p>
                  ) : suggestions.length ? (
                    suggestions.slice(0, 8).map((item) => (
                      <button key={`${item.latitude}-${item.longitude}`} type="button" onClick={() => {
                        const label = mapSuggestionLabel(item)
                        const picked = { name: label, latitude: item.latitude, longitude: item.longitude, timezone: item.timezone || 'auto' }
                        setQuery(label)
                        setSelectedLocation(picked)
                        setSuggestions([])
                        setShowSuggestions(false)
                        void requestForecast({ location: picked })
                      }}>
                        {mapSuggestionLabel(item)}
                      </button>
                    ))
                  ) : (
                    <p className="wx-suggestions-empty">No matching location found.</p>
                  )}
                </div>
              ) : null}
            </form>

            <div className="wx-top-actions">
              <button type="button" className="wx-location-btn" onClick={() => void onUseMyLocation()} disabled={locating || loadingForecast}>
                <Navigation size={13} />{locating ? 'Locating...' : 'Use My Location'}
              </button>
              <div className="wx-notification-wrap" ref={notificationRef}>
                <button
                  type="button"
                  className={`wx-bell-btn${notificationsOpen ? ' active' : ''}`}
                  aria-label="Notifications"
                  onClick={onToggleNotifications}
                >
                  <Bell size={14} />
                  {uiSettings.notifications_enabled && unreadNotifications > 0 ? (
                    <span className="wx-bell-badge">{Math.min(unreadNotifications, 9)}</span>
                  ) : null}
                </button>
                {notificationsOpen ? (
                  <div className="wx-notification-panel">
                    <div className="wx-notification-head">
                      <h4>Notifications</h4>
                      <button
                        type="button"
                        onClick={() => void loadNotifications(true)}
                        disabled={notificationsLoading || !uiSettings.notifications_enabled}
                      >
                        {notificationsLoading ? 'Loading...' : 'Refresh'}
                      </button>
                    </div>
                    {notificationsPayload.generated_at_utc ? (
                      <p className="wx-notification-time">
                        Updated {formatNotificationLabel(notificationsPayload.generated_at_utc)}
                      </p>
                    ) : null}
                    {!uiSettings.notifications_enabled ? <p className="wx-slide-empty">Notifications are disabled in settings.</p> : null}
                    {uiSettings.notifications_enabled && notificationsError ? <p className="wx-error">{notificationsError}</p> : null}
                    {uiSettings.notifications_enabled && notificationsLoading ? (
                      <p className="wx-slide-loading"><Loader2 size={14} className="spin" />Loading notifications...</p>
                    ) : null}
                    {uiSettings.notifications_enabled && !notificationsLoading && !notificationsError ? (
                      notificationItems.length ? (
                        <div className="wx-notification-list">
                          {notificationItems.map((item, idx) => {
                            const severity = String(item.severity || 'low').toLowerCase()
                            return (
                              <article key={`${item.id || 'notification'}-${idx}`} className={`wx-notification-item ${severity}`}>
                                <strong>{item.title || 'Weather Update'}</strong>
                                <p>{item.message || 'Forecast updated.'}</p>
                                <small>{formatNotificationLabel(item.timestamp_utc)}</small>
                              </article>
                            )
                          })}
                        </div>
                      ) : (
                        <p className="wx-slide-empty">No notifications right now.</p>
                      )
                    ) : null}
                  </div>
                ) : null}
              </div>
              <div className="wx-profile">
                <img src="https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=120&q=80" alt="User avatar" />
                <span>{currentUser?.name || 'User'}</span>
                <ChevronDown size={13} />
              </div>
              <button type="button" className="wx-logout-btn" onClick={onLogout}><LogOut size={13} />Logout</button>
            </div>
          </header>

          {welcomeMessage ? <p className="wx-welcome-banner">{welcomeMessage}</p> : null}

          {activeSection === 'dashboard' ? (
            <>
              <section className="wx-section">
                <h1>Todo Weather Forecast</h1>
                <div className="wx-city-row">
                  {CITY_CARDS.map((city) => (
                    <button key={city.name} type="button" className="wx-city-card" onClick={() => {
                      setQuery(city.name)
                      setSelectedLocation(null)
                      setShowSuggestions(false)
                      void requestForecast({ locationQuery: city.name })
                    }}>
                      <div className="wx-city-thumb-wrap">
                        <img src={city.image} alt={city.name} loading="lazy" className="wx-city-thumb" />
                        <span className="wx-city-time">{formatClockLabel(new Date())}</span>
                      </div>
                      <p>{city.name}</p>
                    </button>
                  ))}
                </div>
              </section>
              <section className="wx-location-row">
                <h2>{locationName}</h2>
                <button type="button" onClick={() => void runSearch()}>{loadingForecast ? 'Refreshing...' : 'Details more'} <span>-&gt;</span></button>
              </section>
              <section className="wx-days-row">
                {daily.map((day) => {
                  const DayIcon = weatherIconFromLabel(day.weather)
                  return (
                    <article key={day.date} className="wx-day-chip">
                      <DayIcon size={16} />
                      <p>{`${formatTemp(day.temp_min_c)}/${formatTemp(day.temp_max_c)}`}</p>
                      <small>{formatDateLabel(day.date)}</small>
                    </article>
                  )
                })}
                <button type="button" className="wx-day-next">&gt;&gt;</button>
              </section>
              <section className="wx-graph-card">
                <div className="wx-graph-head">
                  <h3>Weather Graph</h3>
                  <div className="wx-graph-tabs">
                    <button type="button">Day</button>
                    <button type="button" className="active">Week</button>
                    <button type="button">10 Days</button>
                    <button type="button">15 Days</button>
                  </div>
                </div>
                <div className="wx-graph-meta">
                  <span><WeatherIcon size={14} />{`Temperature (${tempUnit})`}</span>
                  <span><Wind size={14} />Wind</span>
                  {loadingForecast ? <strong><Loader2 size={13} className="spin" />Updating</strong> : <strong>Live</strong>}
                </div>
                <div className="wx-chart-wrap">
                  <svg viewBox="0 0 760 260" preserveAspectRatio="none">
                    <line className="wx-grid" x1="0" y1="230" x2="760" y2="230" />
                    <line className="wx-grid" x1="0" y1="190" x2="760" y2="190" />
                    <line className="wx-grid" x1="0" y1="150" x2="760" y2="150" />
                    <line className="wx-grid" x1="0" y1="110" x2="760" y2="110" />
                    <line className="wx-grid" x1="0" y1="70" x2="760" y2="70" />
                    <polyline points={chartSeries.blue} className="wx-line-blue" />
                    <polyline points={chartSeries.red} className="wx-line-red" />
                    {chartSeries.dots.map((dot, idx) => (
                      <circle key={`dot-${dot.x}-${dot.y}`} cx={dot.x} cy={dot.y} r={idx % 3 === 0 ? 4.5 : 3.2} className={idx < Math.max(4, Math.floor(chartSeries.dots.length / 2)) ? 'wx-dot-blue' : 'wx-dot-red'} />
                    ))}
                  </svg>
                  <div className="wx-chart-hours">{hourly.map((item) => <span key={item.time}>{formatHourLabel(item.time)}</span>)}</div>
                </div>
              </section>
              {renderVoiceAssistantCard()}
            </>
          ) : (
            <section className="wx-slide-shell">
              <div className="wx-slide-head">
                <h1>{slidePayload?.title || `${activeNav?.label || activeSection} Slide`}</h1>
                <button type="button" className="wx-slide-refresh" onClick={() => void loadSection(activeSection, true)} disabled={sectionLoading}>
                  {sectionLoading ? <><Loader2 size={14} className="spin" />Refreshing...</> : 'Refresh Slide'}
                </button>
              </div>
              {sectionError ? <p className="wx-error">{sectionError}</p> : null}
              {!sectionError && sectionLoading ? <p className="wx-slide-loading"><Loader2 size={14} className="spin" />Loading slide...</p> : null}
              {!sectionError && !sectionLoading ? (
                activeSection === 'statistics' ? renderStatisticsFeatures() : (
                  activeSection === 'map' ? renderMapFeatures() : (
                    activeSection === 'calendar' ? renderCalendarFeatures() : (
                      activeSection === 'setting' ? renderSettingFeatures() : (
                        <div className="wx-slide-card">
                          <p className="wx-slide-empty">No feature available for this section.</p>
                        </div>
                      )
                    )
                  )
                )
              ) : null}
            </section>
          )}

          {error ? <p className="wx-error">{error}</p> : null}
        </main>
      </div>
    </div>
  )
}

export default App
