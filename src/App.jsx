import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  ArrowRight,
  BellRing,
  Bot,
  BrainCircuit,
  CalendarDays,
  CheckCircle2,
  CloudRain,
  CloudSun,
  Gauge,
  Loader2,
  MapPin,
  PlayCircle,
  Sparkles,
  Sun,
  Wind,
} from 'lucide-react'
import { fetchHealth, fetchPersonalizedForecast } from './api/weatherApi'

const features = [
  {
    title: 'Personalized Forecast Engine',
    description: 'Learns your schedule and adapts forecast timing to your routine.',
    icon: BrainCircuit,
  },
  {
    title: 'Hyper-Local Predictions',
    description: 'Street-level weather intelligence tuned to your location.',
    icon: MapPin,
  },
  {
    title: 'Smart Alerts',
    description: 'Action-oriented notifications only when needed.',
    icon: BellRing,
  },
  {
    title: 'AI Activity Planner',
    description: 'Suggests ideal windows for workout and travel.',
    icon: CalendarDays,
  },
]

const steps = [
  ['Connect Location', 'Securely connect your location and key places.'],
  ['Set Daily Routine', 'Set wake, commute, workout, and sleep times.'],
  ['AI Learns Behavior', 'Behavior model evaluates weather impact patterns.'],
  ['Get Recommendations', 'Receive decisions instead of raw weather data.'],
]

const fallbackHourly = [
  { time: '09:00', temperature_c: 22, wind_kph: 12, rain_probability: 18, uv_index: 2 },
  { time: '10:00', temperature_c: 24, wind_kph: 14, rain_probability: 22, uv_index: 3 },
  { time: '11:00', temperature_c: 26, wind_kph: 18, rain_probability: 30, uv_index: 4 },
  { time: '12:00', temperature_c: 27, wind_kph: 16, rain_probability: 42, uv_index: 6 },
  { time: '13:00', temperature_c: 25, wind_kph: 24, rain_probability: 38, uv_index: 5 },
  { time: '14:00', temperature_c: 23, wind_kph: 20, rain_probability: 28, uv_index: 3 },
  { time: '15:00', temperature_c: 22, wind_kph: 14, rain_probability: 20, uv_index: 2 },
  { time: '16:00', temperature_c: 21, wind_kph: 12, rain_probability: 16, uv_index: 1 },
]

const fallbackInsights = [
  {
    title: 'Commute risk',
    time: '08:30',
    severity: 'attention',
    summary: 'Morning showers likely. Keep rain gear ready.',
    action: 'Carry a compact umbrella.',
  },
  {
    title: 'Workout planner',
    time: '18:00',
    severity: 'safe',
    summary: 'Wind and UV levels are low in the evening.',
    action: 'Good outdoor training window.',
  },
  {
    title: 'Morning readiness',
    time: '07:00',
    severity: 'safe',
    summary: 'Comfortable morning conditions expected.',
    action: 'Comfortable start expected.',
  },
]

const fallbackDaily = [
  { date: '2026-02-19', weather: 'Partly cloudy', temp_max_c: 28, temp_min_c: 20, rain_probability_max: 42 },
  { date: '2026-02-20', weather: 'Clear sky', temp_max_c: 30, temp_min_c: 21, rain_probability_max: 24 },
  { date: '2026-02-21', weather: 'Light rain', temp_max_c: 27, temp_min_c: 19, rain_probability_max: 58 },
]

function formatHour(input) {
  if (!input) return '--:--'
  const date = new Date(input)
  if (Number.isNaN(date.getTime())) return String(input).slice(-5)
  return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`
}

function formatDate(input) {
  const date = new Date(input)
  if (Number.isNaN(date.getTime())) return input
  return date.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })
}

function severityPillClass(severity) {
  if (severity === 'warning') return 'border-rose-300 bg-rose-50 text-rose-700'
  if (severity === 'attention') return 'border-amber-300 bg-amber-50 text-amber-700'
  return 'border-emerald-300 bg-emerald-50 text-emerald-700'
}

function App() {
  const [apiStatus, setApiStatus] = useState('Checking API...')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [updatedAt, setUpdatedAt] = useState('')
  const [forecast, setForecast] = useState(null)
  const [parallax, setParallax] = useState({ x: 0, y: 0 })
  const [form, setForm] = useState({
    locationQuery: 'New York',
    wakeTime: '07:00',
    commuteTime: '08:30',
    workoutTime: '18:00',
    sleepTime: '22:30',
    outdoorCommute: true,
    uvSensitive: true,
    airQualitySensitive: false,
  })

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const result = await fetchHealth()
        setApiStatus(`${result.status.toUpperCase()} (${result.service})`)
      } catch {
        setApiStatus('Offline - start backend on http://localhost:8000')
      }
    }
    void checkHealth()
  }, [])

  const requestForecast = async () => {
    setLoading(true)
    setError('')
    try {
      const result = await fetchPersonalizedForecast({
        location_query: form.locationQuery,
        routine: {
          wake_time: form.wakeTime,
          commute_time: form.commuteTime,
          workout_time: form.workoutTime,
          sleep_time: form.sleepTime,
        },
        preferences: {
          outdoor_commute: form.outdoorCommute,
          uv_sensitive: form.uvSensitive,
          air_quality_sensitive: form.airQualitySensitive,
        },
      })
      setForecast(result)
      setUpdatedAt(new Date().toLocaleTimeString())
    } catch (requestError) {
      setError(requestError.message || 'Unable to load personalized forecast')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void requestForecast()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const current = forecast?.current ?? {
    temperature_c: 24,
    humidity_percent: 62,
    weather: 'Partly cloudy',
    wind_kph: 16,
    aqi_us: 47,
  }
  const metrics = forecast?.metrics ?? {
    rain_peak_probability: 42,
    uv_peak: 6.2,
    wind_peak_kph: 24,
  }
  const hourly = (forecast?.hourly?.length ? forecast.hourly : fallbackHourly).slice(0, 8)
  const daily = (forecast?.daily?.length ? forecast.daily : fallbackDaily).slice(0, 3)
  const insights = forecast?.insights?.length ? forecast.insights : fallbackInsights
  const actions = forecast?.recommended_actions?.length
    ? forecast.recommended_actions
    : fallbackInsights.map((item) => item.action)
  const activityWindow = forecast?.activity_windows?.[0]

  const tempBars = useMemo(() => {
    const temps = hourly.map((item) => item.temperature_c || 0)
    const min = Math.min(...temps)
    const max = Math.max(...temps)
    const spread = Math.max(1, max - min)
    return hourly.map((item) => {
      const value = item.temperature_c || min
      return {
        label: formatHour(item.time),
        value,
        height: 18 + ((value - min) / spread) * 90,
      }
    })
  }, [hourly])

  const windPolyline = useMemo(() => {
    const values = hourly.map((item) => item.wind_kph || 0)
    const max = Math.max(1, ...values)
    return values
      .map((value, index) => {
        const x = index * 44 + 8
        const y = 98 - (value / max) * 76
        return `${x},${y}`
      })
      .join(' ')
  }, [hourly])

  return (
    <div className="relative overflow-x-clip pb-12">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="ambient-shape ambient-shape--one left-[-7rem] top-20 animate-drift" />
        <div className="ambient-shape ambient-shape--two right-[-8rem] top-[32rem] animate-drift" />
        <div className="ambient-shape ambient-shape--three left-[46%] top-[70rem] animate-pulse-soft" />
      </div>

      <header className="mx-auto max-w-6xl px-4 pt-7 sm:px-6 lg:px-8">
        <nav className="glass-card flex items-center justify-between gap-6 px-5 py-4 sm:px-6">
          <a href="#home" className="font-display text-xl font-bold tracking-tight text-brand-ink">
            NimbusIQ
          </a>
          <div className="hidden items-center gap-8 text-sm font-semibold text-slate-600 md:flex">
            <a className="transition hover:text-brand-blue" href="#features">Features</a>
            <a className="transition hover:text-brand-blue" href="#dashboard">Dashboard</a>
            <a className="transition hover:text-brand-blue" href="#how-it-works">How It Works</a>
            <a className="transition hover:text-brand-blue" href="#premium">Premium</a>
          </div>
          <a href="#dashboard" className="btn-primary px-5 py-2.5 text-xs sm:text-sm">Live Demo</a>
        </nav>
      </header>

      <main>
        <section
          id="home"
          className="mx-auto grid max-w-6xl gap-12 px-4 pb-8 pt-16 sm:px-6 md:grid-cols-2 md:items-center md:pt-20 lg:px-8"
          onMouseMove={(event) => {
            const rect = event.currentTarget.getBoundingClientRect()
            const x = ((event.clientX - rect.left) / rect.width - 0.5) * 18
            const y = ((event.clientY - rect.top) / rect.height - 0.5) * 18
            setParallax({ x, y })
          }}
          onMouseLeave={() => setParallax({ x: 0, y: 0 })}
        >
          <div className="space-y-8 animate-reveal">
            <span className="inline-flex items-center gap-2 rounded-full border border-brand-blue/20 bg-brand-blue/10 px-4 py-2 text-xs font-bold uppercase tracking-[0.16em] text-brand-blue">
              <Sparkles size={14} />
              AI-Powered Weather Intelligence
            </span>
            <div className="space-y-4">
              <h1 className="font-display text-4xl font-extrabold leading-[0.95] text-brand-ink sm:text-5xl lg:text-6xl">
                SMART AI WEATHER
                <span className="block bg-gradient-to-r from-brand-blue via-brand-sky to-brand-violet bg-clip-text text-transparent">
                  INTELLIGENCE
                </span>
              </h1>
              <p className="max-w-xl text-base leading-relaxed text-slate-600 sm:text-lg">
                Personalized weather forecasts powered by AI. Get decisions, not just data.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <a href="#dashboard" className="btn-primary">Get Started <ArrowRight size={16} /></a>
              <a href="#dashboard" className="btn-secondary"><PlayCircle size={16} />View Live Forecast</a>
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="glass-card p-4 text-sm"><p className="font-semibold text-slate-500">Forecast Accuracy</p><p className="mt-2 text-xl font-bold text-brand-ink">96.4%</p></div>
              <div className="glass-card p-4 text-sm"><p className="font-semibold text-slate-500">API Status</p><p className="mt-2 text-sm font-bold text-brand-ink">{apiStatus}</p></div>
              <div className="glass-card p-4 text-sm"><p className="font-semibold text-slate-500">Smart Alerts</p><p className="mt-2 text-xl font-bold text-brand-ink">Real-time</p></div>
            </div>
          </div>

          <div className="relative animate-reveal" style={{ animationDelay: '160ms', transform: `translate3d(${parallax.x * 0.35}px, ${parallax.y * 0.35}px, 0)` }}>
            <div className="glass-card relative overflow-hidden px-6 pb-7 pt-8 sm:px-8 sm:pb-8">
              <div className="absolute -right-12 -top-12 h-44 w-44 rounded-full bg-brand-blue/25 blur-3xl animate-pulse-soft" />
              <div className="absolute -bottom-12 left-8 h-44 w-44 rounded-full bg-brand-violet/30 blur-3xl animate-pulse-soft [animation-delay:1.3s]" />
              <div className="relative mx-auto flex h-[22rem] max-w-sm items-center justify-center">
                <div className="absolute h-56 w-56 rounded-[40%] bg-gradient-to-br from-brand-blue to-brand-violet shadow-glow" style={{ transform: `rotate(8deg) translateX(${parallax.x * 0.2}px)` }} />
                <div className="absolute h-64 w-64 rounded-full border border-white/40 bg-white/20 backdrop-blur-md" />
                <div className="relative z-10 flex h-32 w-32 items-center justify-center rounded-3xl border border-white/40 bg-white/70 shadow-xl animate-float"><Bot className="h-14 w-14 text-brand-blue" strokeWidth={1.7} /></div>
                <CloudSun className="absolute right-16 top-16 h-10 w-10 text-brand-sky animate-pulse-soft" />
                <CloudRain className="absolute bottom-14 left-14 h-10 w-10 text-brand-violet animate-float" />
              </div>
              <div className="floating-chip left-4 top-5 [animation-delay:250ms]"><p className="text-[0.65rem] uppercase tracking-[0.12em] text-slate-500">Temperature</p><p className="mt-1 text-lg font-bold text-brand-ink">{Math.round(current.temperature_c)}C</p></div>
              <div className="floating-chip -right-3 top-20 [animation-delay:900ms]"><p className="text-[0.65rem] uppercase tracking-[0.12em] text-slate-500">Rain Chance</p><p className="mt-1 text-lg font-bold text-brand-ink">{Math.round(metrics.rain_peak_probability)}%</p></div>
              <div className="floating-chip bottom-6 left-10 [animation-delay:1.2s]"><p className="text-[0.65rem] uppercase tracking-[0.12em] text-slate-500">Wind</p><p className="mt-1 text-lg font-bold text-brand-ink">{Math.round(current.wind_kph)} km/h</p></div>
            </div>
          </div>
        </section>

        <section id="features" className="mx-auto max-w-6xl px-4 py-14 sm:px-6 lg:px-8">
          <div className="space-y-3">
            <h2 className="section-title">AI That Understands Your Routine</h2>
            <p className="section-copy">The platform blends behavior modeling with live atmosphere data to deliver practical daily recommendations.</p>
          </div>
          <div className="mt-8 grid gap-4 sm:mt-10 md:grid-cols-2 xl:grid-cols-4">
            {features.map((item, index) => {
              const Icon = item.icon
              return (
                <article key={item.title} className="feature-card animate-reveal" style={{ animationDelay: `${index * 120}ms` }}>
                  <div className="mb-5 inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-blue/25 to-brand-violet/25 text-brand-blue"><Icon size={22} /></div>
                  <h3 className="text-lg font-bold text-brand-ink">{item.title}</h3>
                  <p className="mt-3 text-sm leading-relaxed text-slate-600">{item.description}</p>
                </article>
              )
            })}
          </div>
        </section>

        <section id="dashboard" className="mx-auto max-w-6xl px-4 py-14 sm:px-6 lg:px-8">
          <div className="space-y-3">
            <h2 className="section-title">Live Weather Dashboard Preview</h2>
            <p className="section-copy">Live backend calls + AI recommendations. Update routine inputs and generate your personalized forecast.</p>
          </div>

          <div className="glass-card mt-8 overflow-hidden p-5 sm:p-8">
            <div className="grid gap-8 lg:grid-cols-[1.1fr_1fr]">
              <div>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold uppercase tracking-[0.12em] text-slate-500">Current Conditions</p>
                    <h3 className="mt-1 text-3xl font-bold text-brand-ink">{Math.round(current.temperature_c)}C, {current.weather}</h3>
                    <p className="mt-1 text-sm text-slate-600">{forecast?.location?.name || form.locationQuery} | Humidity {Math.round(current.humidity_percent || 0)}%</p>
                  </div>
                  <span className="rounded-full border border-emerald-300/70 bg-emerald-100 px-3 py-1 text-xs font-bold text-emerald-700">{updatedAt ? `Updated ${updatedAt}` : 'Waiting for first sync'}</span>
                </div>

                <form className="mt-6 rounded-2xl border border-white/70 bg-white/80 p-5" onSubmit={(event) => { event.preventDefault(); void requestForecast() }}>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <label className="flex flex-col gap-2 text-sm font-semibold text-slate-700 sm:col-span-2">Location
                      <input type="text" required value={form.locationQuery} onChange={(event) => setForm((prev) => ({ ...prev, locationQuery: event.target.value }))} className="rounded-xl border border-slate-300 bg-white px-3 py-2 font-medium outline-none transition focus:border-brand-blue" />
                    </label>
                    <label className="flex flex-col gap-2 text-sm font-semibold text-slate-700">Wake Time
                      <input type="time" value={form.wakeTime} onChange={(event) => setForm((prev) => ({ ...prev, wakeTime: event.target.value }))} className="rounded-xl border border-slate-300 bg-white px-3 py-2 font-medium outline-none transition focus:border-brand-blue" />
                    </label>
                    <label className="flex flex-col gap-2 text-sm font-semibold text-slate-700">Commute Time
                      <input type="time" value={form.commuteTime} onChange={(event) => setForm((prev) => ({ ...prev, commuteTime: event.target.value }))} className="rounded-xl border border-slate-300 bg-white px-3 py-2 font-medium outline-none transition focus:border-brand-blue" />
                    </label>
                    <label className="flex flex-col gap-2 text-sm font-semibold text-slate-700">Workout Time
                      <input type="time" value={form.workoutTime} onChange={(event) => setForm((prev) => ({ ...prev, workoutTime: event.target.value }))} className="rounded-xl border border-slate-300 bg-white px-3 py-2 font-medium outline-none transition focus:border-brand-blue" />
                    </label>
                    <label className="flex flex-col gap-2 text-sm font-semibold text-slate-700">Sleep Time
                      <input type="time" value={form.sleepTime} onChange={(event) => setForm((prev) => ({ ...prev, sleepTime: event.target.value }))} className="rounded-xl border border-slate-300 bg-white px-3 py-2 font-medium outline-none transition focus:border-brand-blue" />
                    </label>
                  </div>
                  <div className="mt-4 grid gap-2 text-sm text-slate-700 sm:grid-cols-3">
                    <label className="inline-flex items-center gap-2"><input type="checkbox" checked={form.outdoorCommute} onChange={(event) => setForm((prev) => ({ ...prev, outdoorCommute: event.target.checked }))} />Outdoor commute</label>
                    <label className="inline-flex items-center gap-2"><input type="checkbox" checked={form.uvSensitive} onChange={(event) => setForm((prev) => ({ ...prev, uvSensitive: event.target.checked }))} />UV sensitive</label>
                    <label className="inline-flex items-center gap-2"><input type="checkbox" checked={form.airQualitySensitive} onChange={(event) => setForm((prev) => ({ ...prev, airQualitySensitive: event.target.checked }))} />AQI sensitive</label>
                  </div>
                  <div className="mt-4 flex flex-wrap items-center gap-3">
                    <button type="submit" className="btn-primary" disabled={loading}>{loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles size={16} />}{loading ? 'Generating...' : 'Generate AI Forecast'}</button>
                    {error ? <p className="text-sm font-semibold text-rose-600">{error}</p> : null}
                  </div>
                </form>

                <div className="mt-4 grid gap-4 sm:grid-cols-3">
                  <article className="rounded-2xl border border-white/70 bg-white/80 p-4"><div className="flex items-center gap-2 text-brand-blue"><Sun size={16} /><p className="text-xs font-semibold uppercase tracking-[0.1em]">UV Peak</p></div><p className="mt-2 text-2xl font-bold text-brand-ink">{Number(metrics.uv_peak || 0).toFixed(1)}</p></article>
                  <article className="rounded-2xl border border-white/70 bg-white/80 p-4"><div className="flex items-center gap-2 text-brand-violet"><Activity size={16} /><p className="text-xs font-semibold uppercase tracking-[0.1em]">AQI</p></div><p className="mt-2 text-2xl font-bold text-brand-ink">{Math.round(current.aqi_us || 0)}</p></article>
                  <article className="rounded-2xl border border-white/70 bg-white/80 p-4"><div className="flex items-center gap-2 text-cyan-600"><CloudRain size={16} /><p className="text-xs font-semibold uppercase tracking-[0.1em]">Rain Peak</p></div><p className="mt-2 text-2xl font-bold text-brand-ink">{Math.round(metrics.rain_peak_probability)}%</p></article>
                </div>
              </div>

              <div className="space-y-4">
                <article className="rounded-2xl border border-white/70 bg-white/80 p-4">
                  <div className="flex items-center justify-between"><p className="text-sm font-semibold text-slate-600">Hourly Forecast</p><Gauge size={16} className="text-brand-blue" /></div>
                  <div className="mt-4 flex h-36 items-end gap-2">
                    {tempBars.map((item) => (
                      <div key={item.label} className="flex flex-1 flex-col items-center gap-2">
                        <div className="w-full rounded-md bg-gradient-to-t from-brand-blue to-brand-violet" style={{ height: `${item.height}px` }} />
                        <p className="text-[0.62rem] font-semibold text-slate-500">{item.label}</p>
                      </div>
                    ))}
                  </div>
                </article>
                <article className="rounded-2xl border border-white/70 bg-white/80 p-4">
                  <div className="flex items-center justify-between"><p className="text-sm font-semibold text-slate-600">Wind Speed Trend</p><Wind size={16} className="text-brand-violet" /></div>
                  <svg viewBox="0 0 320 110" className="mt-4 w-full"><polyline fill="none" stroke="#7C3AED" strokeWidth="4" strokeLinecap="round" points={windPolyline} /></svg>
                </article>
                <article className="rounded-2xl border border-white/70 bg-white/80 p-4">
                  <p className="text-sm font-semibold text-slate-600">Rain Probability</p>
                  <div className="mt-3 flex items-end gap-2">
                    {hourly.map((item) => (
                      <div key={`rain-${item.time}`} className="w-full rounded-sm bg-gradient-to-t from-cyan-500 to-cyan-300" style={{ height: `${Math.max(8, (item.rain_probability || 0) * 1.05)}px` }} />
                    ))}
                  </div>
                </article>
              </div>
            </div>

            <div className="mt-6 grid gap-4 lg:grid-cols-2">
              <article className="rounded-2xl border border-white/70 bg-white/80 p-5">
                <p className="text-sm font-semibold uppercase tracking-[0.1em] text-slate-500">Recommended Actions</p>
                <div className="mt-4 space-y-3">{actions.slice(0, 4).map((item, idx) => <p key={`${item}-${idx}`} className="flex items-start gap-2 text-sm text-slate-700"><CheckCircle2 className="mt-0.5 h-4 w-4 text-brand-blue" />{item}</p>)}</div>
              </article>
              <article className="rounded-2xl border border-white/70 bg-white/80 p-5">
                <p className="text-sm font-semibold uppercase tracking-[0.1em] text-slate-500">Best Activity Window</p>
                <p className="mt-3 text-2xl font-bold text-brand-ink">{activityWindow?.best_start || '--:--'} - {activityWindow?.best_end || '--:--'}</p>
                <p className="mt-2 text-sm text-slate-600">{activityWindow?.reason || 'Generate forecast to unlock personalized activity timing.'}</p>
              </article>
            </div>

            <div className="mt-6 grid gap-4 lg:grid-cols-3">
              {insights.map((item) => (
                <article key={`${item.title}-${item.time}`} className="rounded-2xl border border-white/70 bg-white/80 p-5">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-base font-bold text-brand-ink">{item.title}</p>
                    <span className={`rounded-full border px-2 py-1 text-[0.7rem] font-bold uppercase ${severityPillClass(item.severity)}`}>{item.severity}</span>
                  </div>
                  <p className="mt-2 text-sm font-semibold text-slate-500">Around {item.time}</p>
                  <p className="mt-2 text-sm leading-relaxed text-slate-600">{item.summary}</p>
                  <p className="mt-2 text-sm font-semibold text-brand-blue">{item.action}</p>
                </article>
              ))}
            </div>

            <div className="mt-6 rounded-2xl border border-white/70 bg-white/80 p-5">
              <p className="text-sm font-semibold uppercase tracking-[0.1em] text-slate-500">3-Day Outlook</p>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                {daily.map((day) => (
                  <article key={day.date} className="rounded-xl border border-slate-200 bg-white p-4">
                    <p className="text-sm font-bold text-brand-ink">{formatDate(day.date)}</p>
                    <p className="mt-2 text-sm text-slate-600">{day.weather}</p>
                    <p className="mt-2 text-sm font-semibold text-slate-700">{Math.round(day.temp_min_c)}C - {Math.round(day.temp_max_c)}C</p>
                    <p className="mt-1 text-xs text-slate-500">Rain Max: {Math.round(day.rain_probability_max || 0)}%</p>
                  </article>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section id="how-it-works" className="mx-auto max-w-6xl px-4 py-14 sm:px-6 lg:px-8">
          <div className="space-y-3">
            <h2 className="section-title">How It Works</h2>
            <p className="section-copy">A four-step onboarding path that turns weather data into decisions in minutes.</p>
          </div>
          <div className="relative mt-10 grid gap-4 md:grid-cols-4">
            <div className="pointer-events-none absolute left-[12%] right-[12%] top-10 hidden h-px bg-gradient-to-r from-brand-blue via-brand-sky to-brand-violet md:block" />
            {steps.map((step, index) => (
              <article key={step[0]} className="timeline-step animate-reveal" style={{ animationDelay: `${index * 130}ms` }}>
                <div className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-brand-blue to-brand-violet text-sm font-bold text-white">0{index + 1}</div>
                <h3 className="text-base font-bold text-brand-ink">{step[0]}</h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-600">{step[1]}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="premium" className="mx-auto max-w-6xl px-4 py-14 sm:px-6 lg:px-8">
          <div className="relative overflow-hidden rounded-[2rem] bg-gradient-to-br from-slate-950 via-blue-950 to-violet-950 p-7 text-white shadow-[0_30px_65px_-32px_rgba(15,23,42,0.9)] sm:p-10">
            <div className="pointer-events-none absolute -right-20 -top-20 h-60 w-60 rounded-full bg-brand-violet/30 blur-3xl" />
            <div className="pointer-events-none absolute -bottom-24 left-[-3rem] h-64 w-64 rounded-full bg-brand-sky/20 blur-3xl" />
            <div className="relative grid gap-9 lg:grid-cols-[1.2fr_1fr] lg:items-center">
              <div>
                <p className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.14em]"><Sparkles size={14} />Premium Intelligence</p>
                <h2 className="mt-4 font-display text-3xl font-bold leading-tight sm:text-4xl">Advanced Weather Analytics for Personal and Business Decisions</h2>
                <p className="mt-4 max-w-xl text-sm leading-relaxed text-blue-100 sm:text-base">Unlock predictive insights, travel/event optimization, and API access for weather-aware operations.</p>
                <div className="mt-7 space-y-3">
                  <p className="flex items-center gap-2 text-sm"><CheckCircle2 className="h-4 w-4 text-cyan-300" />AI predictive insights with hourly confidence scoring.</p>
                  <p className="flex items-center gap-2 text-sm"><CheckCircle2 className="h-4 w-4 text-cyan-300" />Travel and event optimization recommendations.</p>
                  <p className="flex items-center gap-2 text-sm"><CheckCircle2 className="h-4 w-4 text-cyan-300" />Business API for weather-driven workflows.</p>
                </div>
                <a href="#home" className="mt-7 inline-flex btn-primary">Unlock Premium <ArrowRight size={16} /></a>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                <article className="rounded-2xl border border-white/15 bg-white/10 p-5 backdrop-blur-lg"><p className="text-xs font-semibold uppercase tracking-[0.1em] text-blue-100">Route Risk</p><p className="mt-2 text-3xl font-bold">-31%</p></article>
                <article className="rounded-2xl border border-white/15 bg-white/10 p-5 backdrop-blur-lg"><p className="text-xs font-semibold uppercase tracking-[0.1em] text-blue-100">Forecast Confidence</p><p className="mt-2 text-3xl font-bold">98.2%</p></article>
                <article className="rounded-2xl border border-white/15 bg-white/10 p-5 backdrop-blur-lg sm:col-span-2"><p className="text-xs font-semibold uppercase tracking-[0.1em] text-blue-100">Enterprise API Calls</p><p className="mt-2 text-3xl font-bold">4.8M / month</p><p className="mt-2 text-sm text-blue-100">Secure, low-latency API for logistics, tourism, and event teams.</p></article>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="mx-auto mt-8 flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 pb-4 pt-2 text-sm text-slate-500 sm:px-6 lg:px-8">
        <p>2026 NimbusIQ Weather Intelligence</p>
        <div className="flex items-center gap-5 font-semibold">
          <a href="#features" className="transition hover:text-brand-blue">Platform</a>
          <a href="#premium" className="transition hover:text-brand-blue">Pricing</a>
          <a href="#home" className="transition hover:text-brand-blue">Contact</a>
        </div>
      </footer>
    </div>
  )
}

export default App
