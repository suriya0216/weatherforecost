const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')

async function fetchWithTimeout(url, options = {}, timeoutMs = 45000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw new Error('Request timed out. Please try again.')
    }
    throw error
  } finally {
    clearTimeout(timer)
  }
}

async function parseJson(response) {
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = payload?.detail || 'Request failed'
    throw new Error(message)
  }
  return payload
}

export async function fetchHealth() {
  const response = await fetchWithTimeout(`${API_BASE_URL}/health`, {}, 10000)
  return parseJson(response)
}

export async function fetchPersonalizedForecast(requestBody) {
  const response = await fetchWithTimeout(`${API_BASE_URL}/forecast/personalized`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
  }, 60000)
  return parseJson(response)
}

export async function fetchGeocodeSuggestions(query) {
  const params = new URLSearchParams({ query })
  const response = await fetchWithTimeout(`${API_BASE_URL}/geocode?${params.toString()}`, {}, 15000)
  return parseJson(response)
}

export async function fetchReverseGeocode(latitude, longitude) {
  const params = new URLSearchParams({
    latitude: String(latitude),
    longitude: String(longitude),
  })
  const response = await fetchWithTimeout(`${API_BASE_URL}/geocode/reverse?${params.toString()}`, {}, 15000)
  return parseJson(response)
}

export async function fetchSidebarSection(section, options = {}) {
  const params = new URLSearchParams()
  if (options?.locationQuery) {
    params.set('location_query', String(options.locationQuery))
  }
  if (options?.location?.latitude !== undefined && options?.location?.latitude !== null) {
    params.set('latitude', String(options.location.latitude))
  }
  if (options?.location?.longitude !== undefined && options?.location?.longitude !== null) {
    params.set('longitude', String(options.location.longitude))
  }
  if (options?.location?.timezone) {
    params.set('timezone', String(options.location.timezone))
  }

  const suffix = params.toString() ? `?${params.toString()}` : ''
  const response = await fetchWithTimeout(`${API_BASE_URL}/sidebar/${encodeURIComponent(section)}${suffix}`, {}, 30000)
  return parseJson(response)
}

export async function fetchNotifications(options = {}) {
  const params = new URLSearchParams()
  if (options?.locationQuery) {
    params.set('location_query', String(options.locationQuery))
  }
  if (options?.location?.latitude !== undefined && options?.location?.latitude !== null) {
    params.set('latitude', String(options.location.latitude))
  }
  if (options?.location?.longitude !== undefined && options?.location?.longitude !== null) {
    params.set('longitude', String(options.location.longitude))
  }
  if (options?.location?.timezone) {
    params.set('timezone', String(options.location.timezone))
  }
  if (options?.limit) {
    params.set('limit', String(options.limit))
  }

  const suffix = params.toString() ? `?${params.toString()}` : ''
  const response = await fetchWithTimeout(`${API_BASE_URL}/notifications${suffix}`, {}, 30000)
  return parseJson(response)
}

export async function fetchVoiceExplanation(options = {}) {
  const params = new URLSearchParams()
  if (options?.locationQuery) {
    params.set('location_query', String(options.locationQuery))
  }
  if (options?.location?.latitude !== undefined && options?.location?.latitude !== null) {
    params.set('latitude', String(options.location.latitude))
  }
  if (options?.location?.longitude !== undefined && options?.location?.longitude !== null) {
    params.set('longitude', String(options.location.longitude))
  }
  if (options?.location?.timezone) {
    params.set('timezone', String(options.location.timezone))
  }
  if (options?.language) {
    params.set('language', String(options.language))
  }
  if (options?.horizonDays) {
    params.set('horizon_days', String(options.horizonDays))
  }

  const suffix = params.toString() ? `?${params.toString()}` : ''
  const response = await fetchWithTimeout(`${API_BASE_URL}/voice/explain${suffix}`, {}, 35000)
  return parseJson(response)
}

export async function fetchMLMetrics() {
  const response = await fetchWithTimeout(`${API_BASE_URL}/ml/metrics`, {}, 15000)
  return parseJson(response)
}

export async function trainMLModel(requestBody) {
  const response = await fetchWithTimeout(`${API_BASE_URL}/ml/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
  }, 180000)
  return parseJson(response)
}

export async function predictMLForecast(requestBody) {
  const response = await fetchWithTimeout(`${API_BASE_URL}/ml/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
  }, 60000)
  return parseJson(response)
}

export { API_BASE_URL }
