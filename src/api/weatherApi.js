const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')

async function parseJson(response) {
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = payload?.detail || 'Request failed'
    throw new Error(message)
  }
  return payload
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL}/health`)
  return parseJson(response)
}

export async function fetchPersonalizedForecast(requestBody) {
  const response = await fetch(`${API_BASE_URL}/forecast/personalized`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
  })
  return parseJson(response)
}

export { API_BASE_URL }
