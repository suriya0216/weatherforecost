# NimbusIQ - AI Weather Forecast Platform

Modern full-stack weather forecast platform with:
- Premium React + Tailwind landing/dashboard UI
- FastAPI backend
- Live Open-Meteo forecast + AQI integration
- Personalized routine-aware recommendation engine

## Project Structure

```text
wethar forcast/
  backend/
    app/
      main.py
      config.py
      schemas.py
      services/
        weather_client.py
        personalizer.py
    tests/
      test_personalizer.py
    requirements.txt
  src/
    api/weatherApi.js
    App.jsx
    index.css
  .env.example
  package.json
```

## 1) One-Command Local Run (Recommended)

1. Install frontend dependencies:

```powershell
npm install
```

2. Install backend dependencies:

```powershell
npm run setup:backend
```

3. Start both frontend + backend together:

```powershell
npm run dev
```

Open:
- `http://localhost:5173`

## 2) Backend Setup Only (FastAPI)

1. Open terminal in project root.
2. Create and activate virtual environment:

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
```

3. Install backend dependencies:

```powershell
pip install -r backend\requirements.txt
```

4. Optional: set CORS origins (defaults already support Vite localhost):

```powershell
Copy-Item backend\.env.example backend\.env
```

5. Run backend server:

```powershell
python -m uvicorn app.main:app --reload --port 8000 --app-dir backend --env-file backend/.env
```

Backend URLs:
- Health: `http://localhost:8000/api/health`
- Personalized forecast: `POST http://localhost:8000/api/forecast/personalized`

## 3) Frontend Setup Only (React + Vite)

1. In a second terminal, from project root:

```powershell
npm install
```

2. Set API base URL (optional; default is already `/api` via Vite proxy):

```powershell
Copy-Item .env.example .env
```

3. Run frontend:

```powershell
npm run dev:frontend
```

Open:
- `http://localhost:5173`

## 4) Manual End-to-End Flow

1. Run `npm run dev` (starts backend on `8000` and frontend on `5173`).
2. Open UI and go to **Live Weather Dashboard Preview**.
3. Enter:
   - Location (city name)
   - Wake/Commute/Workout/Sleep time
   - Preference toggles (outdoor commute, UV sensitive, AQI sensitive)
4. Click **Generate AI Forecast**.
5. Validate output:
   - Current conditions
   - Hourly temperature/rain/wind charts
   - AI insights with severity
   - Recommended actions
   - Activity window recommendation

## 5) API Contract

### `POST /api/forecast/personalized`

Request example:

```json
{
  "location_query": "New York",
  "routine": {
    "wake_time": "07:00",
    "commute_time": "08:30",
    "workout_time": "18:00",
    "sleep_time": "22:30"
  },
  "preferences": {
    "outdoor_commute": true,
    "uv_sensitive": true,
    "air_quality_sensitive": false
  }
}
```

Response includes:
- `location`
- `current`
- `metrics`
- `hourly` (next 24h)
- `daily` (3-day)
- `insights` (routine-aware)
- `activity_windows`
- `recommended_actions`

## 6) Verification Commands

Frontend checks:

```powershell
npm run lint
npm run build
```

Backend unit test:

```powershell
backend\.venv\Scripts\Activate.ps1
cd backend
python -m pytest
```

## 7) Notes

- Weather and geocoding data are fetched from Open-Meteo APIs.
- If backend is not running, UI still renders design and shows offline API status.
- This implementation is production-style starter architecture and ready for next phase:
  auth, user profiles, persistent storage, and model retraining pipeline.
