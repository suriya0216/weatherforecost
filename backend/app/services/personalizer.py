from __future__ import annotations

from datetime import datetime, timedelta

from app.schemas import Coordinates, RoutineProfile, UserPreferences
from app.services.weather_client import weather_code_to_label


def _to_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _time_to_minutes(text: str) -> int:
    hour, minute = text.split(":")
    return int(hour) * 60 + int(minute)


def _nearest_entry_by_minutes(hourly: list[dict], target_minutes: int) -> dict:
    def score(entry: dict) -> int:
        stamp = _to_datetime(entry["time"])
        return abs((stamp.hour * 60 + stamp.minute) - target_minutes)

    return min(hourly, key=score)


def _build_hourly_table(weather_payload: dict, aqi_payload: dict) -> list[dict]:
    hourly = weather_payload["hourly"]
    aqi_hourly = aqi_payload.get("hourly", {})
    aqi_time_index = {stamp: idx for idx, stamp in enumerate(aqi_hourly.get("time", []))}

    rows: list[dict] = []
    for idx, stamp in enumerate(hourly.get("time", [])):
        aqi_value = None
        aqi_idx = aqi_time_index.get(stamp)
        if aqi_idx is not None:
            us_aqi = aqi_hourly.get("us_aqi", [])
            if aqi_idx < len(us_aqi):
                aqi_value = us_aqi[aqi_idx]

        rows.append(
            {
                "time": stamp,
                "temperature_c": hourly.get("temperature_2m", [None])[idx],
                "rain_probability": hourly.get("precipitation_probability", [None])[idx],
                "wind_kph": hourly.get("wind_speed_10m", [None])[idx],
                "uv_index": hourly.get("uv_index", [None])[idx],
                "aqi_us": aqi_value,
            }
        )
    return rows


def _slice_upcoming(hourly_rows: list[dict], reference_time: str, hours: int = 24) -> list[dict]:
    try:
        start_idx = next(i for i, item in enumerate(hourly_rows) if item["time"] == reference_time)
    except StopIteration:
        start_idx = 0
    return hourly_rows[start_idx : start_idx + hours]


def _insight_level(score: float) -> str:
    if score >= 70:
        return "warning"
    if score >= 40:
        return "attention"
    return "safe"


def _risk_score(entry: dict, preferences: UserPreferences) -> float:
    rain = (entry.get("rain_probability") or 0) * 0.7
    wind = max((entry.get("wind_kph") or 0) - 18, 0) * 1.3
    uv = max((entry.get("uv_index") or 0) - (5 if preferences.uv_sensitive else 7), 0) * 9
    heat = max((entry.get("temperature_c") or 0) - 30, 0) * 4
    aqi_penalty = 0
    if preferences.air_quality_sensitive:
        aqi_penalty = max((entry.get("aqi_us") or 0) - 70, 0) * 0.5
    return min(100.0, rain + wind + uv + heat + aqi_penalty)


def _pick_activity_window(hourly_rows: list[dict], preferences: UserPreferences) -> dict:
    if len(hourly_rows) < 2:
        return {
            "activity": "Outdoor workout",
            "best_start": None,
            "best_end": None,
            "reason": "Not enough forecast points yet.",
        }

    best_score = float("inf")
    best_window = None

    for idx in range(len(hourly_rows) - 1):
        first = hourly_rows[idx]
        second = hourly_rows[idx + 1]
        score = (_risk_score(first, preferences) + _risk_score(second, preferences)) / 2
        if score < best_score:
            best_score = score
            best_window = (first, second)

    if best_window is None:
        return {
            "activity": "Outdoor workout",
            "best_start": None,
            "best_end": None,
            "reason": "No stable activity window found.",
        }

    start = _to_datetime(best_window[0]["time"])
    end = _to_datetime(best_window[1]["time"]) + timedelta(hours=1)
    score_level = _insight_level(best_score)
    reason = {
        "safe": "Low rain and UV risk with comfortable wind.",
        "attention": "Moderate weather shifts. Keep a backup plan.",
        "warning": "High variability expected. Prefer indoor alternatives.",
    }[score_level]
    return {
        "activity": "Outdoor workout",
        "best_start": start.strftime("%H:%M"),
        "best_end": end.strftime("%H:%M"),
        "reason": reason,
        "risk_score": round(best_score, 1),
    }


def _build_daily_rows(weather_payload: dict) -> list[dict]:
    daily = weather_payload["daily"]
    rows: list[dict] = []
    for idx, date_value in enumerate(daily.get("time", [])):
        weather_code = daily.get("weather_code", [None])[idx]
        rows.append(
            {
                "date": date_value,
                "weather": weather_code_to_label(weather_code),
                "temp_max_c": daily.get("temperature_2m_max", [None])[idx],
                "temp_min_c": daily.get("temperature_2m_min", [None])[idx],
                "rain_probability_max": daily.get("precipitation_probability_max", [None])[idx],
            }
        )
    return rows


def _build_routine_insights(
    hourly_rows: list[dict],
    routine: RoutineProfile,
    preferences: UserPreferences,
) -> list[dict]:
    commute_slot = _nearest_entry_by_minutes(hourly_rows, _time_to_minutes(routine.commute_time))
    workout_slot = _nearest_entry_by_minutes(hourly_rows, _time_to_minutes(routine.workout_time))
    wake_slot = _nearest_entry_by_minutes(hourly_rows, _time_to_minutes(routine.wake_time))

    insights = []

    commute_risk = _risk_score(commute_slot, preferences)
    commute_summary = (
        f"At {routine.commute_time}, rain chance is {commute_slot['rain_probability']}% and wind is "
        f"{commute_slot['wind_kph']} km/h."
    )
    commute_action = "Carry a compact umbrella." if (commute_slot["rain_probability"] or 0) >= 35 else "Standard commute conditions."
    insights.append(
        {
            "title": "Commute risk",
            "time": routine.commute_time,
            "severity": _insight_level(commute_risk),
            "summary": commute_summary,
            "action": commute_action,
        }
    )

    workout_risk = _risk_score(workout_slot, preferences)
    workout_summary = (
        f"Workout slot near {routine.workout_time} shows UV {workout_slot['uv_index']} and "
        f"temperature {workout_slot['temperature_c']}C."
    )
    if (workout_slot["uv_index"] or 0) >= 6 and preferences.uv_sensitive:
        workout_action = "Shift outdoor workout to lower UV hours."
    elif (workout_slot["rain_probability"] or 0) >= 45:
        workout_action = "Use an indoor backup workout."
    else:
        workout_action = "Good outdoor training window."
    insights.append(
        {
            "title": "Workout planner",
            "time": routine.workout_time,
            "severity": _insight_level(workout_risk),
            "summary": workout_summary,
            "action": workout_action,
        }
    )

    wake_risk = _risk_score(wake_slot, preferences)
    wake_summary = (
        f"Morning around {routine.wake_time} starts at {wake_slot['temperature_c']}C with rain chance "
        f"{wake_slot['rain_probability']}%."
    )
    wake_action = (
        "Carry a light layer in the morning."
        if (wake_slot["temperature_c"] or 0) <= 16
        else "Comfortable start expected."
    )
    insights.append(
        {
            "title": "Morning readiness",
            "time": routine.wake_time,
            "severity": _insight_level(wake_risk),
            "summary": wake_summary,
            "action": wake_action,
        }
    )

    return insights


def build_personalized_response(
    *,
    location: Coordinates,
    weather_payload: dict,
    aqi_payload: dict,
    routine: RoutineProfile,
    preferences: UserPreferences,
) -> dict:
    current = weather_payload["current"]
    hourly_rows = _build_hourly_table(weather_payload, aqi_payload)
    upcoming_rows = _slice_upcoming(hourly_rows, current["time"], hours=24)
    daily_rows = _build_daily_rows(weather_payload)

    current_aqi = None
    for row in upcoming_rows:
        if row["time"] == current["time"]:
            current_aqi = row.get("aqi_us")
            break
    if current_aqi is None and upcoming_rows:
        current_aqi = upcoming_rows[0].get("aqi_us")

    activity_window = _pick_activity_window(upcoming_rows, preferences)
    routine_insights = _build_routine_insights(upcoming_rows, routine, preferences)

    rain_values = [item.get("rain_probability") or 0 for item in upcoming_rows]
    wind_values = [item.get("wind_kph") or 0 for item in upcoming_rows]
    uv_values = [item.get("uv_index") or 0 for item in upcoming_rows]
    temp_values = [item.get("temperature_c") or 0 for item in upcoming_rows]

    recommended_actions = [item["action"] for item in routine_insights]
    recommended_actions.append(activity_window["reason"])

    return {
        "location": {
            "name": location.name,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "timezone": weather_payload.get("timezone", location.timezone),
        },
        "current": {
            "time": current.get("time"),
            "temperature_c": current.get("temperature_2m"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_kph": current.get("wind_speed_10m"),
            "weather": weather_code_to_label(current.get("weather_code")),
            "is_day": bool(current.get("is_day", 1)),
            "aqi_us": current_aqi,
        },
        "metrics": {
            "max_temp_next_24h": round(max(temp_values), 1) if temp_values else None,
            "min_temp_next_24h": round(min(temp_values), 1) if temp_values else None,
            "rain_peak_probability": round(max(rain_values), 1) if rain_values else None,
            "wind_peak_kph": round(max(wind_values), 1) if wind_values else None,
            "uv_peak": round(max(uv_values), 1) if uv_values else None,
        },
        "hourly": upcoming_rows,
        "daily": daily_rows,
        "insights": routine_insights,
        "activity_windows": [activity_window],
        "recommended_actions": recommended_actions,
    }

