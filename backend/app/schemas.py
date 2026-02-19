from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Coordinates(BaseModel):
    name: str | None = Field(default=None, description="City or place name.")
    latitude: float
    longitude: float
    timezone: str = Field(default="auto")


class RoutineProfile(BaseModel):
    wake_time: str = Field(default="07:00", pattern=r"^\d{2}:\d{2}$")
    commute_time: str = Field(default="08:30", pattern=r"^\d{2}:\d{2}$")
    workout_time: str = Field(default="18:00", pattern=r"^\d{2}:\d{2}$")
    sleep_time: str = Field(default="22:30", pattern=r"^\d{2}:\d{2}$")


class UserPreferences(BaseModel):
    outdoor_commute: bool = True
    uv_sensitive: bool = True
    air_quality_sensitive: bool = False


class PersonalizedForecastRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    location_query: str | None = Field(
        default=None,
        description="City or region text to geocode if coordinates are not provided.",
    )
    location: Coordinates | None = None
    routine: RoutineProfile = Field(default_factory=RoutineProfile)
    preferences: UserPreferences = Field(default_factory=UserPreferences)

    @model_validator(mode="after")
    def validate_location_inputs(self) -> "PersonalizedForecastRequest":
        if self.location is None and (self.location_query is None or not self.location_query.strip()):
            raise ValueError("Provide either location_query or location coordinates.")
        return self

