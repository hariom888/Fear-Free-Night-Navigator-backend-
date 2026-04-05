"""schemas.py — Pydantic request/response models for Fear-Free Navigator API."""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class LatLon(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class UserProfile(BaseModel):
    persona: Literal["solo_woman", "elderly", "delivery", "general"] = "general"
    safety_threshold: float = Field(0.65, ge=0.0, le=1.0)
    speed_weight: float = Field(0.4, ge=0.0, le=1.0,
                                description="0 = pure safety, 1 = pure speed")


class RouteRequest(BaseModel):
    origin: LatLon
    destination: LatLon
    departure_epoch: int = Field(..., description="Unix timestamp")
    profile: UserProfile = UserProfile()


class RouteSegment(BaseModel):
    node_idx: int
    css_score: float
    is_mun: bool = Field(False, description="Mandatory Unsafe Node flag")


class RouteTier(BaseModel):
    tier: Literal["safe_express", "balanced", "safe_scenic"]
    n_segments: int
    path_safety: float = Field(..., description="0.6*mean + 0.4*min CSS")
    total_cost: float
    extra_vs_fastest: float
    explanation: str
    path_node_indices: List[int]


class RouteResponse(BaseModel):
    tiers: List[RouteTier]
    time_band: int
    persona: str
    mun_alerts: List[int] = Field(default_factory=list,
                                   description="Node indices of unavoidable unsafe segments")


class SegmentSafetyResponse(BaseModel):
    edge_id: int
    time_band: int
    css_score: float
    is_safe: bool


class HeatmapRequest(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    time_band: int = 10


class HeatmapFeature(BaseModel):
    edge_id: int
    u: int
    v: int
    css_score: float
    lat: float
    lon: float


class HeatmapResponse(BaseModel):
    features: List[HeatmapFeature]
    time_band: int
    count: int


class FeedbackRequest(BaseModel):
    edge_id: int
    time_band: int
    perceived_safe: bool
    comment: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    css_cache_rows: int
    model_features: int
