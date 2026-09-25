from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SexValue = Literal["Male", "Female"]
BinaryChoice = Literal["No", "Yes"]
ChestPainValue = Literal[
    "Chest pain during physical activity",
    "Mild or unusual chest pain",
    "Chest pain not related to the heart",
    "No chest pain symptoms",
]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=1, le=100)
    sex: SexValue
    cp: ChestPainValue
    trestbps: int = Field(ge=50, le=250)
    chol: int = Field(ge=100, le=600)
    fbs: BinaryChoice
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=50, le=250)
    exang: BinaryChoice
    oldpeak: float = Field(ge=0.0, le=10.0)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)


class PredictionResponse(BaseModel):
    probability: float
    percentage: float
    interpretation: Literal["Higher probability", "Lower probability"]
    threshold: float
    model: str
    feature_count: int


class HealthResponse(BaseModel):
    status: Literal["ready"]
    model_ready: bool


class MetadataResponse(BaseModel):
    model: str
    feature_count: int
    threshold: float
    educational_only: bool
    disclaimer: str
