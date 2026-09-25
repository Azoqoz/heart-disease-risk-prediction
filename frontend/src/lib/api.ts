import type {
  ClinicalProfile,
  HealthResponse,
  ModelMetadata,
  PredictionResult,
} from "./types";

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function getMetadata(): Promise<ModelMetadata> {
  return request<ModelMetadata>("/metadata");
}

export function runPrediction(
  profile: ClinicalProfile,
): Promise<PredictionResult> {
  return request<PredictionResult>("/predict", {
    method: "POST",
    body: JSON.stringify(profile),
  });
}
