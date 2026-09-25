export type SexValue = "Male" | "Female";
export type BinaryChoice = "No" | "Yes";
export type ChestPainValue =
  | "Chest pain during physical activity"
  | "Mild or unusual chest pain"
  | "Chest pain not related to the heart"
  | "No chest pain symptoms";

export type ClinicalProfile = {
  age: number;
  sex: SexValue;
  cp: ChestPainValue;
  trestbps: number;
  chol: number;
  fbs: BinaryChoice;
  restecg: number;
  thalach: number;
  exang: BinaryChoice;
  oldpeak: number;
  slope: number;
  ca: number;
  thal: number;
};

export type HealthResponse = {
  status: "ready";
  model_ready: boolean;
};

export type ModelMetadata = {
  model: string;
  feature_count: number;
  threshold: number;
  educational_only: boolean;
  disclaimer: string;
};

export type PredictionResult = {
  probability: number;
  percentage: number;
  interpretation: "Higher probability" | "Lower probability";
  threshold: number;
  model: string;
  feature_count: number;
};
