"use client";

import type { CSSProperties, FormEvent } from "react";
import { useEffect, useState } from "react";

import { getHealth, getMetadata, runPrediction } from "@/lib/api";
import type {
  ClinicalProfile,
  ModelMetadata,
  PredictionResult,
} from "@/lib/types";

const DEFAULT_CLINICAL_SIGNALS = {
  restecg: 0,
  oldpeak: 1.0,
  slope: 0,
  ca: 0,
  thal: 0,
} as const;

const DEFAULT_PROFILE: ClinicalProfile = {
  age: 25,
  sex: "Male",
  cp: "Chest pain during physical activity",
  trestbps: 120,
  chol: 200,
  fbs: "No",
  restecg: DEFAULT_CLINICAL_SIGNALS.restecg,
  thalach: 150,
  exang: "No",
  oldpeak: DEFAULT_CLINICAL_SIGNALS.oldpeak,
  slope: DEFAULT_CLINICAL_SIGNALS.slope,
  ca: DEFAULT_CLINICAL_SIGNALS.ca,
  thal: DEFAULT_CLINICAL_SIGNALS.thal,
};

const STAGES = [
  { number: "01", short: "Patient", title: "Patient profile", note: "Begin with two foundational characteristics." },
  { number: "02", short: "Vitals", title: "Vital measurements", note: "Enter the recorded cardiovascular measurements." },
  { number: "03", short: "Symptoms", title: "Symptoms", note: "Describe the observed symptom indicators." },
  { number: "04", short: "Signals", title: "Optional clinical signals", note: "Adjust these values if available. Otherwise, the current defaults will be used." },
] as const;

type ConnectionState = "connecting" | "ready" | "unavailable";
type Screen = "scan" | "review" | "result";

type StepperControlProps = {
  label: string;
  value: number;
  minimum: number;
  maximum: number;
  step?: number;
  onChange: (value: number) => void;
};

function formatDisplayedPercentage(value: number) {
  const rounded = value.toFixed(1);
  return value < 100 && rounded === "100.0" ? "99.9" : rounded;
}

function StepperControl({
  label,
  value,
  minimum,
  maximum,
  step = 1,
  onChange,
}: StepperControlProps) {
  const precision = step.toString().split(".")[1]?.length ?? 0;

  function adjustedValue(direction: -1 | 1) {
    const nextValue = value + direction * step;
    return Number(
      Math.min(maximum, Math.max(minimum, nextValue)).toFixed(precision),
    );
  }

  return (
    <fieldset className="measurement-control stepper-field">
      <legend className="control-label">{label}</legend>
      <div className="signal-stepper">
        <button
          type="button"
          aria-label={`Decrease ${label}`}
          disabled={value <= minimum}
          onClick={() => onChange(adjustedValue(-1))}
        >
          −
        </button>
        <output aria-live="polite">{value.toFixed(precision)}</output>
        <button
          type="button"
          aria-label={`Increase ${label}`}
          disabled={value >= maximum}
          onClick={() => onChange(adjustedValue(1))}
        >
          +
        </button>
      </div>
    </fieldset>
  );
}

export function ClinicalWorkbench() {
  const [profile, setProfile] = useState<ClinicalProfile>(DEFAULT_PROFILE);
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [screen, setScreen] = useState<Screen>("scan");
  const [stage, setStage] = useState(0);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isDirty, setIsDirty] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    Promise.all([getHealth(), getMetadata()])
      .then(([health, modelMetadata]) => {
        if (!active) return;
        setMetadata(modelMetadata);
        setConnection(health.model_ready ? "ready" : "unavailable");
      })
      .catch(() => {
        if (active) setConnection("unavailable");
      });

    return () => {
      active = false;
    };
  }, []);

  function updateField<K extends keyof ClinicalProfile>(
    field: K,
    value: ClinicalProfile[K],
  ) {
    setProfile((current) => ({ ...current, [field]: value }));
    if (result) setIsDirty(true);
    setError(null);
  }

  function continueScan(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (stage < STAGES.length - 1) {
      setStage((current) => current + 1);
    } else {
      setScreen("review");
    }
  }

  function goBack() {
    setStage((current) => Math.max(0, current - 1));
  }

  async function runModel() {
    setIsLoading(true);
    setError(null);

    try {
      const nextResult = await runPrediction(profile);
      setResult(nextResult);
      setIsDirty(false);
      setConnection("ready");
      setScreen("result");
    } catch {
      setError("The model service could not complete this screening. Check the API connection and try again.");
      setConnection("unavailable");
    } finally {
      setIsLoading(false);
    }
  }

  function navigateToStage(targetStage: number) {
    setStage(targetStage);
    setScreen("scan");
    setError(null);
  }

  function useSignalDefaultsAndReview() {
    setProfile((current) => ({ ...current, ...DEFAULT_CLINICAL_SIGNALS }));
    if (result) setIsDirty(true);
    setError(null);
    setScreen("review");
  }

  function newScreening() {
    setProfile(DEFAULT_PROFILE);
    setResult(null);
    setIsDirty(false);
    setError(null);
    setStage(0);
    setScreen("scan");
  }

  const currentStage = STAGES[stage];
  const percentage = result ? Math.min(100, Math.max(0, result.percentage)) : 0;
  const displayedPercentage = formatDisplayedPercentage(percentage);
  const threshold = result?.threshold ?? metadata?.threshold ?? 0.5;
  const higher = result?.interpretation === "Higher probability";
  const signalsUseDefaults =
    profile.restecg === DEFAULT_CLINICAL_SIGNALS.restecg &&
    profile.oldpeak === DEFAULT_CLINICAL_SIGNALS.oldpeak &&
    profile.slope === DEFAULT_CLINICAL_SIGNALS.slope &&
    profile.ca === DEFAULT_CLINICAL_SIGNALS.ca &&
    profile.thal === DEFAULT_CLINICAL_SIGNALS.thal;
  const trackStyle = {
    "--probability": `${percentage}%`,
    "--threshold": `${threshold * 100}%`,
  } as CSSProperties;

  return (
    <main className={`application-shell application-shell--${screen}`}>
      <header className="topbar">
        <div className="brand-lockup">
          <p>Heart Disease Detection</p>
          <span>ML Screening</span>
        </div>
        <div className={`api-state api-state--${connection}`} aria-live="polite">
          <span aria-hidden="true" />
          {connection === "connecting" && "Connecting"}
          {connection === "ready" && "API ready"}
          {connection === "unavailable" && "API unavailable"}
        </div>
      </header>

      {screen !== "result" && (
        <nav className="progress-rail" aria-label="Clinical scan progress">
          {STAGES.map((item, index) => (
            <button
              type="button"
              className={`progress-step ${index === stage && screen === "scan" ? "progress-step--active" : ""} ${index < stage || screen === "review" ? "progress-step--complete" : ""}`}
              key={item.number}
              onClick={() => navigateToStage(index)}
              aria-current={index === stage && screen === "scan" ? "step" : undefined}
            >
              <span>{item.number}</span>
              <strong>{item.short}</strong>
            </button>
          ))}
        </nav>
      )}

      {screen === "scan" && (
        <section className="scan-stage" aria-labelledby="stage-title">
          <div className="stage-heading">
            <p>Guided clinical scan · {currentStage.number} / 04</p>
            <h1 id="stage-title">{currentStage.title}</h1>
            <span>{currentStage.note}</span>
          </div>

          <form className={`stage-form stage-form--${stage}`} onSubmit={continueScan}>
            {stage === 0 && (
              <div className="control-stack">
                <label className="measurement-control">
                  <span className="control-label">Age</span>
                  <span className="numeric-input">
                    <input
                      type="number"
                      min="1"
                      max="100"
                      required
                      value={profile.age}
                      onChange={(event) => updateField("age", event.target.valueAsNumber)}
                    />
                    <strong>years</strong>
                  </span>
                </label>
                <fieldset className="instrument-choice">
                  <legend>Sex</legend>
                  <div className="segmented-control">
                    {(["Male", "Female"] as const).map((value) => (
                      <button
                        type="button"
                        aria-pressed={profile.sex === value}
                        className={profile.sex === value ? "selected" : ""}
                        key={value}
                        onClick={() => updateField("sex", value)}
                      >
                        {value}
                      </button>
                    ))}
                  </div>
                </fieldset>
              </div>
            )}

            {stage === 1 && (
              <div className="measurement-grid">
                <label className="measurement-control">
                  <span className="control-label">Resting blood pressure</span>
                  <span className="numeric-input">
                    <input
                      type="number"
                      min="50"
                      max="250"
                      required
                      value={profile.trestbps}
                      onChange={(event) => updateField("trestbps", event.target.valueAsNumber)}
                    />
                    <strong>mmHg</strong>
                  </span>
                </label>
                <label className="measurement-control">
                  <span className="control-label">Cholesterol</span>
                  <span className="numeric-input">
                    <input
                      type="number"
                      min="100"
                      max="600"
                      required
                      value={profile.chol}
                      onChange={(event) => updateField("chol", event.target.valueAsNumber)}
                    />
                    <strong>mg/dL</strong>
                  </span>
                </label>
                <label className="measurement-control measurement-control--wide">
                  <span className="control-label">Maximum heart rate</span>
                  <span className="numeric-input">
                    <input
                      type="number"
                      min="50"
                      max="250"
                      required
                      value={profile.thalach}
                      onChange={(event) => updateField("thalach", event.target.valueAsNumber)}
                    />
                    <strong>bpm</strong>
                  </span>
                </label>
              </div>
            )}

            {stage === 2 && (
              <div className="control-stack">
                <label className="select-control">
                  <span className="control-label">Chest pain type</span>
                  <select
                    value={profile.cp}
                    onChange={(event) => updateField("cp", event.target.value as ClinicalProfile["cp"])}
                  >
                    <option>Chest pain during physical activity</option>
                    <option>Mild or unusual chest pain</option>
                    <option>Chest pain not related to the heart</option>
                    <option>No chest pain symptoms</option>
                  </select>
                </label>
                <div className="choice-grid">
                  <fieldset className="instrument-choice">
                    <legend>Exercise-induced angina</legend>
                    <div className="segmented-control">
                      {(["No", "Yes"] as const).map((value) => (
                        <button
                          type="button"
                          aria-pressed={profile.exang === value}
                          className={profile.exang === value ? "selected" : ""}
                          key={value}
                          onClick={() => updateField("exang", value)}
                        >
                          {value}
                        </button>
                      ))}
                    </div>
                  </fieldset>
                  <fieldset className="instrument-choice">
                    <legend>High fasting blood sugar</legend>
                    <div className="segmented-control">
                      {(["No", "Yes"] as const).map((value) => (
                        <button
                          type="button"
                          aria-pressed={profile.fbs === value}
                          className={profile.fbs === value ? "selected" : ""}
                          key={value}
                          onClick={() => updateField("fbs", value)}
                        >
                          {value}
                        </button>
                      ))}
                    </div>
                  </fieldset>
                </div>
              </div>
            )}

            {stage === 3 && (
              <div className="signal-grid">
                <StepperControl
                  label="Resting ECG"
                  value={profile.restecg}
                  minimum={0}
                  maximum={2}
                  onChange={(value) => updateField("restecg", value)}
                />
                <StepperControl
                  label="ST depression"
                  value={profile.oldpeak}
                  minimum={0}
                  maximum={10}
                  step={0.1}
                  onChange={(value) => updateField("oldpeak", value)}
                />
                <StepperControl
                  label="ST slope"
                  value={profile.slope}
                  minimum={0}
                  maximum={2}
                  onChange={(value) => updateField("slope", value)}
                />
                <StepperControl
                  label="Number of major vessels"
                  value={profile.ca}
                  minimum={0}
                  maximum={4}
                  onChange={(value) => updateField("ca", value)}
                />
                <div className="signal-wide">
                  <StepperControl
                    label="Thalassemia"
                    value={profile.thal}
                    minimum={0}
                    maximum={3}
                    onChange={(value) => updateField("thal", value)}
                  />
                </div>
              </div>
            )}

            {stage === 3 && result && isDirty && (
              <p className="dirty-note dirty-note--signals">Inputs changed since the last model reading.</p>
            )}

            <div className={`stage-actions ${stage === 3 ? "stage-actions--signals" : ""}`}>
              {stage > 0 ? (
                <button className="text-action" type="button" onClick={goBack}>Back</button>
              ) : (
                <span />
              )}
              {stage === 3 ? (
                <div className="signal-action-cluster">
                  <button className="signal-default-action" type="button" onClick={useSignalDefaultsAndReview}>Use defaults &amp; review</button>
                  <button className="primary-action" type="submit">Review inputs</button>
                </div>
              ) : (
                <button className="primary-action" type="submit">Continue</button>
              )}
            </div>
          </form>

          {stage !== 3 && result && isDirty && <p className="dirty-note">Inputs changed since the last model reading.</p>}
        </section>
      )}

      {screen === "review" && (
        <section className="review-screen" aria-labelledby="review-title">
          <div className="review-heading">
            <p>Ready to screen</p>
            <h1 id="review-title">13 <span>inputs captured</span></h1>
            <small>Review the clinical profile before running the model.</small>
          </div>

          <div className="review-content">
            <div className="profile-review">
              <section className="review-group" aria-labelledby="review-patient">
                <h2 id="review-patient">Patient</h2>
                <dl>
                  <div><dt>Age</dt><dd>{profile.age} years</dd></div>
                  <div><dt>Sex</dt><dd>{profile.sex}</dd></div>
                </dl>
              </section>
              <section className="review-group" aria-labelledby="review-vitals">
                <h2 id="review-vitals">Vitals</h2>
                <dl>
                  <div><dt>Resting blood pressure</dt><dd>{profile.trestbps} mmHg</dd></div>
                  <div><dt>Cholesterol</dt><dd>{profile.chol} mg/dL</dd></div>
                  <div><dt>Maximum heart rate</dt><dd>{profile.thalach} bpm</dd></div>
                </dl>
              </section>
              <section className="review-group" aria-labelledby="review-symptoms">
                <h2 id="review-symptoms">Symptoms</h2>
                <dl>
                  <div><dt>Chest pain type</dt><dd>{profile.cp}</dd></div>
                  <div><dt>Exercise-induced angina</dt><dd>{profile.exang}</dd></div>
                  <div><dt>High fasting blood sugar</dt><dd>{profile.fbs}</dd></div>
                </dl>
              </section>
              <section className="review-group" aria-labelledby="review-signals">
                <h2 id="review-signals">
                  Clinical Signals
                  {signalsUseDefaults && <span>Default values</span>}
                </h2>
                <dl>
                  <div><dt>Resting ECG</dt><dd>{profile.restecg}</dd></div>
                  <div><dt>ST depression</dt><dd>{profile.oldpeak}</dd></div>
                  <div><dt>ST slope</dt><dd>{profile.slope}</dd></div>
                  <div><dt>Number of major vessels</dt><dd>{profile.ca}</dd></div>
                  <div><dt>Thalassemia</dt><dd>{profile.thal}</dd></div>
                </dl>
              </section>
            </div>

            {isDirty && <p className="dirty-note">Inputs changed since the last model reading.</p>}
            {error && <p className="form-error" role="alert">{error}</p>}

            <div className="review-actions">
              <button className="text-action" type="button" onClick={() => navigateToStage(0)}>Edit inputs</button>
              <button className="primary-action primary-action--run" type="button" disabled={isLoading} onClick={runModel}>
                {isLoading ? "Running model…" : "Run model"}
              </button>
            </div>
          </div>
        </section>
      )}

      {screen === "result" && result && (
        <section className={`result-screen ${higher ? "result-screen--higher" : "result-screen--lower"}`} aria-labelledby="result-title">
          <div className="result-reading">
            <p>Model reading</p>
            <h1 id="result-title">
              {displayedPercentage}<span>%</span>
            </h1>
            <strong>{higher ? "Higher probability" : "Lower probability"}</strong>
            <small>Model-estimated probability</small>
          </div>

          <div className="result-scale" style={trackStyle} aria-label={`Model probability ${displayedPercentage} percent`}>
            <div className="scale-line">
              <span className="scale-threshold" />
              <span className="scale-marker" />
            </div>
            <div className="scale-labels"><span>0%</span><span>50% threshold</span><span>100%</span></div>
          </div>

          <dl className="result-details">
            <div><dt>Model signal</dt><dd>{result.interpretation === "Higher probability" ? "Higher" : "Lower"} model-estimated probability</dd></div>
            <div><dt>Model</dt><dd>{result.model}</dd></div>
            <div><dt>Input vector</dt><dd>{result.feature_count} features</dd></div>
            <div><dt>Threshold</dt><dd>{result.threshold * 100}%</dd></div>
          </dl>

          <div className="result-actions">
            <button className="secondary-action" type="button" onClick={() => navigateToStage(0)}>Review inputs</button>
            <button className="text-action" type="button" onClick={newScreening}>New screening</button>
          </div>
        </section>
      )}

      <footer className="disclaimer">
        {metadata?.disclaimer ?? "Educational screening tool only. This model output is not a medical diagnosis and does not replace evaluation by a qualified healthcare professional."}
      </footer>
    </main>
  );
}
