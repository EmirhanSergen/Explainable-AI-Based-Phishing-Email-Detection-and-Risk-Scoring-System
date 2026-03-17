/**
 * Phishing Email Detection - Frontend
 * API base URL: same origin
 */

const API_BASE = "";

const emailText = document.getElementById("emailText");
const analyzeBtn = document.getElementById("analyzeBtn");
const generateBtn = document.getElementById("generateBtn");
const resultsSection = document.getElementById("results");
const summaryBanner = document.getElementById("comparisonSummary");
const sampleMeta = document.getElementById("sampleMeta");
const mainResult = document.getElementById("mainResult");
const hybridResult = document.getElementById("hybridResult");

const state = {
  metrics: {},
};

const MODEL_CONFIG = {
  main: {
    label: "Main Model",
    metricKey: "logistic_regression",
    element: mainResult,
    description: "TF-IDF + security features",
  },
  hybrid: {
    label: "Hybrid Model",
    metricKey: "logistic_regression_hybrid",
    element: hybridResult,
    description: "TF-IDF + security features + embeddings",
  },
};

analyzeBtn.addEventListener("click", async () => {
  await analyzeCurrentEmail();
});

generateBtn.addEventListener("click", async () => {
  generateBtn.disabled = true;
  try {
    await loadSampleEmail({ analyzeAfterLoad: true });
  } catch (err) {
    console.error(err);
    alert("A saved test email could not be loaded.");
  } finally {
    generateBtn.disabled = false;
  }
});

document.addEventListener("DOMContentLoaded", () => {
  initializeApp().catch((err) => {
    console.error(err);
  });
});

async function initializeApp() {
  try {
    await loadMetrics();
  } catch (err) {
    console.error(err);
  }

  try {
    await loadSampleEmail({ analyzeAfterLoad: true });
  } catch (err) {
    console.error(err);
    sampleMeta.textContent = "Could not load a saved test email. You can still paste text manually.";
  }
}

async function loadMetrics() {
  state.metrics = await fetchJson(`${API_BASE}/model_metrics`);
}

async function loadSampleEmail(options = {}) {
  const { analyzeAfterLoad = false } = options;
  const sample = await fetchJson(`${API_BASE}/sample_email`);
  emailText.value = sample.text || "";
  sampleMeta.textContent = sample.label
    ? `Loaded a saved ${sample.label} email from the test split.`
    : "Loaded a saved email from the test split.";

  if (analyzeAfterLoad && emailText.value.trim()) {
    await analyzeCurrentEmail();
  }
}

async function analyzeCurrentEmail() {
  const text = emailText.value.trim();
  if (!text) {
    alert("Please enter an email before analyzing.");
    return;
  }

  setBusyState(true);

  try {
    const [mainResponse, hybridResponse] = await Promise.allSettled([
      analyzeModel("main", text),
      analyzeModel("hybrid", text),
    ]);

    renderComparison(
      {
        main: mainResponse.status === "fulfilled" ? mainResponse.value : null,
        hybrid: hybridResponse.status === "fulfilled" ? hybridResponse.value : null,
      },
      {
        main: mainResponse.status === "rejected" ? mainResponse.reason : null,
        hybrid: hybridResponse.status === "rejected" ? hybridResponse.reason : null,
      }
    );
    resultsSection.classList.remove("hidden");
  } catch (err) {
    console.error(err);
    alert("The analysis request failed. Make sure the API is running and both trained models exist.");
  } finally {
    setBusyState(false);
  }
}

async function analyzeModel(model, text) {
  return fetchJson(`${API_BASE}/analyze_email?model=${encodeURIComponent(model)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

function renderComparison(results, errors) {
  const successfulResults = Object.values(results).filter(Boolean);
  if (successfulResults.length === 0) {
    throw new Error("Both model analyses failed.");
  }

  const mainData = results.main;
  const hybridData = results.hybrid;

  summaryBanner.innerHTML = buildSummaryBanner(mainData, hybridData, errors);
  renderModelCard("main", mainData, errors.main);
  renderModelCard("hybrid", hybridData, errors.hybrid);
}

function renderModelCard(modelName, data, error) {
  const config = MODEL_CONFIG[modelName];
  const metrics = (state.metrics[modelName] || {})[config.metricKey] || null;
  const metricsLabel =
    modelName === "main"
      ? "Phase 1 benchmark (TF-IDF space)"
      : "Phase 2 benchmark (TF-IDF + security + embeddings)";

  if (error || !data) {
    config.element.innerHTML = `
      <div class="result-card">
        <div class="model-card-header">
          <div>
            <p class="eyebrow">${config.label}</p>
            <h2>${config.description}</h2>
          </div>
        </div>
        <p class="error-text">This model is currently unavailable.</p>
        <p class="muted-text">${escapeHtml(getErrorMessage(error))}</p>
      </div>
    `;
    return;
  }

  config.element.innerHTML = `
    <div class="result-card">
      <div class="model-card-header">
        <div>
          <p class="eyebrow">${config.label}</p>
          <h2>${config.description}</h2>
        </div>
        <span class="pill ${data.prediction === "phishing" ? "pill-danger" : "pill-success"}">
          ${escapeHtml(data.prediction)}
        </span>
      </div>

      <div class="hero-metrics">
        <div>
          <span class="hero-label">Risk Score</span>
          <strong>${Number(data.risk_score).toFixed(1)}</strong>
        </div>
        <div>
          <span class="hero-label">Probability</span>
          <strong>${formatPercent(data.probability)}</strong>
        </div>
        <div>
          <span class="hero-label">Risk Level</span>
          <strong class="risk-level-${String(data.risk_level).toLowerCase()}">${escapeHtml(data.risk_level)}</strong>
        </div>
      </div>

      <details class="help-block">
        <summary>What do these numbers mean?</summary>
        <div class="help-content">
          <ul class="help-list">
            <li><strong>Probability</strong> is the model’s phishing probability (0–100%).</li>
            <li><strong>Risk Score</strong> is a 0–100 score combining probability + security signals (URLs, keywords).</li>
            <li><strong>S_prob</strong> is the probability contribution (scaled to 0–100).</li>
            <li><strong>S_url</strong> increases risk when URLs are present (more URLs ⇒ higher score).</li>
            <li><strong>S_kw</strong> increases risk when critical phishing keywords are present.</li>
            <li><strong>Positive Indicators</strong> are words/phrases pushing prediction toward phishing.</li>
            <li><strong>Negative Indicators</strong> are words/phrases pushing prediction toward legitimate.</li>
          </ul>
          <p class="muted-text">
            Note: indicators come from the TF-IDF linear part. In the <strong>Hybrid</strong> model, embeddings also influence the
            probability. Below, we show embedding impact as a numeric contribution (not word-level).
          </p>
        </div>
      </details>

      ${renderEmbeddingImpact(modelName, data.group_contributions)}

      <section class="subsection">
        <h3>Training Metrics</h3>
        <p class="muted-text">${metricsLabel} • showing <code>${config.metricKey}</code></p>
        ${renderMetricGrid(metrics)}
      </section>

      <section class="subsection">
        <h3>Risk Components</h3>
        <div class="metric-list">
          <div><span>S_prob</span><strong>${Number(data.risk_components.s_prob).toFixed(1)}</strong></div>
          <div><span>S_url</span><strong>${Number(data.risk_components.s_url).toFixed(1)}</strong></div>
          <div><span>S_kw</span><strong>${Number(data.risk_components.s_kw).toFixed(1)}</strong></div>
        </div>
      </section>

      <section class="subsection indicators-grid">
        <div>
          <h3>Positive Indicators</h3>
          ${renderIndicatorList(data.top_indicators_pos || [], "positive")}
        </div>
        <div>
          <h3>Negative Indicators</h3>
          ${renderIndicatorList(data.top_indicators_neg || [], "negative")}
        </div>
      </section>
    </div>
  `;
}

function renderEmbeddingImpact(modelName, groupContributions) {
  if (modelName !== "hybrid") {
    return "";
  }
  if (!groupContributions || !groupContributions.embedding) {
    return `
      <section class="subsection">
        <h3>Embedding Impact</h3>
        <p class="muted-text">Embedding contribution is not available for this response.</p>
      </section>
    `;
  }

  const tfidf = Number((groupContributions.tfidf || [0])[0] || 0);
  const security = Number((groupContributions.security || [0])[0] || 0);
  const embedding = Number((groupContributions.embedding || [0])[0] || 0);
  const bias = Number(groupContributions.bias || 0);
  const total = tfidf + security + embedding + bias;

  return `
    <section class="subsection">
      <h3>Embedding Impact</h3>
      <div class="metric-grid metric-grid-4">
        <div><span>TF-IDF logit</span><strong>${tfidf.toFixed(3)}</strong></div>
        <div><span>Security logit</span><strong>${security.toFixed(3)}</strong></div>
        <div><span>Embedding logit</span><strong class="${embedding >= 0 ? "text-danger" : "text-success"}">${embedding.toFixed(3)}</strong></div>
        <div><span>Total logit</span><strong>${total.toFixed(3)}</strong></div>
      </div>
      <p class="muted-text">
        These are contributions to the model’s internal score (logit). Positive values push toward <strong>phishing</strong>, negative values push toward <strong>legitimate</strong>.
      </p>
    </section>
  `;
}

function buildSummaryBanner(mainData, hybridData, errors) {
  if (!mainData && !hybridData) {
    return "<p>Both models are unavailable.</p>";
  }

  if (!mainData || !hybridData) {
    const availableModel = mainData ? "Main model" : "Hybrid model";
    const missingError = !mainData ? getErrorMessage(errors.main) : getErrorMessage(errors.hybrid);
    return `
      <p><strong>${availableModel}</strong> returned a result, but the other model failed.</p>
      <p class="muted-text">${escapeHtml(missingError)}</p>
    `;
  }

  const riskDelta = (hybridData.risk_score - mainData.risk_score).toFixed(1);
  const moreSevereModel =
    hybridData.risk_score > mainData.risk_score ? "Hybrid model" : "Main model";

  return `
    <p><strong>${moreSevereModel}</strong> assigned the higher risk score for this email.</p>
    <p class="muted-text">
      Main: ${Number(mainData.risk_score).toFixed(1)} |
      Hybrid: ${Number(hybridData.risk_score).toFixed(1)} |
      Delta: ${riskDelta}
    </p>
  `;
}

function renderMetricGrid(metrics) {
  if (!metrics) {
    return '<p class="muted-text">Training metrics are not available yet.</p>';
  }

  return `
    <div class="metric-grid">
      <div><span>Accuracy</span><strong>${formatPercent(metrics.accuracy)}</strong></div>
      <div><span>Precision</span><strong>${formatPercent(metrics.precision)}</strong></div>
      <div><span>Recall</span><strong>${formatPercent(metrics.recall)}</strong></div>
      <div><span>F1</span><strong>${formatPercent(metrics.f1)}</strong></div>
    </div>
  `;
}

function renderIndicatorList(indicators, tone) {
  if (!indicators.length) {
    return '<p class="muted-text">No strong indicators found.</p>';
  }

  return `
    <ul class="indicator-list">
      ${indicators
        .map(
          (item) => `
            <li>
              <span>${escapeHtml(item.word)}</span>
              <strong class="${tone === "positive" ? "text-danger" : "text-success"}">
                ${formatSignedPercent(item.contribution)}
              </strong>
            </li>
          `
        )
        .join("")}
    </ul>
  `;
}

function setBusyState(isBusy) {
  analyzeBtn.disabled = isBusy;
  generateBtn.disabled = isBusy;
  analyzeBtn.textContent = isBusy ? "Analyzing..." : "Analyze Both Models";
  generateBtn.textContent = isBusy ? "Loading..." : "Generate Email";
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatSignedPercent(value) {
  const number = Number(value) * 100;
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(1)}%`;
}

function getErrorMessage(error) {
  if (!error) {
    return "Unknown error.";
  }
  return error.message || String(error);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (err) {
      console.error(err);
    }
    throw new Error(detail);
  }
  return response.json();
}
