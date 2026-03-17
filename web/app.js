/**
 * Phishing Email Detection - Frontend
 * API base URL: same origin
 */

const API_BASE = "";

const emailText = document.getElementById("emailText");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultsSection = document.getElementById("results");

analyzeBtn.addEventListener("click", async () => {
  const text = emailText.value.trim();
  if (!text) {
    alert("Please enter an email before analyzing.");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const res = await fetch(`${API_BASE}/analyze_email`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) throw new Error("Analysis failed");

    const data = await res.json();
    renderResults(data);
    resultsSection.classList.remove("hidden");
  } catch (err) {
    console.error(err);
    alert("The analysis request failed. Make sure the API is running and a trained model exists.");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze";
  }
});

function renderResults(data) {
  document.getElementById("prediction").textContent = data.prediction;
  document.getElementById("probability").textContent =
    "Probability: " + (data.probability * 100).toFixed(1) + "%";
  document.getElementById("riskScore").textContent = data.risk_score.toFixed(0);
  const riskEl = document.getElementById("riskLevel");
  riskEl.textContent = data.risk_level;
  riskEl.className = "risk-level-" + data.risk_level.toLowerCase();

  const rc = data.risk_components;
  document.getElementById("sProb").textContent = rc.s_prob.toFixed(1);
  document.getElementById("sUrl").textContent = rc.s_url.toFixed(1);
  document.getElementById("sKw").textContent = rc.s_kw.toFixed(1);

  renderIndicatorList("topIndicatorsPos", data.top_indicators_pos || []);
  renderIndicatorList("topIndicatorsNeg", data.top_indicators_neg || []);
}

function renderIndicatorList(elementId, indicators) {
  const list = document.getElementById(elementId);
  list.innerHTML = "";
  indicators.forEach(function (item) {
    const li = document.createElement("li");
    li.innerHTML =
      "<span>" +
      item.word +
      "</span><span>" +
      (item.contribution * 100).toFixed(1) +
      "%</span>";
    list.appendChild(li);
  });
}
