// --- элементы ---
const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const dropText = document.getElementById("dropText");
const targetSelect = document.getElementById("target");
const runBtn = document.getElementById("run");
const statusEl = document.getElementById("status");
const results = document.getElementById("results");
const grid = document.querySelector(".grid");
const footInfo = document.getElementById("footInfo");
const predictInputs = document.getElementById("predictInputs");
const predictBtn = document.getElementById("predictBtn");
const predictOut = document.getElementById("predictOut");

let selectedFile = null;
let currentTarget = null;
let chart = null;

// --- выбор файла ---
drop.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  dropText.textContent = `[ ${file.name} ]`;
  const reader = new FileReader();
  reader.onload = () => {
    const header = reader.result.split(/\r?\n/)[0];
    const cols = header.split(",").map(c => c.trim()).filter(Boolean);
    targetSelect.innerHTML = cols
      .map((c, i) => `<option ${i === cols.length - 1 ? "selected" : ""}>${c}</option>`)
      .join("");
    targetSelect.disabled = false;
    runBtn.disabled = false;
  };
  reader.readAsText(file.slice(0, 65536));
}

// --- анализ ---
runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  setBusy(runBtn, true, "RUN");

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("target", targetSelect.value);

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analysis error");
    renderResults(data);
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.classList.add("error");
  } finally {
    setBusy(runBtn, false, "RUN");
  }
});

function setBusy(btn, busy, label) {
  btn.disabled = busy;
  btn.textContent = busy ? "..." : label;
  if (busy) { statusEl.textContent = ""; statusEl.classList.remove("error"); }
}

// --- отрисовка результата ---
function renderResults(data) {
  results.hidden = false;
  currentTarget = data.target;

  document.getElementById("bestFeature").textContent = data.best_feature;
  document.getElementById("bestScore").textContent = `${data.score_metric} ${data.best_score}`;
  document.getElementById("chartTitle").textContent = `${data.chart.x_label} → ${data.target}`;
  footInfo.textContent = `${data.task.toUpperCase()} / target: ${data.target}`;

  renderChart(data.chart);
  renderCards(data.feature_scores, data.best_feature, data.score_metric);
  buildPredictForm(data.features);
  predictOut.textContent = "";
}

function renderCards(scores, best, metric) {
  grid.querySelectorAll(".card--feature").forEach(el => el.remove());

  const entries = Object.entries(scores);
  // столбик по реальной силе: лучший = 100%, бесполезные (<=0) = почти пусто
  const maxPos = Math.max(...entries.map(([, v]) => v), 1e-9);

  for (const [name, val] of entries) {
    const pct = val > 0 ? Math.max(4, (val / maxPos) * 100) : 4;
    const isBest = name === best;
    const card = document.createElement("article");
    card.className = "card card--feature" + (isBest ? " card--best" : "");
    card.innerHTML = `
      <div class="card__thumb">
        <span class="score">${val.toFixed(2)}</span>
        <i style="height:${pct}%"></i>
      </div>
      <div class="card__meta">
        <span class="card__title">${name}</span>
        <span class="card__cat">${metric}</span>
      </div>`;
    grid.appendChild(card);
  }
}

// --- форма предсказания ---
function buildPredictForm(features) {
  predictInputs.innerHTML = features.map(f => {
    const input = f.type === "number"
      ? `<input data-name="${f.name}" type="number" step="any" placeholder="0" />`
      : `<select data-name="${f.name}">${f.options.map(o => `<option>${o}</option>`).join("")}</select>`;
    return `<div class="predict__field"><label>${f.name}</label>${input}</div>`;
  }).join("");
}

predictBtn.addEventListener("click", async () => {
  if (!selectedFile || !currentTarget) return;
  const values = {};
  predictInputs.querySelectorAll("[data-name]").forEach(el => {
    values[el.dataset.name] = el.value;
  });

  setBusy(predictBtn, true, "PREDICT");
  predictOut.textContent = "";

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("target", currentTarget);
  form.append("values", JSON.stringify(values));

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction error");
    predictOut.innerHTML = `Predicted ${data.target}: <b>${data.prediction}</b>`;
  } catch (err) {
    predictOut.textContent = err.message;
  } finally {
    setBusy(predictBtn, false, "PREDICT");
  }
});

// --- график ---
function renderChart(c) {
  if (chart) chart.destroy();
  const ctx = document.getElementById("chart");
  const grid_ = { color: "#e5e5e5" };
  const ticks = { color: "#8a8a8a", font: { family: "Space Mono" } };
  const common = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: grid_, ticks, title: { display: true, text: c.x_label, color: "#0b0b0b" } },
      y: { grid: grid_, ticks, title: { display: true, text: c.y_label, color: "#0b0b0b" } },
    },
  };

  if (c.type === "scatter") {
    chart = new Chart(ctx, {
      type: "scatter",
      data: { datasets: [{
        data: c.x.map((x, i) => ({ x, y: c.y[i] })),
        backgroundColor: "#0b0b0b",
        pointRadius: 3,
      }] },
      options: common,
    });
  } else {
    chart = new Chart(ctx, {
      type: "bar",
      data: { labels: c.labels, datasets: [{ data: c.values, backgroundColor: "#0b0b0b" }] },
      options: common,
    });
  }
}
