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

let selectedFile = null;
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
  setBusy(true);

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
    setBusy(false);
  }
});

function setBusy(busy) {
  runBtn.disabled = busy;
  runBtn.textContent = busy ? "..." : "RUN";
  if (busy) { statusEl.textContent = ""; statusEl.classList.remove("error"); }
}

// --- отрисовка ---
function renderResults(data) {
  results.hidden = false;
  document.getElementById("bestFeature").textContent = data.best_feature;
  document.getElementById("bestCat").textContent =
    `BEST FEATURE / ${data.score_metric} ${data.best_score}`;
  footInfo.textContent = `${data.task.toUpperCase()} / target: ${data.target}`;

  renderChart(data.chart);
  renderCards(data.feature_scores, data.best_feature, data.score_metric);
}

function renderCards(scores, best, metric) {
  // убираем старые карточки (всё, кроме первой широкой с графиком)
  grid.querySelectorAll(".card--feature").forEach(el => el.remove());

  const entries = Object.entries(scores);
  const max = Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-9);

  for (const [name, val] of entries) {
    const pct = Math.max(2, (Math.abs(val) / max) * 100);
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
