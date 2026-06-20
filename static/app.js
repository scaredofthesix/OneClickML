// --- элементы ---
const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const dropText = document.getElementById("dropText");
const targetField = document.getElementById("targetField");
const targetSelect = document.getElementById("target");
const runBtn = document.getElementById("run");
const statusEl = document.getElementById("status");
const results = document.getElementById("results");
const stage = document.querySelector(".stage");
const caption = document.getElementById("robotCaption");
const video = document.getElementById("robot");

let selectedFile = null;
let chart = null;

// --- выбор файла ---
drop.addEventListener("click", () => fileInput.click());
["dragover", "dragenter"].forEach(e =>
  drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(e =>
  drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", ev => handleFile(ev.dataTransfer.files[0]));
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  dropText.textContent = file.name;
  // читаем первую строку, чтобы вытащить названия колонок для выбора таргета
  const reader = new FileReader();
  reader.onload = () => {
    const header = reader.result.split(/\r?\n/)[0];
    const cols = header.split(",").map(c => c.trim()).filter(Boolean);
    targetSelect.innerHTML = cols
      .map((c, i) => `<option ${i === cols.length - 1 ? "selected" : ""}>${c}</option>`)
      .join("");
    targetField.hidden = false;
    runBtn.disabled = false;
  };
  reader.readAsText(file.slice(0, 65536));
}

// --- состояния робота через перемотку видео ---
// первый кадр = профиль (думает), последний = смотрит на нас (покой)
function scrubTo(targetTime, ms) {
  return new Promise(resolve => {
    if (!video.duration) return resolve();
    const start = video.currentTime;
    const t0 = performance.now();
    video.pause();
    (function step(now) {
      const p = Math.min(1, (now - t0) / ms);
      video.currentTime = start + (targetTime - start) * p;
      p < 1 ? requestAnimationFrame(step) : resolve();
    })(t0);
  });
}

function playForward() {
  return new Promise(resolve => {
    video.currentTime = 0;
    const done = () => { video.removeEventListener("ended", done); resolve(); };
    video.addEventListener("ended", done);
    video.play();
  });
}

const turnAway = () => scrubTo(0, 900);                  // лицо -> профиль
const turnBack = () => playForward();                    // профиль -> лицо

// интро при загрузке: повернулся к нам и замер
video.addEventListener("loadeddata", () => playForward(), { once: true });

// --- анализ ---
runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  setBusy(true);
  stage.classList.add("thinking");
  caption.textContent = "Думаю над данными...";

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("target", targetSelect.value);

  try {
    const [, res] = await Promise.all([
      turnAway(),
      fetch("/api/analyze", { method: "POST", body: form }),
    ]);
    const data = await res.json();
    stage.classList.remove("thinking");

    if (!res.ok) throw new Error(data.detail || "Ошибка анализа");

    renderResults(data);
    caption.textContent = `Лучший признак: ${data.best_feature}`;
    await turnBack();
  } catch (err) {
    stage.classList.remove("thinking");
    statusEl.textContent = err.message;
    statusEl.classList.add("error");
    caption.textContent = "Что-то пошло не так";
    await turnBack();
  } finally {
    setBusy(false);
  }
});

function setBusy(busy) {
  runBtn.disabled = busy;
  runBtn.textContent = busy ? "Анализирую..." : "Анализировать";
  if (busy) { statusEl.textContent = ""; statusEl.classList.remove("error"); }
}

// --- отрисовка результата ---
function renderResults(data) {
  document.getElementById("bestFeature").textContent = data.best_feature;
  document.getElementById("bestScore").textContent = data.best_score;
  document.getElementById("scoreMetric").textContent = data.score_metric;
  renderScores(data.feature_scores);
  renderChart(data.chart);
  results.hidden = false;
}

function renderScores(scores) {
  const entries = Object.entries(scores);
  const max = Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-9);
  document.getElementById("scoreList").innerHTML = entries
    .map(([name, val]) => `
      <li>
        <div class="row"><span>${name}</span><span>${val.toFixed(3)}</span></div>
        <div class="bar"><i style="width:${Math.max(0, (val / max) * 100)}%"></i></div>
      </li>`)
    .join("");
}

function renderChart(c) {
  if (chart) chart.destroy();
  const ctx = document.getElementById("chart");
  const grid = { color: "rgba(255,255,255,0.06)" };
  const ticks = { color: "#8a94a3" };
  const common = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid, ticks, title: { display: true, text: c.x_label, color: "#8a94a3" } },
      y: { grid, ticks, title: { display: true, text: c.y_label, color: "#8a94a3" } },
    },
  };

  if (c.type === "scatter") {
    chart = new Chart(ctx, {
      type: "scatter",
      data: { datasets: [{
        data: c.x.map((x, i) => ({ x, y: c.y[i] })),
        backgroundColor: "#00e5ff",
        pointRadius: 4,
      }] },
      options: common,
    });
  } else {
    chart = new Chart(ctx, {
      type: "bar",
      data: { labels: c.labels, datasets: [{ data: c.values, backgroundColor: "#00e5ff" }] },
      options: common,
    });
  }
}
