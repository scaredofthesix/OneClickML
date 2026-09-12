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
const datasetsGrid = document.getElementById("datasetsGrid");
const datasetsHint = document.getElementById("datasetsHint");
const historyBody = document.getElementById("historyBody");
const historyCount = document.getElementById("historyCount");

let selectedFile = null;
let currentTarget = null;
let chart = null;
let lastResult = null;
let lastChart = null;
let catalog = [];
let activeSlug = null;
let history = [];

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("theme", theme);
  document.querySelectorAll("#themeSwitch button").forEach((b) => {
    b.classList.toggle("is-active", b.dataset.theme === theme);
  });
  if (lastChart) renderChart(lastChart);
}

function applyLang(lang) {
  currentLang = lang;
  localStorage.setItem("lang", lang);
  document.documentElement.lang = lang;
  document.querySelectorAll("#langSwitch button").forEach((b) => {
    b.classList.toggle("is-active", b.dataset.lang === lang);
  });
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.innerHTML = t(el.dataset.i18n);
  });
  if (selectedFile) dropText.textContent = `[ ${selectedFile.name} ]`;
  if (lastResult) {
    footInfo.textContent = t("foot.info", {
      task: t(`task.${lastResult.task}`),
      target: lastResult.target,
    });
  }
  renderCatalog();
  renderHistory();
}

document.getElementById("themeSwitch").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (btn) applyTheme(btn.dataset.theme);
});

document.getElementById("langSwitch").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (btn) applyLang(btn.dataset.lang);
});

drop.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

function handleFile(file, preferredTarget) {
  if (!file) return;
  selectedFile = file;
  dropText.textContent = `[ ${file.name} ]`;
  const reader = new FileReader();
  reader.onload = () => {
    const header = reader.result.split(/\r?\n/)[0].replace(/^﻿/, "");
    const delims = [",", ";", "\t", "|"];
    const delim = delims.reduce((best, d) =>
      header.split(d).length > header.split(best).length ? d : best, ",");
    const cols = header.split(delim)
      .map((c) => c.trim().replace(/^"|"$/g, ""))
      .filter(Boolean);
    const wanted = preferredTarget && cols.includes(preferredTarget)
      ? preferredTarget
      : cols[cols.length - 1];
    targetSelect.innerHTML = cols
      .map((c) => `<option ${c === wanted ? "selected" : ""}>${c}</option>`)
      .join("");
    targetSelect.disabled = false;
    runBtn.disabled = false;
  };
  reader.readAsText(file.slice(0, 65536));
}

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  setBusy(runBtn, true, "run");

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("target", targetSelect.value);

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || t("error.analyze"));
    renderResults(data);
    loadHistory();
  } catch (err) {
    statusEl.textContent = err.message;
    statusEl.classList.add("error");
  } finally {
    setBusy(runBtn, false, "run");
  }
});

function setBusy(btn, busy, labelKey) {
  btn.disabled = busy;
  btn.textContent = busy ? "..." : t(labelKey);
  if (busy) {
    statusEl.textContent = "";
    statusEl.classList.remove("error");
  }
}

function renderResults(data) {
  lastResult = data;
  results.hidden = false;
  currentTarget = data.target;

  document.getElementById("bestFeature").textContent = data.best_feature;
  document.getElementById("bestScore").textContent = `${data.score_metric} ${data.best_score}`;
  document.getElementById("chartTitle").textContent = `${data.chart.x_label} → ${data.target}`;
  footInfo.textContent = t("foot.info", {
    task: t(`task.${data.task}`),
    target: data.target,
  });

  renderChart(data.chart);
  renderCards(data.feature_scores, data.best_feature, data.score_metric);
  buildPredictForm(data.features);
  predictOut.textContent = "";
}

function renderCards(scores, best, metric) {
  grid.querySelectorAll(".card--feature").forEach((el) => el.remove());

  const entries = Object.entries(scores);
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

function buildPredictForm(features) {
  predictInputs.innerHTML = features.map((f) => {
    const input = f.type === "number"
      ? `<input data-name="${f.name}" type="number" step="any" placeholder="0" />`
      : `<select data-name="${f.name}">${f.options.map((o) => `<option>${o}</option>`).join("")}</select>`;
    return `<div class="predict__field"><label>${f.name}</label>${input}</div>`;
  }).join("");
}

predictBtn.addEventListener("click", async () => {
  if (!selectedFile || !currentTarget) return;
  const values = {};
  predictInputs.querySelectorAll("[data-name]").forEach((el) => {
    values[el.dataset.name] = el.value;
  });

  setBusy(predictBtn, true, "predict.button");
  predictOut.textContent = "";

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("target", currentTarget);
  form.append("values", JSON.stringify(values));

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || t("error.predict"));
    predictOut.innerHTML = `${t("predict.result", { target: data.target })} <b>${data.prediction}</b>`;
  } catch (err) {
    predictOut.textContent = err.message;
  } finally {
    setBusy(predictBtn, false, "predict.button");
  }
});

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function renderChart(c) {
  lastChart = c;
  if (chart) chart.destroy();
  const ctx = document.getElementById("chart");
  const ink = css("--ink");
  const gridLine = { color: css("--line") };
  const ticks = { color: css("--muted"), font: { family: "Space Mono" } };
  const common = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: gridLine, ticks, title: { display: true, text: c.x_label, color: ink } },
      y: { grid: gridLine, ticks, title: { display: true, text: c.y_label, color: ink } },
    },
  };

  if (c.type === "scatter") {
    chart = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [{
          data: c.x.map((x, i) => ({ x, y: c.y[i] })),
          backgroundColor: ink,
          pointRadius: 3,
        }],
      },
      options: common,
    });
  } else {
    chart = new Chart(ctx, {
      type: "bar",
      data: { labels: c.labels, datasets: [{ data: c.values, backgroundColor: ink }] },
      options: common,
    });
  }
}

async function loadDatasets() {
  try {
    const res = await fetch("/api/datasets");
    const data = await res.json();
    if (!data.datasets.length) throw new Error("empty");
    catalog = data.datasets;
    renderCatalog();
  } catch (err) {
    catalog = [];
    datasetsHint.textContent = t("datasets.error");
    datasetsGrid.innerHTML = "";
  }
}

function renderCatalog() {
  if (!catalog.length) return;
  datasetsHint.textContent = t("datasets.hint", { n: catalog.length });
  datasetsGrid.innerHTML = catalog.map((d) => `
    <article class="ds${d.slug === activeSlug ? " ds--active" : ""}" data-slug="${d.slug}">
      <div class="ds__top">
        <span class="ds__title">${d.title[currentLang] || d.title.en}</span>
        <span class="ds__task">${t(`task.${d.task}`)}</span>
      </div>
      <p class="ds__why">${d.why[currentLang] || d.why.en}</p>
      <div class="ds__meta">
        <span>${t("target")}: ${d.target}</span>
        <span>${d.rows} x ${d.cols}</span>
      </div>
    </article>`).join("");

  datasetsGrid.querySelectorAll(".ds").forEach((card) => {
    card.addEventListener("click", () => pickDataset(card.dataset.slug));
  });
}

async function pickDataset(slug) {
  activeSlug = slug;
  datasetsGrid.querySelectorAll(".ds").forEach((c) => {
    c.classList.toggle("ds--active", c.dataset.slug === slug);
  });
  const res = await fetch(`/api/datasets/${slug}`);
  const data = await res.json();
  const file = new File([data.csv], `${data.slug}.csv`, { type: "text/csv" });
  handleFile(file, data.target);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function loadHistory() {
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    history = data.runs;
    renderHistory();
  } catch (err) {
    history = [];
    historyCount.textContent = "";
    historyBody.innerHTML = `<tr><td class="history__empty" colspan="6">${t("history.error")}</td></tr>`;
  }
}

function renderHistory() {
  if (!historyBody) return;
  historyCount.textContent = history.length ? t("history.count", { n: history.length }) : "";
  if (!history.length) {
    historyBody.innerHTML = `<tr><td class="history__empty" colspan="6">${t("history.empty")}</td></tr>`;
    return;
  }
  historyBody.innerHTML = history.map((r) => `
    <tr>
      <td>${r.created_at.replace("T", " ")}</td>
      <td>${r.filename}</td>
      <td>${r.target}</td>
      <td>${t(`task.${r.task}`)}</td>
      <td>${r.best_feature}</td>
      <td>${r.score_metric} ${r.best_score}</td>
    </tr>`).join("");
}

document.getElementById("historyClear").addEventListener("click", async () => {
  await fetch("/api/history", { method: "DELETE" });
  loadHistory();
});

applyTheme(localStorage.getItem("theme") || "light");
applyLang(currentLang);
loadDatasets();
loadHistory();
