"use strict";

/* ---------------------------------------------------------------------------
   Spam Scanner front end – talks to the Flask API in ../app.py
--------------------------------------------------------------------------- */
const $ = (id) => document.getElementById(id);

const API = { health: "/health", predict: "/predict", batch: "/predict_batch" };
const HISTORY_KEY = "spamscan.history.v1";
const THRESHOLD_KEY = "spamscan.threshold.v1";
const MAX_HISTORY = 200;
const SHOW_HISTORY = 50;

const SAMPLES = {
  spam: "Subject: You've won a free prize!!!\n\nCongratulations! You have been selected to receive a $1000 gift card. Click the link below to claim your reward before it expires. Act now - limited time offer!",
  ham: "Hi Priya,\n\nThanks for sending the report. I've gone through it and have a few comments. Can we meet tomorrow at 3 pm to discuss before the review?\n\nRegards,\nArun",
};

const state = { threshold: 0.5, history: [], lastBatch: [] };

/* ----------------------------- helpers ----------------------------------- */
async function post(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

const pct = (p) => `${(p * 100).toFixed(1)}%`;
const verdictClass = (v) => v.toLowerCase();
const snippet = (s, n = 90) => (s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s).replace(/\s+/g, " ");

function showError(el, message) {
  el.textContent = message;
  el.hidden = !message;
}

function setBusy(button, busy, idle, working) {
  button.disabled = busy;
  button.textContent = busy ? working : idle;
}

function makeBadge(verdict) {
  const span = document.createElement("span");
  span.className = `badge ${verdictClass(verdict)}`;
  span.textContent = verdict;
  return span;
}

function cell(text, className) {
  const td = document.createElement("td");
  if (className) td.className = className;
  if (text instanceof Node) td.appendChild(text);
  else td.textContent = text;
  return td;
}

/* CSV export (escapes quotes and neutralises spreadsheet formulas) */
function csvValue(v) {
  let s = String(v ?? "");
  if (/^[=+\-@\t\r]/.test(s)) s = "'" + s;
  return `"${s.replace(/"/g, '""')}"`;
}
function downloadCsv(filename, headers, rows) {
  const lines = [headers.map(csvValue).join(",")].concat(rows.map((r) => r.map(csvValue).join(",")));
  const blob = new Blob(["\ufeff" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const stamp = () => new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");

/* ----------------------------- model status ------------------------------ */
async function checkHealth() {
  const pill = $("statusPill");
  try {
    const res = await fetch(API.health, { cache: "no-store" });
    const data = await res.json();
    if (data.status === "ready") {
      pill.dataset.state = "ready";
      $("statusText").textContent = "model ready";
      pill.title = "";
    } else {
      pill.dataset.state = "error";
      $("statusText").textContent = "model offline";
      pill.title = data.error || "";
    }
  } catch {
    pill.dataset.state = "error";
    $("statusText").textContent = "server unreachable";
  }
}

/* ----------------------------- tabs -------------------------------------- */
function selectTab(name) {
  const single = name === "single";
  $("panelSingle").hidden = !single;
  $("panelBatch").hidden = single;
  $("tabSingle").classList.toggle("active", single);
  $("tabBatch").classList.toggle("active", !single);
  $("tabSingle").setAttribute("aria-selected", String(single));
  $("tabBatch").setAttribute("aria-selected", String(!single));
}

/* ----------------------------- threshold --------------------------------- */
function setThreshold(value) {
  state.threshold = Math.min(0.95, Math.max(0.05, value));
  $("threshold").value = Math.round(state.threshold * 100);
  $("thresholdValue").textContent = state.threshold.toFixed(2);
  try { localStorage.setItem(THRESHOLD_KEY, String(state.threshold)); } catch { /* storage unavailable */ }
}

/* ----------------------------- single scan ------------------------------- */
function renderResult(r) {
  $("result").hidden = false;
  const badge = $("verdictBadge");
  badge.textContent = r.verdict;
  badge.className = `badge ${verdictClass(r.verdict)}`;
  $("percent").textContent = pct(r.probability);

  const colour = { Spam: "var(--spam)", Borderline: "var(--border)", Safe: "var(--safe)" }[r.verdict];
  const fill = $("meterFill");
  fill.style.background = colour;
  fill.style.width = "0";
  requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${(r.probability * 100).toFixed(1)}%`; }));

  $("meterMarker").style.left = `${r.threshold * 100}%`;
  $("markerLabel").textContent = `threshold ${Math.round(r.threshold * 100)}%`;

  $("explain").textContent = {
    Spam: "The model is confident this message is spam. Avoid clicking links or replying.",
    Borderline: "The message is close to the spam threshold. Treat links and attachments with caution.",
    Safe: "The message looks like a normal message. Always stay careful with unexpected links.",
  }[r.verdict];
}

async function scanSingle() {
  const text = $("messageInput").value.trim();
  showError($("error"), "");
  if (!text) { showError($("error"), "Please enter a message to scan."); return; }
  setBusy($("scanBtn"), true, "Scan message", "Scanning…");
  try {
    const r = await post(API.predict, { text, threshold: state.threshold });
    renderResult(r);
    addHistory({ text, probability: r.probability, verdict: r.verdict, threshold: r.threshold });
  } catch (err) {
    $("result").hidden = true;
    showError($("error"), err.message);
  } finally {
    setBusy($("scanBtn"), false, "Scan message", "Scanning…");
    checkHealth();
  }
}

function clearSingle() {
  $("messageInput").value = "";
  $("charCount").textContent = "0 characters";
  $("result").hidden = true;
  showError($("error"), "");
  $("messageInput").focus();
}

/* ----------------------------- batch scan -------------------------------- */
function batchLines() {
  return $("batchInput").value.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
}

async function scanBatch() {
  const messages = batchLines();
  showError($("batchError"), "");
  if (!messages.length) { showError($("batchError"), "Enter at least one message (one per line)."); return; }
  setBusy($("batchBtn"), true, "Scan batch", "Scanning…");
  try {
    const data = await post(API.batch, { messages, threshold: state.threshold });
    state.lastBatch = data.results;
    renderBatch(data);
    data.results.forEach((r) => addHistory({ text: r.text, probability: r.probability, verdict: r.verdict, threshold: data.threshold }, false));
    saveHistory();
    renderHistory();
  } catch (err) {
    showError($("batchError"), err.message);
  } finally {
    setBusy($("batchBtn"), false, "Scan batch", "Scanning…");
    checkHealth();
  }
}

function renderBatch(data) {
  const s = data.summary;
  const chips = $("batchSummary");
  chips.replaceChildren();
  [["Total", s.total, ""], ["Spam", s.spam, "spam"], ["Borderline", s.borderline, "borderline"], ["Safe", s.safe, "safe"]].forEach(([label, n, cls]) => {
    const chip = document.createElement("span");
    chip.className = `chip ${cls}`;
    chip.append(label);
    const b = document.createElement("b");
    b.textContent = n;
    chip.appendChild(b);
    chips.appendChild(chip);
  });
  chips.hidden = false;

  const body = $("batchBody");
  body.replaceChildren();
  data.results.forEach((r) => {
    const tr = document.createElement("tr");
    tr.append(cell(r.index, "num"), cell(snippet(r.text, 110), "msg"), cell(pct(r.probability), "num"), cell(makeBadge(r.verdict)));
    tr.children[1].title = r.text;
    body.appendChild(tr);
  });
  $("batchTableWrap").hidden = false;
  $("batchExport").disabled = false;
}

function exportBatch() {
  if (!state.lastBatch.length) return;
  downloadCsv(`spam-batch-${stamp()}.csv`, ["index", "message", "spam_probability", "verdict"],
    state.lastBatch.map((r) => [r.index, r.text, r.probability, r.verdict]));
}

function loadFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const lines = String(reader.result).split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    $("batchInput").value = lines.join("\n");
    updateBatchCount();
  };
  reader.onerror = () => showError($("batchError"), "Could not read that file.");
  reader.readAsText(file);
}

function updateBatchCount() {
  const n = batchLines().length;
  $("batchCount").textContent = `${n} message${n === 1 ? "" : "s"}`;
}

/* ----------------------------- history ----------------------------------- */
function loadHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    state.history = Array.isArray(parsed) ? parsed : [];
  } catch { state.history = []; }
}
function saveHistory() {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history)); } catch { /* storage full/unavailable */ }
}
function addHistory(entry, persist = true) {
  state.history.unshift({ time: new Date().toISOString(), ...entry });
  if (state.history.length > MAX_HISTORY) state.history.length = MAX_HISTORY;
  if (persist) { saveHistory(); renderHistory(); }
}
function renderHistory() {
  const body = $("historyBody");
  body.replaceChildren();
  state.history.slice(0, SHOW_HISTORY).forEach((h) => {
    const tr = document.createElement("tr");
    tr.append(cell(new Date(h.time).toLocaleString(), "num"), cell(snippet(h.text), "msg"), cell(pct(h.probability), "num"), cell(makeBadge(h.verdict)));
    tr.children[1].title = h.text;
    body.appendChild(tr);
  });
  $("historyEmpty").hidden = state.history.length > 0;
  $("historyExport").disabled = $("historyClear").disabled = state.history.length === 0;
}
function exportHistory() {
  downloadCsv(`spam-history-${stamp()}.csv`, ["time", "message", "spam_probability", "verdict", "threshold"],
    state.history.map((h) => [h.time, h.text, h.probability, h.verdict, h.threshold]));
}
function clearHistory() {
  if (!state.history.length || !confirm("Delete all saved scan history from this browser?")) return;
  state.history = [];
  saveHistory();
  renderHistory();
}

/* ----------------------------- wiring ------------------------------------ */
function init() {
  try {
    const saved = parseFloat(localStorage.getItem(THRESHOLD_KEY));
    if (!Number.isNaN(saved)) state.threshold = saved;
  } catch { /* ignore */ }
  setThreshold(state.threshold);
  loadHistory();
  renderHistory();
  updateBatchCount();

  $("tabSingle").addEventListener("click", () => selectTab("single"));
  $("tabBatch").addEventListener("click", () => selectTab("batch"));
  $("scanBtn").addEventListener("click", scanSingle);
  $("clearBtn").addEventListener("click", clearSingle);
  $("sampleSpam").addEventListener("click", () => { $("messageInput").value = SAMPLES.spam; $("messageInput").dispatchEvent(new Event("input")); });
  $("sampleHam").addEventListener("click", () => { $("messageInput").value = SAMPLES.ham; $("messageInput").dispatchEvent(new Event("input")); });
  $("messageInput").addEventListener("input", () => {
    const n = $("messageInput").value.length;
    $("charCount").textContent = `${n.toLocaleString()} character${n === 1 ? "" : "s"}`;
  });
  $("messageInput").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") scanSingle(); });

  $("batchBtn").addEventListener("click", scanBatch);
  $("batchClear").addEventListener("click", () => {
    $("batchInput").value = "";
    updateBatchCount();
    $("batchSummary").hidden = $("batchTableWrap").hidden = true;
    $("batchExport").disabled = true;
    state.lastBatch = [];
    showError($("batchError"), "");
  });
  $("batchInput").addEventListener("input", updateBatchCount);
  $("fileInput").addEventListener("change", (e) => { loadFile(e.target.files[0]); e.target.value = ""; });
  $("batchExport").addEventListener("click", exportBatch);

  $("threshold").addEventListener("input", (e) => setThreshold(Number(e.target.value) / 100));
  $("historyExport").addEventListener("click", exportHistory);
  $("historyClear").addEventListener("click", clearHistory);

  checkHealth();
  setInterval(checkHealth, 15000);
}

document.addEventListener("DOMContentLoaded", init);
