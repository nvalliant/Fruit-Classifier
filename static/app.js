/**
 * RIPE.AI — app.js
 * Photo Scanner + Realtime Camera Scanner
 * Flask API (server.py) → JS fallback
 */
"use strict";

/* ═══════════════════════════════════════════
   CONFIG
═══════════════════════════════════════════ */
const API_BASE =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? "http://localhost:5000"
    : "https://fruit-classifier-three.vercel.app/";
const API_PREDICT = `${API_BASE}/predict`;
const API_HEALTH = `${API_BASE}/health`;
let serverAvailable = false;

/* ═══════════════════════════════════════════
   1. THEME TOGGLE
═══════════════════════════════════════════ */
const themeToggle = document.getElementById("themeToggle");
const html = document.documentElement;
html.setAttribute("data-theme", localStorage.getItem("ripeai-theme") || "dark");
themeToggle.addEventListener("click", () => {
  const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
  html.setAttribute("data-theme", next);
  localStorage.setItem("ripeai-theme", next);
});

/* ═══════════════════════════════════════════
   2. SERVER HEALTH CHECK (silent)
═══════════════════════════════════════════ */
async function checkServer() {
  try {
    const res = await fetch(API_HEALTH, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    serverAvailable = data.status === "ok" && data.model_loaded;
  } catch {
    serverAvailable = false;
  }
}
checkServer();
setInterval(checkServer, 10_000);

/* ═══════════════════════════════════════════
   3. FEATURE EXTRACTION — JS mirror of Cell 2
═══════════════════════════════════════════ */
function toGray(data, w, h) {
  const g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++)
    g[i] =
      0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  return g;
}
function blur5(gray, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0,
        c = 0;
      for (let dy = -2; dy <= 2; dy++)
        for (let dx = -2; dx <= 2; dx++) {
          const ny = y + dy,
            nx = x + dx;
          if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
            s += gray[ny * w + nx];
            c++;
          }
        }
      out[y * w + x] = s / c;
    }
  return out;
}
function otsu(gray) {
  const hist = new Int32Array(256);
  for (const v of gray) hist[Math.min(255, Math.floor(v))]++;
  const tot = gray.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0,
    wB = 0,
    mx = 0,
    thresh = 0;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (!wB) continue;
    const wF = tot - wB;
    if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB,
      mF = (sum - sumB) / wF,
      v = wB * wF * (mB - mF) ** 2;
    if (v > mx) {
      mx = v;
      thresh = t;
    }
  }
  const mask = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) mask[i] = gray[i] > thresh ? 1 : 0;
  return mask;
}
function shapeFeatures(mask, w, h) {
  let x0 = w,
    x1 = 0,
    y0 = h,
    y1 = 0,
    area = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      area++;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  if (!area) return { aspect_ratio: 1, extent: 0 };
  const bw = x1 - x0 + 1,
    bh = y1 - y0 + 1;
  return {
    aspect_ratio: bh > 0 ? bw / bh : 1,
    extent: bw * bh > 0 ? area / (bw * bh) : 0,
  };
}
function rgbHSV(data, w, h) {
  const H = new Float32Array(w * h),
    S = new Float32Array(w * h),
    V = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = data[i * 4] / 255,
      g = data[i * 4 + 1] / 255,
      b = data[i * 4 + 2] / 255;
    const mx = Math.max(r, g, b),
      mn = Math.min(r, g, b),
      d = mx - mn;
    let hue = 0;
    if (d > 0) {
      if (mx === r) hue = ((((g - b) / d) % 6) + 6) % 6;
      else if (mx === g) hue = (b - r) / d + 2;
      else hue = (r - g) / d + 4;
      hue = (hue * 60 + 360) % 360;
    }
    H[i] = (hue / 360) * 180;
    S[i] = mx > 0 ? (d / mx) * 255 : 0;
    V[i] = mx * 255;
  }
  return { H, S, V };
}
function mstat(ch, mask) {
  const v = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) v.push(ch[i]);
  if (!v.length) return { mean: 0, std: 0 };
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  return {
    mean: m,
    std: Math.sqrt(v.reduce((a, x) => a + (x - m) ** 2, 0) / v.length),
  };
}
function glcm(mg, w, h) {
  const L = 256,
    q = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++)
    q[i] = Math.min(L - 1, Math.floor((mg[i] / 256) * L));
  const g0 = Array.from({ length: L }, () => new Float32Array(L));
  const g45 = Array.from({ length: L }, () => new Float32Array(L));
  let t0 = 0,
    t45 = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (x + 5 < w) {
        const j = y * w + x + 5,
          a = q[i],
          b = q[j];
        g0[a][b]++;
        g0[b][a]++;
        t0 += 2;
      }
      if (x + 5 < w && y - 5 >= 0) {
        const j = (y - 5) * w + (x + 5),
          a = q[i],
          b = q[j];
        g45[a][b]++;
        g45[b][a]++;
        t45 += 2;
      }
    }
  if (t0)
    for (let a = 0; a < L; a++) for (let b = 0; b < L; b++) g0[a][b] /= t0;
  if (t45)
    for (let a = 0; a < L; a++) for (let b = 0; b < L; b++) g45[a][b] /= t45;
  function props(g) {
    let mi = 0,
      mj = 0;
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++) {
        mi += i * g[i][j];
        mj += j * g[i][j];
      }
    let si = 0,
      sj = 0;
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++) {
        si += (i - mi) ** 2 * g[i][j];
        sj += (j - mj) ** 2 * g[i][j];
      }
    si = Math.sqrt(si);
    sj = Math.sqrt(sj);
    let ct = 0,
      co = 0,
      en = 0,
      ho = 0;
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++) {
        const p = g[i][j];
        ct += (i - j) ** 2 * p;
        en += p * p;
        ho += p / (1 + Math.abs(i - j));
        if (si > 0 && sj > 0) co += ((i - mi) * (j - mj) * p) / (si * sj);
      }
    return { ct, co, en, ho };
  }
  const p0 = props(g0),
    p1 = props(g45);
  return {
    contrast: (p0.ct + p1.ct) / 2,
    correlation: (p0.co + p1.co) / 2,
    energy: (p0.en + p1.en) / 2,
    homogeneity: (p0.ho + p1.ho) / 2,
  };
}
function extractFeaturesJS(imgData, w, h) {
  const gray = toGray(imgData, w, h),
    blr = blur5(gray, w, h),
    mask = otsu(blr);
  const { aspect_ratio, extent } = shapeFeatures(mask, w, h);
  const { H, S, V } = rgbHSV(imgData, w, h);
  const { mean: h_mean, std: h_std } = mstat(H, mask);
  const { mean: s_mean, std: s_std } = mstat(S, mask);
  const { mean: v_mean, std: v_std } = mstat(V, mask);
  const mg = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) mg[i] = mask[i] ? gray[i] : 0;
  const { contrast, correlation, energy, homogeneity } = glcm(mg, w, h);
  const raw = {
    h_mean,
    s_mean,
    v_mean,
    h_std,
    s_std,
    v_std,
    contrast,
    correlation,
    energy,
    homogeneity,
    aspect_ratio,
    extent,
  };
  return raw;
}

/* ═══════════════════════════════════════════
   4. JS FALLBACK CLASSIFIER — REMOVED
   The trained SVM model cannot run in the browser.
   All classification must go through the Flask server.
   If the server is unavailable, we show an error rather than
   returning silently wrong predictions from heuristic rules.
═══════════════════════════════════════════ */
function classifyJS() {
  throw new Error(
    "Server tidak tersedia. Pastikan Flask server berjalan di " + API_BASE,
  );
}

const ALL_CLASSES = [
  "banana_ripe",
  "banana_unripe",
  "banana_rotten",
  "strawberry_ripe",
  "strawberry_unripe",
  "strawberry_rotten",
  "orange_ripe",
  "orange_unripe",
  "orange_rotten",
];
const FRUIT_EMOJI = {
  banana: "🍌",
  strawberry: "🍓",
  orange: "🍊",
};
const RIPENESS_ICON = { ripe: "✅", unripe: "🌱", rotten: "⚠️" };
const RIPENESS_COLOR = {
  ripe: "#22c55e",
  unripe: "#f59e0b",
  rotten: "#ef4444",
};

/* ═══════════════════════════════════════════
   5. CANVAS HELPER — extract from ImageData
═══════════════════════════════════════════ */
function analyzeImageData(imgData, w, h) {
  const raw = extractFeaturesJS(imgData, w, h);
  const result = classifyJS(raw);
  result.features = raw;
  return result;
}
function analyzeFromCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  return analyzeImageData(imgData, canvas.width, canvas.height);
}

/* ═══════════════════════════════════════════
   6. SERVER API CALL
═══════════════════════════════════════════ */
async function callServerAPI(dataUrl) {
  const res = await fetch(API_PREDICT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl }),
    signal: AbortSignal.timeout(30_000),
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error(e.error || `Server error ${res.status}`);
  }
  const data = await res.json();
  return {
    fruitType: data.fruit_type,
    ripenessStage: data.ripeness_stage,
    classLabel: data.prediction,
    confidence: data.confidence,
    probs: data.probabilities,
    features: data.features,
    source: "python_svm_model",
  };
}

/* ═══════════════════════════════════════════
   7. PHOTO SCANNER UI
═══════════════════════════════════════════ */
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const resultCard = document.getElementById("resultCard");
const previewImg = document.getElementById("previewImg");
const scanOverlay = document.getElementById("scanOverlay");
const liveBadge = document.getElementById("liveBadge");
const loadingState = document.getElementById("loadingState");
const loadingText = document.getElementById("loadingText");
const progressFill = document.getElementById("progressFill");
const stepsList = document.getElementById("stepsList");
const analysisResults = document.getElementById("analysisResults");
const resetBtn = document.getElementById("resetBtn");
const sourceTag = document.getElementById("sourceTag");
const fruitTypeEl = document.getElementById("fruitType");
const ripenessDot = document.getElementById("ripenessDot");
const ripenessStatusEl = document.getElementById("ripenessStatus");
const confidencePct = document.getElementById("confidencePct");
const confFill = document.getElementById("confFill");
const probsList = document.getElementById("probsList");
const featuresChips = document.getElementById("featuresChips");

const STEPS_SERVER = [
  { label: "Encoding gambar ke base64", pct: 12 },
  { label: "Mengirim ke Flask API", pct: 25 },
  { label: "Resize 128×128 + BGR2GRAY", pct: 38 },
  { label: "GaussianBlur + Otsu threshold", pct: 52 },
  { label: "HSV color feature extraction", pct: 65 },
  { label: "GLCM texture features", pct: 78 },
  { label: "StandardScaler + SVM prediction", pct: 94 },
  { label: "Menampilkan hasil...", pct: 100 },
];
const STEPS_JS = [
  { label: "Memeriksa koneksi ke server...", pct: 50 },
  { label: "Server tidak tersedia", pct: 100 },
];

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () =>
  dropZone.classList.remove("drag-over"),
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) handleImage(f);
});
dropZone.addEventListener("click", (e) => {
  if (e.target === uploadBtn || uploadBtn.contains(e.target)) return;
  fileInput.click();
});
uploadBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleImage(fileInput.files[0]);
});
resetBtn.addEventListener("click", resetUI);

function handleImage(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.onload = () => startPhotoAnalysis(e.target.result);
  };
  reader.readAsDataURL(file);
}

async function startPhotoAnalysis(dataUrl) {
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  scanOverlay.hidden = false;
  liveBadge.hidden = false;
  show(loadingState);
  hide(analysisResults);
  progressFill.style.width = "0%";
  const steps = serverAvailable ? STEPS_SERVER : STEPS_JS;
  loadingText.textContent = steps[0].label;
  renderSteps(steps, -1);
  let idx = 0;
  const iv = setInterval(() => {
    if (idx < steps.length) {
      loadingText.textContent = steps[idx].label;
      progressFill.style.width = steps[idx].pct + "%";
      renderSteps(steps, idx);
      idx++;
    } else clearInterval(iv);
  }, 280);
  try {
    let result;
    if (serverAvailable) {
      result = await callServerAPI(dataUrl);
    } else {
      throw new Error(
        "Server tidak tersedia. Jalankan Flask server di " +
          API_BASE +
          " lalu muat ulang halaman ini.",
      );
    }
    await sleep(Math.max(0, steps.length * 280 + 100 - idx * 280));
    clearInterval(iv);
    renderSteps(steps, steps.length);
    await sleep(350);
    scanOverlay.hidden = true;
    hide(loadingState);
    showPhotoResults(result);
  } catch (err) {
    clearInterval(iv);
    scanOverlay.hidden = true;
    hide(loadingState);
    show(analysisResults);
    if (fruitTypeEl) fruitTypeEl.textContent = "⚠️ Error";
    if (ripenessStatusEl) ripenessStatusEl.textContent = err.message;
  }
}

function renderSteps(steps, activeIdx) {
  stepsList.innerHTML = steps
    .map(
      (s, i) =>
        `<div class="loading-step-item ${i < activeIdx ? "done" : i === activeIdx ? "active" : ""}"><span class="loading-step-item__dot"></span><span>${s.label}</span></div>`,
    )
    .join("");
}

function showPhotoResults(result) {
  show(analysisResults);
  if (sourceTag) {
    sourceTag.textContent =
      result.source === "python_svm_model"
        ? "🐍 Python SVM Model"
        : "⚠️ Server Error";
    sourceTag.className = `source-tag source-tag--${result.source === "python_svm_model" ? "python" : "js"}`;
    sourceTag.hidden = false;
  }
  fruitTypeEl.textContent = `${FRUIT_EMOJI[result.fruitType] || "🍑"} ${cap(result.fruitType)}`;
  ripenessDot.style.background =
    RIPENESS_COLOR[result.ripenessStage] || "#3B82F6";
  ripenessStatusEl.textContent = `${RIPENESS_ICON[result.ripenessStage]} ${cap(result.ripenessStage)}`;
  const pct = Math.round(result.confidence * 100);
  confidencePct.textContent = pct + "%";
  requestAnimationFrame(() => {
    confFill.style.width = pct + "%";
  });
  const sorted = Object.entries(result.probs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);
  probsList.innerHTML = sorted
    .map(([cls, prob], i) => {
      const p = Math.round(prob * 100),
        [ft, rs] = cls.split("_");
      return `<div class="prob-row"><span class="prob-row__name">${FRUIT_EMOJI[ft] || "🍑"} ${cap(ft)} ${cap(rs)}</span><div class="prob-row__track"><div class="prob-row__fill ${i === 0 ? "prob-row__fill--top" : ""}" style="width:0%" data-pct="${p}"></div></div><span class="prob-row__pct">${p}%</span></div>`;
    })
    .join("");
  setTimeout(
    () =>
      document.querySelectorAll(".prob-row__fill").forEach((el) => {
        el.style.width = el.dataset.pct + "%";
      }),
    60,
  );
  const raw = result.features || {};
  const chips = [
    { n: "H mean", v: fmtN(raw.h_mean, 1) },
    { n: "S mean", v: fmtN(raw.s_mean, 1) },
    { n: "V mean", v: fmtN(raw.v_mean, 1) },
    { n: "H std", v: fmtN(raw.h_std, 2) },
    { n: "S std", v: fmtN(raw.s_std, 2) },
    { n: "V std", v: fmtN(raw.v_std, 2) },
    { n: "Contrast", v: fmtN(raw.contrast, 4) },
    { n: "Correlation", v: fmtN(raw.correlation, 4) },
    { n: "Energy", v: fmtN(raw.energy, 4) },
    { n: "Homogeneity", v: fmtN(raw.homogeneity, 4) },
    { n: "Aspect Ratio", v: fmtN(raw.aspect_ratio, 3) },
    { n: "Extent", v: fmtN(raw.extent, 3) },
  ];
  featuresChips.innerHTML = chips
    .map(
      (c) => `<div class="feature-chip">${c.n}: <strong>${c.v}</strong></div>`,
    )
    .join("");
}

function resetUI() {
  resultCard.hidden = true;
  liveBadge.hidden = true;
  scanOverlay.hidden = true;
  hide(loadingState);
  hide(analysisResults);
  progressFill.style.width = "0%";
  confFill.style.width = "0%";
  previewImg.src = "";
  fileInput.value = "";
  if (sourceTag) sourceTag.hidden = true;
}

/* ═══════════════════════════════════════════
   8. REALTIME CAMERA SCANNER
═══════════════════════════════════════════ */
const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const cameraSelect = document.getElementById("cameraSelect");
const cameraVideo = document.getElementById("cameraVideo");
const cameraIdle = document.getElementById("cameraIdle");
const cameraOverlay = document.getElementById("cameraOverlay");
const cameraLiveResult = document.getElementById("cameraLiveResult");
const liveFruit = document.getElementById("liveFruit");
const liveRipeness = document.getElementById("liveRipeness");
const liveConf = document.getElementById("liveConf");
const realtimeIdle = document.getElementById("realtimeIdle");
const realtimeData = document.getElementById("realtimeData");
const rtFruitType = document.getElementById("rtFruitType");
const rtRipenessDot = document.getElementById("rtRipenessDot");
const rtRipenessStatus = document.getElementById("rtRipenessStatus");
const rtConfPct = document.getElementById("rtConfPct");
const rtConfFill = document.getElementById("rtConfFill");
const rtProbsList = document.getElementById("rtProbsList");
const rtFpsEl = document.getElementById("rtFps");

let stream = null;
let realtimeInterval = null;
let lastFrameTime = 0;
let frameCount = 0;
let fpsAccum = 0;

startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", stopCamera);
captureBtn.addEventListener("click", captureFrame);
cameraSelect.addEventListener("change", switchCamera);

async function startCamera() {
  try {
    // Populate camera list first
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput");
    if (cams.length > 1) {
      cameraSelect.innerHTML = cams
        .map(
          (d, i) =>
            `<option value="${d.deviceId}">${d.label || "Camera " + (i + 1)}</option>`,
        )
        .join("");
      cameraSelect.hidden = false;
    }

    const deviceId = cameraSelect.value || undefined;
    stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId
        ? {
            deviceId: { exact: deviceId },
            width: { ideal: 640 },
            height: { ideal: 480 },
          }
        : { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });

    cameraVideo.srcObject = stream;
    cameraVideo.hidden = false;
    cameraIdle.hidden = true;
    cameraOverlay.hidden = false;

    startCameraBtn.hidden = true;
    stopCameraBtn.hidden = false;
    captureBtn.hidden = false;

    // Start continuous analysis
    frameCount = 0;
    fpsAccum = 0;
    lastFrameTime = performance.now();
    realtimeInterval = setInterval(analyzeFrame, 1200);
    analyzeFrame();
  } catch (err) {
    cameraIdle.querySelector(".camera-idle__text").textContent =
      "⚠️ Camera access denied";
    cameraIdle.querySelector(".camera-idle__sub").textContent = err.message;
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  clearInterval(realtimeInterval);
  realtimeInterval = null;

  cameraVideo.hidden = true;
  cameraVideo.srcObject = null;
  cameraIdle.hidden = false;
  cameraOverlay.hidden = true;
  cameraLiveResult.hidden = true;

  startCameraBtn.hidden = false;
  stopCameraBtn.hidden = true;
  captureBtn.hidden = true;
  cameraSelect.hidden = true;

  hide(realtimeData);
  show(realtimeIdle);
  rtFpsEl.textContent = "—";
  cameraIdle.querySelector(".camera-idle__text").textContent =
    "Camera not started";
  cameraIdle.querySelector(".camera-idle__sub").textContent =
    "Click the button below to begin";
}

async function switchCamera() {
  if (!stream) return;
  stopCamera();
  await sleep(300);
  startCamera();
}

async function analyzeFrame() {
  if (!stream || cameraVideo.readyState < 2) return;
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  ctx.translate(128, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(cameraVideo, 0, 0, 128, 128);
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  // Always try the Python server. If unavailable, show a clear error.
  let result;
  if (serverAvailable) {
    try {
      const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
      result = await callServerAPI(dataUrl);
    } catch (err) {
      updateRealtimeUI({
        fruitType: "error",
        ripenessStage: "unknown",
        classLabel: "error",
        confidence: 0,
        probs: {},
        features: {},
        source: "error",
        errorMessage: err.message,
      });
      return;
    }
  } else {
    updateRealtimeUI({
      fruitType: "error",
      ripenessStage: "unknown",
      classLabel: "error",
      confidence: 0,
      probs: {},
      features: {},
      source: "error",
      errorMessage:
        "Server tidak tersedia — pastikan Flask berjalan di " + API_BASE,
    });
    return;
  }
  result.features = result.features || {};
  updateRealtimeUI(result);
}

function captureFrame() {
  if (!stream || cameraVideo.readyState < 2) return;
  const canvas = document.createElement("canvas");
  canvas.width = cameraVideo.videoWidth || 640;
  canvas.height = cameraVideo.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(cameraVideo, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
  // Send to photo scanner
  previewImg.src = dataUrl;
  document
    .getElementById("scanner-photo")
    .scrollIntoView({ behavior: "smooth" });
  startPhotoAnalysis(dataUrl);
}

function updateRealtimeUI(result) {
  show(realtimeData);
  hide(realtimeIdle);

  if (result.source === "error") {
    cameraLiveResult.hidden = false;
    liveFruit.textContent = "⚠️ Error";
    liveRipeness.textContent = result.errorMessage || "Server tidak tersedia";
    liveConf.textContent = "";
    rtFruitType.textContent = "⚠️ Server tidak tersedia";
    rtRipenessDot.style.background = "#ef4444";
    rtRipenessStatus.textContent =
      result.errorMessage || "Periksa koneksi Flask";
    rtConfPct.textContent = "—";
    rtConfFill.style.width = "0%";
    rtProbsList.innerHTML = "";
    return;
  }
  liveFruit.textContent = `${FRUIT_EMOJI[result.fruitType] || "?"} ${cap(result.fruitType)}`;
  liveRipeness.textContent = `${RIPENESS_ICON[result.ripenessStage]} ${cap(result.ripenessStage)}`;
  liveConf.textContent = `${Math.round(result.confidence * 100)}% confidence`;

  // Side panel
  rtFruitType.textContent = `${FRUIT_EMOJI[result.fruitType] || "?"} ${cap(result.fruitType)}`;
  rtRipenessDot.style.background = RIPENESS_COLOR[result.ripenessStage];
  rtRipenessStatus.textContent = `${RIPENESS_ICON[result.ripenessStage]} ${cap(result.ripenessStage)}`;

  const pct = Math.round(result.confidence * 100);
  rtConfPct.textContent = pct + "%";
  requestAnimationFrame(() => {
    rtConfFill.style.width = pct + "%";
  });

  const sorted = Object.entries(result.probs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);
  rtProbsList.innerHTML = sorted
    .map(([cls, prob], i) => {
      const p = Math.round(prob * 100),
        [ft, rs] = cls.split("_");
      return `<div class="prob-row"><span class="prob-row__name">${FRUIT_EMOJI[ft] || "?"} ${cap(ft)} ${cap(rs)}</span><div class="prob-row__track"><div class="prob-row__fill ${i === 0 ? "prob-row__fill--top" : ""}" style="width:${p}%"></div></div><span class="prob-row__pct">${p}%</span></div>`;
    })
    .join("");
}

/* ═══════════════════════════════════════════
   9. ACTIVE NAV LINK on scroll
═══════════════════════════════════════════ */
const sections = [
  { id: "home", link: "#home" },
  { id: "about", link: "#about" },
  { id: "technology", link: "#technology" },
  { id: "features", link: "#features" },
  { id: "scanner-photo", link: "#scanner-photo" },
  { id: "scanner-realtime", link: "#scanner-realtime" },
];
const navLinks = document.querySelectorAll(".navbar__links a");
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        navLinks.forEach((a) => a.classList.remove("nav-link--active"));
        const active = document.querySelector(
          `.navbar__links a[href="#${e.target.id}"]`,
        );
        if (active) active.classList.add("nav-link--active");
      }
    });
  },
  { threshold: 0.3 },
);
sections.forEach((s) => {
  const el = document.getElementById(s.id);
  if (el) observer.observe(el);
});

/* ═══════════════════════════════════════════
   UTILS
═══════════════════════════════════════════ */
const show = (el) => {
  el.hidden = false;
};
const hide = (el) => {
  el.hidden = true;
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const fmtN = (v, d) =>
  v !== undefined && v !== null ? Number(v).toFixed(d) : "—";
