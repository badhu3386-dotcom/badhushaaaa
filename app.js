import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const STORAGE_KEY = "sign_converter_signs_v1";
const STABLE_FRAMES = 6;     // frames a sign must stay the same before it is shown
const HOLD_FRAMES = 20;      // frames a sign must be held before it is added to the sentence
const LOST_FRAMES = 10;      // frames without a sign before the same word can be added again
const K = 5;                 // neighbours used for matching

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");

let handLandmarker = null;
let running = false;
let lastVideoTime = -1;

// ---------- state ----------
let signs = loadSigns(); // [{label, vec:[...]}]
let sentence = [];
let candidate = null;
let streak = 0;
let lostCount = 0;
let lastAdded = null;
let recording = null; // {label, phase, until, samples, lastCapture}

// ---------- storage ----------
function loadSigns() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}
function saveSigns() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(signs));
  } catch (e) {
    alert("Could not save signs (browser storage full or blocked). Use Export to keep a backup.");
  }
}

// ---------- geometry helpers ----------
const d3 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

// Convert normalized landmarks to pixel-like coordinates (keeps real proportions)
function toPixels(lm) {
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  return lm.map((p) => ({ x: p.x * w, y: p.y * h, z: p.z * w }));
}

function handScale(p) {
  return d3(p[0], p[9]) || 1;
}

// [thumb, index, middle, ring, pinky] -> true when extended
function fingerStates(p) {
  const hs = handScale(p);
  const thumb = d3(p[4], p[5]) > hs * 0.75;
  const ext = (tip, pip) => d3(p[tip], p[0]) > d3(p[pip], p[0]) * 1.05;
  return [thumb, ext(8, 6), ext(12, 10), ext(16, 14), ext(20, 18)];
}

// Build a feature vector from 1 or 2 hands (sorted left-to-right on screen)
function buildFeature(hands) {
  const vec = new Array(63 * 2 + 3).fill(0);
  hands.forEach((h, slot) => {
    const p = h.p;
    const s = handScale(p);
    const w = p[0];
    let i = slot * 63;
    for (const q of p) {
      vec[i++] = (q.x - w.x) / s;
      vec[i++] = (q.y - w.y) / s;
      vec[i++] = (q.z - w.z) / s;
    }
  });
  if (hands.length === 2) {
    const a = hands[0].p[0];
    const b = hands[1].p[0];
    const s = handScale(hands[0].p);
    vec[126] = (b.x - a.x) / s;
    vec[127] = (b.y - a.y) / s;
    vec[128] = (b.z - a.z) / s;
  }
  return vec;
}

function dist(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

// ---------- recognition ----------
const SINGLE = {
  "01000": "One",
  "01100": "Two",
  "01110": "Three",
  "01111": "Four",
  "11111": "Five / Hello",
  "00000": "Yes",
  "10000": "Good",
  "10001": "Call me",
  "11001": "I love you",
};

function builtinSign(hands) {
  const keys = hands.map((h) => h.fingers.map((f) => (f ? "1" : "0")).join(""));
  if (hands.length === 1) return SINGLE[keys[0]] || null;
  if (hands.length === 2) {
    const open = (k) => k === "11111" || k === "01111";
    if (open(keys[0]) && open(keys[1])) {
      const avgScale = (handScale(hands[0].p) + handScale(hands[1].p)) / 2;
      const gap = d3(hands[0].p[9], hands[1].p[9]);
      if (gap < avgScale * 1.0) return "Namaste";
      if (gap > avgScale * 2.5) return "Welcome";
      return null;
    }
    if (keys[0] === "10000" && keys[1] === "10000") return "Very good";
    if (keys[0] === "00000" && keys[1] === "00000") return "Stop";
  }
  return null;
}

function customSign(vec) {
  if (!signs.length) return null;
  const threshold = parseFloat($("threshold").value);
  const scored = signs
    .map((s) => ({ label: s.label, d: dist(vec, s.vec) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, K)
    .filter((n) => n.d < threshold);
  if (!scored.length) return null;
  const votes = {};
  for (const n of scored) votes[n.label] = (votes[n.label] || 0) + 1 / (n.d + 0.05);
  return Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
}

function recognise(hands, vec) {
  const custom = customSign(vec);
  if (custom) return { label: custom, source: "your sign" };
  if ($("builtinToggle").checked) {
    const b = builtinSign(hands);
    if (b) return { label: b, source: "built-in sign" };
  }
  return null;
}

// ---------- speech ----------
function speak(text) {
  if (!("speechSynthesis" in window) || !text) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = $("lang").value;
  window.speechSynthesis.speak(u);
}

// ---------- UI ----------
function renderSentence() {
  $("sentence").textContent = sentence.join(" ");
}
function renderSigns() {
  const box = $("signList");
  box.innerHTML = "";
  const counts = {};
  signs.forEach((s) => (counts[s.label] = (counts[s.label] || 0) + 1));
  const labels = Object.keys(counts);
  if (!labels.length) {
    box.innerHTML = '<div class="empty">No custom signs yet.</div>';
    return;
  }
  labels.forEach((label) => {
    const row = document.createElement("div");
    row.className = "sign-item";
    const name = document.createElement("span");
    name.textContent = label;
    const small = document.createElement("small");
    small.textContent = `${counts[label]} samples`;
    name.appendChild(small);
    const del = document.createElement("button");
    del.textContent = "Delete";
    del.onclick = () => {
      signs = signs.filter((s) => s.label !== label);
      saveSigns();
      renderSigns();
    };
    row.append(name, del);
    box.appendChild(row);
  });
}
function setCurrent(label, source) {
  $("current").textContent = label || "—";
  $("source").textContent = label ? source : "";
}

// ---------- recording new signs ----------
function startRecording() {
  const label = $("signName").value.trim();
  if (!running) return alert("Start the camera first.");
  if (!label) return alert("Type the word for this sign first.");
  recording = { label, phase: "countdown", until: performance.now() + 3000, samples: [], lastCapture: 0 };
  $("recordBtn").disabled = true;
}
function updateRecording(vec, now) {
  if (!recording) return;
  if (recording.phase === "countdown") {
    const left = Math.ceil((recording.until - now) / 1000);
    if (left > 0) {
      $("recStatus").textContent = `Get ready… ${left}`;
      return;
    }
    recording.phase = "capture";
    recording.until = now + 3000;
  }
  if (recording.phase === "capture") {
    const left = Math.ceil((recording.until - now) / 1000);
    $("recStatus").textContent = vec
      ? `Recording “${recording.label}”… ${left} (${recording.samples.length} samples)`
      : `Show your hand(s)! ${left}`;
    if (vec && now - recording.lastCapture > 80) {
      recording.samples.push(vec);
      recording.lastCapture = now;
    }
    if (now >= recording.until) finishRecording();
  }
}
function finishRecording() {
  const { label, samples } = recording;
  if (samples.length < 5) {
    $("recStatus").textContent = "No hands were seen. Try again.";
  } else {
    samples.forEach((vec) => signs.push({ label, vec: vec.map((v) => Math.round(v * 1000) / 1000) }));
    saveSigns();
    renderSigns();
    $("recStatus").textContent = `Saved ${samples.length} samples for “${label}” ✔`;
  }
  recording = null;
  $("recordBtn").disabled = false;
}

// ---------- main loop ----------
function drawHands(result) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const du = new DrawingUtils(ctx);
  for (const lm of result.landmarks || []) {
    du.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: "#35d07f", lineWidth: 3 });
    du.drawLandmarks(lm, { color: "#ff5d6c", radius: 3 });
  }
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const now = performance.now();
    const result = handLandmarker.detectForVideo(video, now);
    drawHands(result);

    // gather and sort hands left-to-right on screen
    let hands = (result.landmarks || []).map((lm) => {
      const p = toPixels(lm);
      return { p, fingers: fingerStates(p) };
    });
    hands.sort((a, b) => a.p[0].x - b.p[0].x);
    hands = hands.slice(0, 2);

    const pill = $("handStatus");
    pill.textContent = `Hands: ${hands.length}`;
    pill.classList.toggle("ok", hands.length === 2);

    const vec = hands.length ? buildFeature(hands) : null;
    updateRecording(vec, now);

    const found = recording ? null : vec ? recognise(hands, vec) : null;
    handleDetection(found);
  }
  requestAnimationFrame(loop);
}

function handleDetection(found) {
  const label = found ? found.label : null;
  if (label === candidate) streak++;
  else {
    candidate = label;
    streak = 1;
  }
  if (!label) {
    lostCount++;
    if (lostCount >= LOST_FRAMES) lastAdded = null;
    if (streak >= STABLE_FRAMES) setCurrent(null);
    return;
  }
  lostCount = 0;
  if (streak >= STABLE_FRAMES) setCurrent(label, found.source);
  if (streak >= HOLD_FRAMES && label !== lastAdded) {
    lastAdded = label;
    sentence.push(label);
    renderSentence();
    if ($("autoSpeak").checked) speak(label);
  }
}

// ---------- start camera ----------
async function start() {
  const btn = $("startBtn");
  btn.disabled = true;
  $("loading").classList.remove("hidden");
  $("loading").textContent = "Loading hand tracking model…";
  try {
    if (!handLandmarker) {
      const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
      const make = (delegate) =>
        HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      try {
        handLandmarker = await make("GPU");
      } catch {
        handLandmarker = await make("CPU");
      }
    }
    $("loading").textContent = "Starting camera…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    $("loading").classList.add("hidden");
    running = true;
    btn.textContent = "Stop camera";
    btn.disabled = false;
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    $("loading").textContent =
      "Could not start: " + (err.message || err) + ". Allow camera access and open this page over https or localhost.";
    btn.disabled = false;
  }
}
function stop() {
  running = false;
  const s = video.srcObject;
  if (s) s.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("startBtn").textContent = "Start camera";
  $("loading").textContent = "Camera stopped";
  $("loading").classList.remove("hidden");
  setCurrent(null);
}

// ---------- events ----------
$("startBtn").onclick = () => (running ? stop() : start());
$("recordBtn").onclick = startRecording;
$("speakBtn").onclick = () => speak(sentence.join(" "));
$("undoBtn").onclick = () => {
  sentence.pop();
  renderSentence();
};
$("clearBtn").onclick = () => {
  sentence = [];
  renderSentence();
};
$("exportBtn").onclick = () => {
  const blob = new Blob([JSON.stringify(signs)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "my-signs.json";
  a.click();
  URL.revokeObjectURL(a.href);
};
$("importBtn").onclick = () => $("importFile").click();
$("importFile").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    if (!Array.isArray(data) || data.some((s) => !s.label || !Array.isArray(s.vec))) throw new Error("bad file");
    signs = signs.concat(data);
    saveSigns();
    renderSigns();
  } catch {
    alert("That file is not a valid signs export.");
  }
  e.target.value = "";
};
$("resetBtn").onclick = () => {
  if (confirm("Delete all your custom signs?")) {
    signs = [];
    saveSigns();
    renderSigns();
  }
};

renderSigns();
renderSentence();
