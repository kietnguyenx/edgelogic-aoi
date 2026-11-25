/**
 * EdgeLogic-AOI — Production Workbench Client
 * Connects directly to FastAPI (/api/models, /api/inspect, /api/conveyor/next, /api/benchmarks)
 * Supports: Real MVTec Conveyor Stream, Static Image Upload, Live Video Frame Stream, and Live Webcam.
 */

const state = {
  mode: "conveyor",
  showJson: false,
  inflight: false,
  lineTimer: null,
  streamTimer: null,
  renderRafId: null,
  ws: null,
  wsSourceName: "",
  lastLogMs: 0,
  mediaStream: null,
  lastImageFile: null,
  lastVideoFile: null,
  videoPlaying: false,
  cameraActive: false,
  lastPayload: null,
  srcImgObj: null,
  heatImgObj: null,
  counts: { Pass: 0, Review: 0, Reject: 0 },
  logCount: 0,
};

const els = {
  modelSelect: document.getElementById("modelSelect"),
  runLineBtn: document.getElementById("runLineBtn"),
  sourceActionSlot: document.getElementById("sourceActionSlot"),
  blurFaultToggle: document.getElementById("blurFaultToggle"),
  imageInput: document.getElementById("imageInput"),
  videoInput: document.getElementById("videoInput"),
  rawCanvas: document.getElementById("rawCanvas"),
  heatCanvas: document.getElementById("heatCanvas"),
  captureCanvas: document.getElementById("captureCanvas"),
  liveVideo: document.getElementById("liveVideo"),
  opacitySlider: document.getElementById("opacitySlider"),
  opacityReadout: document.getElementById("opacityReadout"),
  bboxToggle: document.getElementById("bboxToggle"),
  diagTable: document.getElementById("diagTable"),
  diagJson: document.getElementById("diagJson"),
  jsonToggleBtn: document.getElementById("jsonToggleBtn"),
  logBody: document.getElementById("logBody"),
};

function decisionColorVar(decision) {
  if (decision === "Pass") return "var(--success)";
  if (decision === "Review") return "var(--warning)";
  return "var(--error)";
}

function renderViewport() {
  const rawCtx = els.rawCanvas.getContext("2d");
  const heatCtx = els.heatCanvas.getContext("2d");
  const w = els.rawCanvas.width;
  const h = els.rawCanvas.height;

  const alpha = Number(els.opacitySlider.value) / 100;
  els.opacityReadout.textContent = `Alpha ${alpha.toFixed(2)}`;

  const isLiveStream = state.mode === "video" || state.mode === "camera";
  if (state.srcImgObj && !isLiveStream) {
    rawCtx.clearRect(0, 0, w, h);
    rawCtx.drawImage(state.srcImgObj, 0, 0, w, h);
  }

  heatCtx.clearRect(0, 0, w, h);
  if (isLiveStream && els.liveVideo.readyState >= 2) {
    heatCtx.drawImage(els.liveVideo, 0, 0, w, h);
  } else if (state.srcImgObj) {
    heatCtx.drawImage(state.srcImgObj, 0, 0, w, h);
  }

  const p = state.lastPayload;
  if (!p) return;

  if (p.decision === "Drift Halt") {
    heatCtx.fillStyle = "rgba(239, 68, 68, 0.24)";
    heatCtx.fillRect(0, 0, w, h);
    heatCtx.fillStyle = "#ffffff";
    heatCtx.font = "600 13px Inter, sans-serif";
    heatCtx.textAlign = "center";
    heatCtx.fillText(`Interlock: ${p.drift_gate.flags.join(", ") || "Optical Drift"}`, w / 2, h / 2);
    return;
  }

  if (state.heatImgObj && alpha > 0) {
    heatCtx.save();
    heatCtx.globalAlpha = alpha * 0.72;
    heatCtx.drawImage(state.heatImgObj, 0, 0, w, h);
    heatCtx.restore();
  }

  const bbox = p.diagnostics && p.diagnostics.hotspot_bbox_norm;
  if (bbox && els.bboxToggle.checked && p.decision !== "Pass") {
    const [x1, y1, x2, y2] = bbox;
    const bx = x1 * w, by = y1 * h, bw = (x2 - x1) * w, bh = (y2 - y1) * h;
    const cx = bx + bw / 2, cy = by + bh / 2;
    const color = p.decision === "Reject" ? "#ef4444" : "#f59e0b";

    heatCtx.save();
    heatCtx.strokeStyle = color;
    heatCtx.lineWidth = 1.75;
    heatCtx.strokeRect(bx, by, bw, bh);
    heatCtx.beginPath();
    heatCtx.moveTo(cx - 6, cy); heatCtx.lineTo(cx + 6, cy);
    heatCtx.moveTo(cx, cy - 6); heatCtx.lineTo(cx, cy + 6);
    heatCtx.stroke();
    heatCtx.fillStyle = color;
    heatCtx.font = "600 11px JetBrains Mono, monospace";
    heatCtx.fillText(`ROI ${p.diagnostics.area_pct}%`, bx, Math.max(13, by - 4));
    heatCtx.restore();
  }
}

function updateUI(payload) {
  state.lastPayload = payload;

  // Header metadata
  document.getElementById("metaPartId").textContent = payload.part_id;
  document.getElementById("metaLatency").textContent = `${payload.latency_ms.inference.toFixed(2)} ms`;
  document.getElementById("metaFps").textContent = `${payload.fps_instant.toFixed(1)} FPS`;
  document.getElementById("srcTag").textContent = payload.source;

  // Stage 1: Drift Gate
  const dg = payload.drift_gate;
  document.getElementById("stage1Dot").style.background = dg.passed ? "var(--success)" : "var(--error)";
  document.getElementById("stage1Text").textContent = dg.passed
    ? `Pass · ${dg.latency_ms.toFixed(2)} ms`
    : `${dg.flags[0] || "Drift"} · Halt`;
  document.getElementById("mSharp").textContent = Number(dg.metrics.laplacian_var || 0).toFixed(2);
  document.getElementById("mBright").textContent = Number(dg.metrics.brightness_mean || 0).toFixed(2);
  document.getElementById("mContrast").textContent = Number(dg.metrics.contrast_std || 0).toFixed(2);
  document.getElementById("mHist").textContent = Number(dg.metrics.bhattacharyya_dist || 0).toFixed(2);

  // Stage 2-3: Conformal Triage & Needle
  const tPass = payload.thresholds.t_pass;
  const tRej = payload.thresholds.t_reject;
  const col = decisionColorVar(payload.decision);

  document.getElementById("triageDot").style.background = col;
  document.getElementById("triageText").textContent = payload.decision;
  document.getElementById("thresholdText").textContent = `Pass ${tPass.toFixed(3)} · Reject ${tRej.toFixed(3)}`;

  if (payload.anomaly_score !== null) {
    document.getElementById("scoreDisplay").textContent = payload.anomaly_score.toFixed(3);
    let pct = 50;
    if (payload.anomaly_score < tPass) {
      pct = Math.max(2, Math.min(54, (payload.anomaly_score / Math.max(tPass, 1e-6)) * 54));
    } else if (payload.anomaly_score <= tRej) {
      const span = Math.max(tRej - tPass, 1e-6);
      pct = 55 + ((payload.anomaly_score - tPass) / span) * 15;
    } else {
      const extra = (payload.anomaly_score - tRej) / Math.max(tRej * 0.4, 1e-6);
      pct = Math.min(97, 70 + extra * 27);
    }
    document.getElementById("scoreNeedle").style.left = `${pct.toFixed(1)}%`;
  } else {
    document.getElementById("scoreDisplay").textContent = "Bypassed";
    document.getElementById("scoreNeedle").style.left = "2%";
  }

  // PLC Register & Counters
  const plcEl = document.getElementById("plcHex");
  plcEl.textContent = `${payload.plc.hex_code} ${payload.plc.label}`;
  plcEl.style.color = col;

  if (payload.decision === "Pass") state.counts.Pass++;
  else if (payload.decision === "Review") state.counts.Review++;
  else state.counts.Reject++;

  state.logCount++;
  document.getElementById("cntPass").textContent = state.counts.Pass;
  document.getElementById("cntRev").textContent = state.counts.Review;
  document.getElementById("cntRej").textContent = state.counts.Reject;
  document.getElementById("tabLogCount").textContent = state.logCount;

  // Stage 4: Root-Cause Diagnostics (Key-Value + JSON)
  const diag = payload.diagnostics || {};
  document.getElementById("copilotMode").textContent = payload.decision === "Pass" ? "Standby" : "Active";
  const rows = [];
  if (payload.decision === "Pass") {
    rows.push(["Mode", "Fast-Path Bypass"]);
    rows.push(["Copilot Latency", "0.00 ms"]);
    rows.push(["Routing", "Good Line"]);
    if (payload.ground_truth) rows.push(["Sample Class", payload.ground_truth]);
  } else if (payload.decision === "Drift Halt") {
    rows.push(["Interlock", "Optical Drift Halt"]);
    rows.push(["Fault Code", diag.fault || "Optical Drift"]);
    rows.push(["AI Inference", "Skipped"]);
    rows.push(["PLC Action", diag.plc_action || "Trigger Andon"]);
  } else {
    rows.push(["Defect Scale", diag.defect_scale || "—"]);
    rows.push(["Estimated Area", `${diag.area_pct ?? 0}%`]);
    rows.push(["Hotspot Region", diag.hotspot_region || "—"]);
    rows.push(["Probable Cause", diag.probable_cause || "—"]);
    rows.push(["PLC Action", diag.plc_action || "—"]);
  }
  els.diagTable.innerHTML = rows
    .map(([k, v]) => `<div class="diag-row"><span class="diag-key">${k}</span><span class="diag-val">${v}</span></div>`)
    .join("");
  els.diagJson.textContent = JSON.stringify({
    part_id: payload.part_id,
    decision: payload.decision,
    anomaly_score: payload.anomaly_score,
    plc: payload.plc,
    diagnostics: diag,
    latency_ms: payload.latency_ms,
  }, null, 2);

  // Append to MES Log (throttled during high-FPS live stream so DOM never drops frames)
  const isLiveStream = state.mode === "video" || state.mode === "camera";
  const nowMs = performance.now();
  const shouldLog = !isLiveStream || (nowMs - state.lastLogMs >= 250) || payload.decision === "Reject" || payload.decision === "Drift Halt";
  if (shouldLog) {
    state.lastLogMs = nowMs;
    const tr = document.createElement("tr");
    const now = new Date().toLocaleTimeString("en-GB", { hour12: false });
    tr.innerHTML = `
      <td class="mono muted-label">${now}</td>
      <td class="mono">${payload.part_id}</td>
      <td>${payload.source}</td>
      <td style="color:${dg.passed ? "var(--success)" : "var(--error)"}">${dg.passed ? "Pass" : dg.flags.join(", ")}</td>
      <td class="mono">${payload.anomaly_score !== null ? payload.anomaly_score.toFixed(3) : "—"}</td>
      <td><span style="display:inline-flex;align-items:center;gap:6px;font-weight:500;"><span class="dot" style="background:${col};"></span>${payload.decision}</span></td>
      <td class="mono">${payload.plc.hex_code}</td>
      <td class="mono text-right">${payload.latency_ms.total.toFixed(2)} ms</td>
    `;
    els.logBody.prepend(tr);
    while (els.logBody.children.length > 200) els.logBody.removeChild(els.logBody.lastChild);
  }

  // Load heatmap (and source image only when provided — skipped in high-FPS video/camera stream)
  if (!payload.source_image_b64) {
    if (payload.heatmap_image_b64) {
      const hmImg = new Image();
      hmImg.onload = () => {
        state.heatImgObj = hmImg;
        if (!state.renderRafId) renderViewport();
      };
      hmImg.src = payload.heatmap_image_b64;
    } else {
      state.heatImgObj = null;
      if (!state.renderRafId) renderViewport();
    }
    return;
  }

  const srcImg = new Image();
  srcImg.onload = () => {
    state.srcImgObj = srcImg;
    if (payload.heatmap_image_b64) {
      const hmImg = new Image();
      hmImg.onload = () => {
        state.heatImgObj = hmImg;
        renderViewport();
      };
      hmImg.src = payload.heatmap_image_b64;
    } else {
      state.heatImgObj = null;
      renderViewport();
    }
  };
  srcImg.src = payload.source_image_b64;
}

async function inspectConveyorNext() {
  if (state.inflight) return;
  state.inflight = true;
  try {
    const fd = new FormData();
    fd.append("model_id", els.modelSelect.value);
    fd.append("simulate_blur", String(els.blurFaultToggle.checked));
    const res = await fetch("/api/conveyor/next", { method: "POST", body: fd });
    if (res.ok) updateUI(await res.json());
  } finally {
    state.inflight = false;
  }
}

async function inspectBlob(blob, sourceName) {
  if (state.inflight) return;
  state.inflight = true;
  try {
    const fd = new FormData();
    fd.append("file", blob, "frame.jpg");
    fd.append("model_id", els.modelSelect.value);
    fd.append("source", sourceName);
    fd.append("simulate_blur", String(els.blurFaultToggle.checked));
    const res = await fetch("/api/inspect", { method: "POST", body: fd });
    if (res.ok) updateUI(await res.json());
  } finally {
    state.inflight = false;
  }
}

function startRenderLoop() {
  if (state.renderRafId) return;
  const tick = () => {
    if (state.mode !== "video" && state.mode !== "camera") {
      state.renderRafId = null;
      return;
    }
    if (!els.liveVideo.paused && els.liveVideo.readyState >= 2) {
      renderViewport();
    }
    state.renderRafId = requestAnimationFrame(tick);
  };
  state.renderRafId = requestAnimationFrame(tick);
}

function stopRenderLoop() {
  if (state.renderRafId) {
    cancelAnimationFrame(state.renderRafId);
    state.renderRafId = null;
  }
}

function sendNextStreamFrame() {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  if (els.liveVideo.paused || els.liveVideo.readyState < 2 || state.inflight) {
    if (state.videoPlaying || state.cameraActive) {
      setTimeout(sendNextStreamFrame, 16);
    }
    return;
  }
  state.inflight = true;
  const ctx = els.captureCanvas.getContext("2d");
  ctx.drawImage(els.liveVideo, 0, 0, 256, 256);
  els.captureCanvas.toBlob(blob => {
    if (blob && state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(blob);
    } else {
      state.inflight = false;
    }
  }, "image/jpeg", 0.72);
}

function closeStreamSocket() {
  if (state.ws) {
    try { state.ws.close(); } catch (_) {}
    state.ws = null;
  }
  state.inflight = false;
}

function startStreamSocket(sourceName) {
  closeStreamSocket();
  state.wsSourceName = sourceName;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const qs = new URLSearchParams({
    model_id: els.modelSelect.value,
    source: sourceName,
    simulate_blur: String(els.blurFaultToggle.checked),
  });
  const ws = new WebSocket(`${proto}://${location.host}/ws/inspect?${qs.toString()}`);
  state.ws = ws;

  ws.onopen = () => {
    startRenderLoop();
    sendNextStreamFrame();
  };

  ws.onmessage = ev => {
    state.inflight = false;
    try {
      updateUI(JSON.parse(ev.data));
    } catch (_) {}
    if (state.videoPlaying || state.cameraActive) {
      sendNextStreamFrame();
    }
  };

  ws.onerror = () => {
    state.inflight = false;
  };

  ws.onclose = () => {
    state.inflight = false;
  };
}

function pushStreamConfig() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({
      model_id: els.modelSelect.value,
      source: state.wsSourceName || "Live Stream",
      simulate_blur: els.blurFaultToggle.checked,
    }));
  }
}

function syncPrimaryActionButton() {
  if (state.mode === "conveyor") {
    els.runLineBtn.textContent = state.lineTimer ? "Stop Line" : "Run Line";
  } else if (state.mode === "image") {
    els.runLineBtn.textContent = "Inspect Image";
  } else if (state.mode === "video") {
    els.runLineBtn.textContent = state.videoPlaying ? "Pause Video" : "Play & Inspect";
  } else if (state.mode === "camera") {
    els.runLineBtn.textContent = state.cameraActive ? "Stop Camera" : "Start Camera";
  }
}

function stopConveyorLine() {
  if (state.lineTimer) {
    clearInterval(state.lineTimer);
    state.lineTimer = null;
  }
}

function stopMediaStreams() {
  stopRenderLoop();
  closeStreamSocket();
  if (state.streamTimer) { clearInterval(state.streamTimer); state.streamTimer = null; }
  if (state.mediaStream) { state.mediaStream.getTracks().forEach(t => t.stop()); state.mediaStream = null; }
  state.videoPlaying = false;
  state.cameraActive = false;
  els.liveVideo.pause();
  els.liveVideo.srcObject = null;
  els.liveVideo.classList.add("hidden");
  els.rawCanvas.classList.remove("hidden");
  const camHint = document.getElementById("camHint");
  if (camHint) camHint.textContent = "Webcam idle";
}

function switchMode(mode) {
  stopMediaStreams();
  stopConveyorLine();
  state.mode = mode;

  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });

  if (mode === "conveyor") {
    els.sourceActionSlot.innerHTML = `<button id="stepBtn" type="button" class="btn">Next Frame</button>`;
    document.getElementById("stepBtn").addEventListener("click", inspectConveyorNext);
    inspectConveyorNext();
  } else if (mode === "image") {
    els.sourceActionSlot.innerHTML = `<button id="pickImgBtn" type="button" class="btn">Upload Image</button>`;
    document.getElementById("pickImgBtn").addEventListener("click", () => els.imageInput.click());
  } else if (mode === "video") {
    els.sourceActionSlot.innerHTML = `<button id="pickVidBtn" type="button" class="btn">Upload Video</button>`;
    document.getElementById("pickVidBtn").addEventListener("click", () => els.videoInput.click());
  } else if (mode === "camera") {
    els.sourceActionSlot.innerHTML = `<span id="camHint" class="muted-label">Webcam idle</span>`;
  }

  syncPrimaryActionButton();
}

async function startWebcam() {
  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60, max: 120 } },
    });
    els.liveVideo.srcObject = state.mediaStream;
    await els.liveVideo.play();
    els.rawCanvas.classList.add("hidden");
    els.liveVideo.classList.remove("hidden");
    state.cameraActive = true;
    const camHint = document.getElementById("camHint");
    if (camHint) camHint.textContent = "Live feed active";
    startStreamSocket("Live Camera");
    syncPrimaryActionButton();
  } catch (err) {
    state.cameraActive = false;
    syncPrimaryActionButton();
    els.diagTable.innerHTML = `<div class="diag-row"><span class="diag-key">Camera Error</span><span class="diag-val">${err.message}</span></div>`;
  }
}

function handlePrimaryAction() {
  if (state.mode === "conveyor") {
    if (state.lineTimer) {
      stopConveyorLine();
    } else {
      inspectConveyorNext();
      state.lineTimer = setInterval(inspectConveyorNext, 900);
    }
    syncPrimaryActionButton();
    return;
  }

  if (state.mode === "image") {
    if (state.lastImageFile) {
      inspectBlob(state.lastImageFile, `Image · ${state.lastImageFile.name}`);
    } else {
      els.imageInput.click();
    }
    return;
  }

  if (state.mode === "video") {
    if (!state.lastVideoFile) {
      els.videoInput.click();
      return;
    }
    if (state.videoPlaying) {
      els.liveVideo.pause();
      stopRenderLoop();
      closeStreamSocket();
      state.videoPlaying = false;
    } else {
      els.rawCanvas.classList.add("hidden");
      els.liveVideo.classList.remove("hidden");
      els.liveVideo.play();
      state.videoPlaying = true;
      startStreamSocket(`Video · ${state.lastVideoFile.name}`);
    }
    syncPrimaryActionButton();
    return;
  }

  if (state.mode === "camera") {
    if (state.cameraActive) {
      stopMediaStreams();
      syncPrimaryActionButton();
    } else {
      startWebcam();
    }
  }
}

async function loadBenchmarks() {
  const res = await fetch("/api/benchmarks");
  if (!res.ok) return;
  const data = await res.json();

  const qGrid = document.getElementById("quantBenchGrid");
  qGrid.innerHTML = Object.entries(data.quantization || {}).map(([key, variants]) => {
    const title = key.replaceAll("_", " ").toUpperCase();
    const rows = Object.entries(variants).map(([vName, m]) => {
      const isHighlight = vName.includes("NNCF") || vName.includes("QAMR");
      const cls = isHighlight ? "fw-600 text-success" : "";
      const microRec = m.micro_recall_at_f1 !== null && m.micro_recall_at_f1 !== undefined
        ? `${(m.micro_recall_at_f1 * 100).toFixed(2)}%`
        : "—";
      const fps = m.fps_async ?? m.fps ?? "—";
      return `
        <tr class="${cls}">
          <td>${vName}</td>
          <td class="mono">${Number(m.auroc).toFixed(4)}</td>
          <td class="mono">${microRec}</td>
          <td class="mono">${Number(m.p50_ms).toFixed(2)} ms</td>
          <td class="mono">${fps}</td>
          <td class="mono text-right">${Number(m.size_mb).toFixed(2)} MB</td>
        </tr>
      `;
    }).join("");

    return `
      <div class="panel-box">
        <div class="panel-box-title">${title}</div>
        <table class="data-table">
          <thead>
            <tr><th>Variant</th><th>AUROC</th><th>Micro Rec</th><th>P50</th><th>Async FPS</th><th class="text-right">Size</th></tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;
  }).join("");

  const eBody = document.getElementById("evalBenchBody");
  eBody.innerHTML = (data.evaluations || []).map(e => `
    <tr>
      <td>${e.dataset}</td>
      <td class="mono">${e.category}</td>
      <td>${e.model}</td>
      <td class="mono fw-600 text-success">${Number(e.image_auroc).toFixed(4)}</td>
      <td class="mono">${Number(e.optimal_f1).toFixed(4)}</td>
      <td class="mono text-right">${e.overkill_fpr !== null ? `${(e.overkill_fpr * 100).toFixed(2)}%` : "—"}</td>
    </tr>
  `).join("");
}

async function init() {
  // Tabs
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      document.querySelectorAll(".tab-view").forEach(v => v.classList.add("hidden"));
      document.getElementById(`view-${target}`).classList.remove("hidden");
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === target));
    });
  });

  // Load models catalog
  const res = await fetch("/api/models");
  if (res.ok) {
    const { models } = await res.json();
    els.modelSelect.innerHTML = models.map(m => `<option value="${m.id}">${m.label}</option>`).join("");
  }

  // Wire controls
  document.querySelectorAll(".mode-btn").forEach(btn => btn.addEventListener("click", () => switchMode(btn.dataset.mode)));
  document.getElementById("stepBtn").addEventListener("click", inspectConveyorNext);
  els.runLineBtn.addEventListener("click", handlePrimaryAction);
  els.opacitySlider.addEventListener("input", renderViewport);
  els.bboxToggle.addEventListener("change", renderViewport);

  const reinspectCurrent = () => {
    if (state.mode === "conveyor") {
      inspectConveyorNext();
    } else if (state.mode === "image" && state.lastImageFile) {
      inspectBlob(state.lastImageFile, `Image · ${state.lastImageFile.name}`);
    } else if (state.mode === "video" || state.mode === "camera") {
      pushStreamConfig();
    }
  };
  els.modelSelect.addEventListener("change", reinspectCurrent);
  els.blurFaultToggle.addEventListener("change", reinspectCurrent);

  els.jsonToggleBtn.addEventListener("click", () => {
    state.showJson = !state.showJson;
    els.diagTable.classList.toggle("hidden", state.showJson);
    els.diagJson.classList.toggle("hidden", !state.showJson);
    els.jsonToggleBtn.style.borderColor = state.showJson ? "var(--primary)" : "var(--border)";
  });

  document.getElementById("clearLogBtn").addEventListener("click", () => {
    els.logBody.innerHTML = "";
    state.logCount = 0;
    document.getElementById("tabLogCount").textContent = "0";
  });

  els.imageInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) {
      state.lastImageFile = file;
      inspectBlob(file, `Image · ${file.name}`);
    }
  });

  els.videoInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (!file) return;
    stopMediaStreams();
    state.lastVideoFile = file;
    els.liveVideo.src = URL.createObjectURL(file);
    els.liveVideo.play();
    els.rawCanvas.classList.add("hidden");
    els.liveVideo.classList.remove("hidden");
    state.videoPlaying = true;
    startStreamSocket(`Video · ${file.name}`);
    syncPrimaryActionButton();
  });

  syncPrimaryActionButton();
  await loadBenchmarks();
  await inspectConveyorNext();
}

init();
