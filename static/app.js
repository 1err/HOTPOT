"use strict";

(() => {
  const DEFAULT_INGREDIENTS = [
    { id: "beef_rolls", name: "肥牛卷", en: "Beef rolls", duration_seconds: null },
    { id: "crab_sticks", name: "蟹棒", en: "Crab sticks", duration_seconds: null },
    { id: "napa_cabbage", name: "大白菜", en: "Napa cabbage", duration_seconds: null },
    { id: "lotus_root", name: "藕片", en: "Lotus root", duration_seconds: null },
    { id: "soft_tofu", name: "嫩豆腐", en: "Soft tofu", duration_seconds: null },
    { id: "wide_noodles", name: "宽粉", en: "Wide noodles", duration_seconds: null },
    { id: "udon_noodles", name: "乌冬面", en: "Udon noodles", duration_seconds: null },
    { id: "shiitake_mushrooms", name: "香菇", en: "Shiitake mushrooms", duration_seconds: null },
    { id: "beef_meatballs", name: "牛肉丸", en: "Beef meatballs", duration_seconds: null },
  ];
  const STORAGE_KEY = "hotpot-lab-v1";
  const $ = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    "global-message", "camera-video", "uploaded-image", "viewfinder", "camera-empty",
    "crop-guide", "source-badge", "frame-label", "inference-label", "empty-start-camera",
    "frame-result", "frame-result-name", "frame-result-state", "frame-result-score",
    "camera-select", "start-camera", "stop-camera", "image-upload", "auto-recognize",
    "recognize-once", "prediction-name", "prediction-detail", "confidence-badge", "confidence-note",
    "score-bars", "model-status-dot", "model-label", "model-message", "reload-model",
    "ingredient-grid", "selected-name", "duration-input", "duration-save-status",
    "start-timer", "timer-help", "timer-count", "active-summary", "timer-list",
    "sound-notifications", "voice-notifications", "test-alert", "sample-label",
    "session-id", "capture-sample", "sample-message", "dataset-counts", "capture-canvas",
    "ingredient-count", "recognition-model", "recognition-caption", "model-usage",
    "prototype-tag", "source-help", "live-recognition-label", "recognize-button-label",
  ].map((id) => [id.replace(/-([a-z])/g, (_, char) => char.toUpperCase()), $(id)]));

  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch { /* A fresh state is safe. */ }
  const state = {
    ingredients: DEFAULT_INGREDIENTS.map((item) => ({ ...item })),
    selected: null,
    model: { state: "loading", mode: "unavailable", label: "Checking recognition model…", message: "" },
    source: null,
    stream: null,
    generation: 0,
    recognitionBusy: false,
    recognitionAbort: null,
    recognitionTimeout: null,
    recognitionStartedAt: 0,
    recognitionCooldown: null,
    modelSwitchBusy: false,
    modelSelectionKey: null,
    cameraBusy: false,
    cameraRequest: 0,
    imageUrl: null,
    predictionHistory: [],
    selectionLockUntil: 0,
    manualSelection: false,
    timers: {},
    timerElements: new Map(),
    durations: {},
    counts: {},
    noticeTimeout: null,
    saveTimeout: null,
    configSaveVersion: 0,
    modelPoll: null,
    audioContext: null,
    sampleBusy: false,
    sampleLabelManuallySet: false,
  };

  for (const item of state.ingredients) {
    const duration = saved.durations?.[item.id];
    if (validDuration(duration)) {
      item.duration_seconds = Number(duration);
      state.durations[item.id] = Number(duration);
    }
    const timer = saved.timers?.[item.id];
    if (timer && Number.isFinite(timer.deadline) && Number.isFinite(timer.started_at)
      && validDuration(timer.duration) && timer.deadline >= timer.started_at) {
      state.timers[item.id] = { ...timer, notified: Boolean(timer.notified) };
    }
  }
  ui.soundNotifications.checked = saved.sound === true;
  ui.voiceNotifications.checked = saved.voice === true;
  if (typeof saved.session_id === "string") ui.sessionId.value = saved.session_id.slice(0, 80);

  function validDuration(value) {
    return value !== null && value !== "" && Number.isInteger(Number(value))
      && Number(value) >= 1 && Number(value) <= 7200;
  }

  function ingredient(id) { return state.ingredients.find((item) => item.id === id); }

  function cloudRecognition() {
    return state.model.provider === "openai" || state.model.cloud === true;
  }

  function recognitionWait() {
    const minimum = cloudRecognition() ? Math.max(3000, Number(state.model.min_interval_ms) || 0) : 0;
    return Math.max(0, state.recognitionStartedAt + minimum - Date.now());
  }

  function persist() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        timers: state.timers, durations: state.durations,
        sound: ui.soundNotifications.checked, voice: ui.voiceNotifications.checked,
        session_id: ui.sessionId.value,
      }));
    } catch {
      notice("Browser storage is unavailable. Keep this tab open to preserve your timers.");
    }
  }

  function notice(message, success = false) {
    clearTimeout(state.noticeTimeout);
    ui.globalMessage.textContent = message;
    ui.globalMessage.classList.toggle("success", success);
    ui.globalMessage.hidden = !message;
    if (success) state.noticeTimeout = setTimeout(() => { ui.globalMessage.hidden = true; }, 7000);
  }

  async function api(path, options = {}) {
    const controller = options.signal ? null : new AbortController();
    const timeout = controller ? setTimeout(() => controller.abort(), 45000) : null;
    try {
      const response = await fetch(path, { ...options, signal: options.signal || controller.signal });
      let payload;
      try { payload = await response.json(); } catch { throw new Error(`Server returned an unreadable response (${response.status}).`); }
      if (!response.ok) {
        const detail = payload.detail || payload.message || payload.error;
        throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status}).`);
      }
      return payload;
    } catch (error) {
      if (error.name === "AbortError") throw error;
      if (error instanceof TypeError) throw new Error("Cannot reach the local server. Check that it is running.");
      throw error;
    } finally { clearTimeout(timeout); }
  }

  function iconFor(id) {
    const paths = {
      beef_rolls: ["M8 8c-5 0-5 12 0 12h13c5 0 5-12 0-12H8Z", "M8 8c5 0 5 12 0 12", "M8 11c2 0 2 6 0 6", "M13 8v12M18 8v12"],
      crab_sticks: ["m6 8 4-3 16 14-4 4L6 8Z", "m10 5-4 3M9 10l14 12M12 7l14 12"],
      napa_cabbage: ["M14 25C3 20 4 6 10 5c2-4 7-3 8 0 6-1 10 12-4 20Z", "M14 25V9M14 15l-5-5M14 19l7-7M14 22l-6-5"],
      lotus_root: ["M26 15a11 11 0 1 1-22 0 11 11 0 0 1 22 0Z", "M17 7a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z", "M9 10a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z", "M24 11a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z", "M12 18a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z", "M22 20a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z", "M17 14a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"],
      soft_tofu: ["m4 11 11-6 11 6-11 6-11-6Z", "M4 11v10l11 6 11-6V11M15 17v10"],
      wide_noodles: ["M7 4c9 4-4 9 3 13s9 4 5 10", "M12 3c9 4-4 9 3 13s9 4 5 10", "M17 2c9 4-4 9 3 13s9 4 5 10"],
      udon_noodles: ["M5 5c13-2 16 3 9 6s-9 7 5 7c8 0 8 6-2 7", "M5 8c13-2 16 3 9 6s-9 7 5 7c4 0 5 2 3 4"],
      shiitake_mushrooms: ["M3 17C3 2 27 2 27 17H3Z", "M12 17v9h6v-9", "m11 8 8 6M19 8l-8 6", "M7 18c3 3 13 3 16 0"],
      beef_meatballs: ["M15 11a5 5 0 1 1-10 0 5 5 0 0 1 10 0Z", "M27 13a5 5 0 1 1-10 0 5 5 0 0 1 10 0Z", "M21 24a5 5 0 1 1-10 0 5 5 0 0 1 10 0Z", "m9 9 2 1M21 11l2 1M15 22l2 1"],
    };
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 30 30");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "1.25");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    svg.classList.add("food-icon");
    for (const d of paths[id] || []) {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", d);
      svg.append(path);
    }
    return svg;
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderIngredients() {
    ui.ingredientCount.textContent = String(state.ingredients.length);
    ui.ingredientGrid.replaceChildren();
    const previousLabel = ui.sampleLabel.value;
    ui.sampleLabel.replaceChildren(new Option("Choose a label", ""));
    for (const item of state.ingredients) {
      const button = element("button", "ingredient-button");
      button.type = "button";
      button.dataset.ingredientId = item.id;
      button.setAttribute("aria-pressed", String(item.id === state.selected));
      button.classList.toggle("selected", item.id === state.selected);
      button.append(iconFor(item.id), element("strong", "", item.name), element("small", "", item.en));
      if (state.timers[item.id]) button.append(element("span", "active-dot"));
      button.addEventListener("click", () => selectIngredient(item.id, true));
      ui.ingredientGrid.append(button);
      ui.sampleLabel.append(new Option(`${item.name} · ${item.en}`, item.id));
    }
    ui.sampleLabel.append(new Option("Background · 空画面 / 手 / 其他食材", "background"));
    ui.sampleLabel.value = ingredient(previousLabel) || previousLabel === "background" ? previousLabel : state.selected || "";
    renderDatasetCounts();
  }

  function selectIngredient(id, manual = false) {
    const item = ingredient(id);
    if (!item) return;
    if (manual) {
      state.selectionLockUntil = Date.now() + 3500;
      state.manualSelection = true;
    }
    const changed = state.selected !== id;
    state.selected = id;
    for (const button of ui.ingredientGrid.children) {
      const selected = button.dataset.ingredientId === id;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    }
    ui.selectedName.textContent = item.name;
    ui.durationInput.disabled = false;
    if (changed || document.activeElement !== ui.durationInput) {
      ui.durationInput.value = item.duration_seconds ?? "";
    }
    if (!state.sampleLabelManuallySet) ui.sampleLabel.value = id;
    ui.durationSaveStatus.textContent = "";
    updateStartControl();
    updateCaptureControl();
  }

  function updateStartControl() {
    const item = ingredient(state.selected);
    const active = item && state.timers[item.id];
    const duration = item && item.duration_seconds;
    ui.startTimer.disabled = !item || !validDuration(duration) || Boolean(active);
    ui.startTimer.firstElementChild.textContent = active
      ? "Already in the pot · 计时中" : "Added to pot · 开始计时";
    ui.timerHelp.textContent = !item ? "Select an ingredient, then set your preferred duration."
      : active ? "Dismiss or cancel this ingredient’s existing timer before starting another."
      : !validDuration(duration) ? "Set your preferred duration in seconds to start this timer."
      : `${state.manualSelection ? "Selected manually. " : ""}${timeText(duration)} configured · Press when you add it. Recognition alone never starts a timer.`;
  }

  async function loadConfig() {
    const config = await api("/api/config");
    if (Array.isArray(config.ingredients) && config.ingredients.length) {
      state.ingredients = config.ingredients.filter((item) => item && typeof item.id === "string")
        .map((item) => ({
          ...item,
          name: item.name || item.id,
          en: item.en || "",
          duration_seconds: validDuration(item.duration_seconds) ? Number(item.duration_seconds) : null,
        }));
      for (const item of state.ingredients) {
        state.durations[item.id] = item.duration_seconds;
      }
    }
    if (!ui.sessionId.value) ui.sessionId.value = config.session_id || `session-${new Date().toISOString().slice(0, 10)}`;
    renderIngredients();
    if (state.selected && ingredient(state.selected)) selectIngredient(state.selected);
    renderTimers();
    persist();
  }

  async function saveDurations(version) {
    try {
      await api("/api/config", {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ durations: state.durations }),
      });
      if (version === state.configSaveVersion) ui.durationSaveStatus.textContent = "Saved";
    } catch (error) {
      if (version === state.configSaveVersion) ui.durationSaveStatus.textContent = "Local";
      notice(`${error.message} Your duration is saved in this browser and can still be used.`);
    }
  }

  function changeDuration() {
    const item = ingredient(state.selected);
    if (!item) return;
    state.selectionLockUntil = Date.now() + 3500;
    const value = ui.durationInput.value;
    if (!validDuration(value)) {
      item.duration_seconds = null;
      ui.durationSaveStatus.textContent = "";
      updateStartControl();
      clearTimeout(state.saveTimeout);
      const version = ++state.configSaveVersion;
      if (value === "") {
        state.durations[item.id] = null;
        persist();
        ui.durationSaveStatus.textContent = "Saving…";
        state.saveTimeout = setTimeout(() => saveDurations(version), 500);
      } else {
        ui.durationSaveStatus.textContent = "1–7200";
      }
      return;
    }
    item.duration_seconds = Number(value);
    state.durations[item.id] = Number(value);
    persist();
    updateStartControl();
    ui.durationSaveStatus.textContent = "Saving…";
    clearTimeout(state.saveTimeout);
    const version = ++state.configSaveVersion;
    state.saveTimeout = setTimeout(() => saveDurations(version), 500);
  }

  function startTimer() {
    const item = ingredient(state.selected);
    if (!item || !validDuration(item.duration_seconds) || state.timers[item.id]) return;
    unlockAudio();
    const now = Date.now();
    state.timers[item.id] = {
      started_at: now, deadline: now + item.duration_seconds * 1000,
      duration: item.duration_seconds, notified: false,
    };
    state.manualSelection = false;
    persist();
    renderTimers();
    renderIngredients();
    updateStartControl();
  }

  function removeTimer(id) {
    delete state.timers[id];
    persist();
    renderTimers();
    renderIngredients();
    updateStartControl();
  }

  function timeText(seconds) {
    const total = Math.max(0, Math.ceil(seconds));
    if (total >= 3600) return `${Math.floor(total / 3600)}:${String(Math.floor(total / 60) % 60).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
  }

  function renderTimers() {
    ui.timerList.replaceChildren();
    state.timerElements.clear();
    const entries = Object.entries(state.timers).filter(([id]) => ingredient(id))
      .sort((a, b) => a[1].deadline - b[1].deadline);
    if (!entries.length) {
      const empty = element("div", "timer-empty");
      const icon = element("span", "", "◷");
      icon.setAttribute("aria-hidden", "true");
      const copy = element("p", "", "A timer for every ingredient.");
      const detail = element("small", "", "One active timer per ingredient. Repeating a recognition won’t restart it.");
      copy.append(detail);
      empty.append(icon, copy);
      ui.timerList.append(empty);
    }
    for (const [id] of entries) {
      const item = ingredient(id);
      const card = element("div", "timer-item");
      const top = element("div", "timer-item-top");
      const name = element("div", "timer-item-name", item.name);
      name.append(element("small", "", item.en));
      const countdown = element("span", "countdown");
      top.append(name, countdown);
      const progress = element("div", "timer-progress");
      const progressFill = element("span");
      progress.append(progressFill);
      const bottom = element("div", "timer-item-bottom");
      const detail = element("span");
      const action = element("button", "", "Cancel");
      action.type = "button";
      action.addEventListener("click", () => removeTimer(id));
      bottom.append(detail, action);
      card.append(top, progress, bottom);
      ui.timerList.append(card);
      state.timerElements.set(id, { card, countdown, progressFill, detail, action, due: false });
    }
    tickTimers();
  }

  function tickTimers() {
    const now = Date.now();
    let active = 0;
    let due = 0;
    let needsPersist = false;
    for (const [id, timer] of Object.entries(state.timers)) {
      if (!ingredient(id)) continue;
      const nodes = state.timerElements.get(id);
      const remaining = (timer.deadline - now) / 1000;
      const isDue = remaining <= 0;
      if (isDue) due++; else active++;
      if (nodes) {
        nodes.card.classList.toggle("due", isDue);
        nodes.countdown.textContent = isDue ? "Time’s up" : timeText(remaining);
        nodes.progressFill.style.width = `${Math.max(0, Math.min(100, (1 - remaining / timer.duration) * 100))}%`;
        nodes.detail.textContent = isDue ? `到时间了 · ${timeText(-remaining)} past timer`
          : `${timer.duration}s configured · started ${new Date(timer.started_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
        nodes.action.textContent = isDue ? "Dismiss · 已取出" : "Cancel";
        if (isDue && !nodes.due) nodes.card.setAttribute("role", "alert");
        nodes.due = isDue;
      }
      if (isDue && !timer.notified) {
        timer.notified = true;
        notifyTimer(ingredient(id));
        needsPersist = true;
      }
    }
    const total = active + due;
    ui.timerCount.textContent = `${total} ${due ? "IN POT" : "ACTIVE"}`;
    ui.timerCount.classList.toggle("live", total > 0);
    ui.activeSummary.textContent = due ? `${due} ready for your attention` : active ? `${active} counting down` : "Nothing added yet";
    if (needsPersist) persist();
  }

  function unlockAudio() {
    if (!ui.soundNotifications.checked) return;
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    try {
      state.audioContext ||= new AudioContext();
      if (state.audioContext.state === "suspended") state.audioContext.resume().catch(() => {});
    } catch { /* The visible reminder remains available. */ }
  }

  function beep() {
    unlockAudio();
    const context = state.audioContext;
    if (!context || context.state !== "running") return;
    for (let i = 0; i < 2; i++) {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const start = context.currentTime + i * 0.22;
      oscillator.type = "sine";
      oscillator.frequency.value = i ? 880 : 660;
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(0.13, start + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.001, start + 0.16);
      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start(start);
      oscillator.stop(start + 0.18);
    }
  }

  function notifyTimer(item) {
    if (ui.soundNotifications.checked) beep();
    if (ui.voiceNotifications.checked && "speechSynthesis" in window) {
      const speech = new SpeechSynthesisUtterance(`${item.name}，到时间了`);
      speech.lang = "zh-CN";
      speech.rate = 0.95;
      window.speechSynthesis.speak(speech);
    }
  }

  async function listCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const previous = ui.cameraSelect.value;
    const devices = await navigator.mediaDevices.enumerateDevices();
    ui.cameraSelect.replaceChildren(new Option("Default camera", ""));
    let index = 0;
    for (const device of devices.filter((device) => device.kind === "videoinput")) {
      ui.cameraSelect.append(new Option(device.label || `Camera ${++index}`, device.deviceId));
    }
    if ([...ui.cameraSelect.options].some((option) => option.value === previous)) ui.cameraSelect.value = previous;
  }

  function invalidateRecognition(keepManualSelection = false) {
    state.generation++;
    state.predictionHistory = [];
    if (!keepManualSelection) state.manualSelection = false;
    ui.frameResultName.textContent = "Hold one ingredient in the guide.";
    ui.frameResultState.textContent = "WAITING TO RECOGNIZE";
    ui.frameResultScore.textContent = "";
    ui.confidenceNote.hidden = true;
    ui.frameResult.classList.remove("confident", "uncertain");
    clearTimeout(state.recognitionTimeout);
    state.recognitionTimeout = null;
    state.recognitionAbort?.abort();
    clearTimeout(state.recognitionCooldown);
    state.recognitionCooldown = null;
  }

  function releaseCamera() {
    if (state.stream) {
      for (const track of state.stream.getTracks()) track.stop();
      state.stream = null;
    }
    ui.cameraVideo.srcObject = null;
  }

  function updateSourceUI() {
    const hasSource = Boolean(state.source);
    ui.cameraEmpty.hidden = hasSource;
    ui.cropGuide.hidden = !hasSource;
    ui.frameResult.hidden = !hasSource;
    ui.cameraVideo.hidden = state.source !== "camera";
    ui.uploadedImage.hidden = state.source !== "upload";
    ui.sourceBadge.textContent = state.source === "camera" ? "CAMERA LIVE" : state.source === "upload" ? "UPLOADED IMAGE" : "NO SOURCE";
    ui.sourceBadge.classList.toggle("live", state.source === "camera");
    ui.frameLabel.textContent = state.source === "camera" ? "GUIDE CROP · LIVE" : state.source === "upload" ? "GUIDE CROP · STILL" : "WAITING FOR AN IMAGE";
    ui.startCamera.disabled = state.cameraBusy;
    ui.emptyStartCamera.disabled = state.cameraBusy;
    ui.startCamera.textContent = state.cameraBusy ? "Opening…" : state.source === "camera" ? "Switch camera" : "Start camera";
    ui.stopCamera.disabled = state.source !== "camera" && !state.cameraBusy;
    const wait = recognitionWait();
    ui.recognizeOnce.disabled = !hasSource || state.recognitionBusy || state.modelSwitchBusy || state.model.state !== "ready" || wait > 0;
    ui.recognizeButtonLabel.textContent = state.recognitionBusy ? "Recognizing…" : hasSource && wait > 0 ? `Wait ${Math.ceil(wait / 1000)}s` : "Recognize frame";
    clearTimeout(state.recognitionCooldown);
    if (hasSource && wait > 0) state.recognitionCooldown = setTimeout(updateSourceUI, Math.min(250, wait));
    updateCaptureControl();
  }

  async function startCamera() {
    if (state.cameraBusy) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      notice("Camera access needs localhost or HTTPS and a supported browser. You can still upload an image.");
      return;
    }
    const request = ++state.cameraRequest;
    state.cameraBusy = true;
    updateSourceUI();
    let stream;
    try {
      const deviceId = ui.cameraSelect.value;
      stream = await navigator.mediaDevices.getUserMedia({
        video: { ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "environment" }), width: { ideal: 1280 }, height: { ideal: 960 } },
        audio: false,
      });
      if (request !== state.cameraRequest) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      invalidateRecognition();
      releaseCamera();
      state.stream = stream;
      ui.cameraVideo.srcObject = stream;
      await ui.cameraVideo.play();
      if (request !== state.cameraRequest) return;
      state.source = "camera";
      for (const track of stream.getVideoTracks()) {
        track.addEventListener("ended", () => {
          if (state.stream === stream) {
            stopCamera();
            notice("The camera disconnected. Reconnect it or upload an image.");
          }
        });
      }
      await listCameras();
      ui.globalMessage.hidden = true;
      scheduleRecognition(100);
    } catch (error) {
      if (request !== state.cameraRequest) return;
      if (stream) stream.getTracks().forEach((track) => track.stop());
      const messages = {
        NotAllowedError: "Camera permission was denied. Allow camera access in your browser, or upload an image.",
        NotFoundError: "No camera was found. Connect a camera, or upload an image.",
        NotReadableError: "The camera is busy or unavailable. Close other camera apps and try again.",
        OverconstrainedError: "This camera is unavailable. Choose a different camera and try again.",
      };
      notice(messages[error.name] || `Could not open the camera: ${error.message}`);
      if (stream && state.stream === stream) releaseCamera();
      if (!state.stream) state.source = null;
    } finally {
      if (request === state.cameraRequest) {
        state.cameraBusy = false;
        updateSourceUI();
      }
    }
  }

  function stopCamera() {
    state.cameraRequest++;
    state.cameraBusy = false;
    invalidateRecognition();
    releaseCamera();
    state.source = null;
    ui.inferenceLabel.textContent = "—";
    updateSourceUI();
  }

  async function uploadImage(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) { notice("Choose an image file."); return; }
    if (file.size > 20 * 1024 * 1024) { notice("Choose an image smaller than 20 MB."); return; }
    const url = URL.createObjectURL(file);
    const probe = new Image();
    try {
      await new Promise((resolve, reject) => {
        probe.onload = resolve;
        probe.onerror = () => reject(new Error("This image could not be decoded. Try JPEG or PNG."));
        probe.src = url;
      });
      state.cameraRequest++;
      state.cameraBusy = false;
      invalidateRecognition();
      releaseCamera();
      if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
      state.imageUrl = url;
      ui.uploadedImage.src = url;
      await ui.uploadedImage.decode();
      state.source = "upload";
      updateSourceUI();
      ui.globalMessage.hidden = true;
      if (state.model.state === "ready" && !cloudRecognition()) recognize(false);
    } catch (error) {
      URL.revokeObjectURL(url);
      notice(error.message);
    } finally { ui.imageUpload.value = ""; }
  }

  async function captureCrop() {
    const source = state.source === "camera" ? ui.cameraVideo : state.source === "upload" ? ui.uploadedImage : null;
    if (!source) throw new Error("Start the camera or upload an image first.");
    const width = state.source === "camera" ? source.videoWidth : source.naturalWidth;
    const height = state.source === "camera" ? source.videoHeight : source.naturalHeight;
    if (!width || !height) throw new Error("The image is not ready yet. Try again in a moment.");
    const viewport = source.getBoundingClientRect();
    const guide = ui.cropGuide.getBoundingClientRect();
    const scale = Math.max(viewport.width / width, viewport.height / height);
    const size = Math.min(width, height, guide.width / scale);
    const displayedWidth = width * scale;
    const displayedHeight = height * scale;
    const centerX = (guide.left + guide.width / 2 - viewport.left + (displayedWidth - viewport.width) / 2) / scale;
    const centerY = (guide.top + guide.height / 2 - viewport.top + (displayedHeight - viewport.height) / 2) / scale;
    const sourceX = Math.max(0, Math.min(width - size, centerX - size / 2));
    const sourceY = Math.max(0, Math.min(height - size, centerY - size / 2));
    const canvas = ui.captureCanvas;
    const outputSize = Math.min(768, Math.round(size));
    canvas.width = outputSize;
    canvas.height = outputSize;
    const context = canvas.getContext("2d");
    context.drawImage(source, sourceX, sourceY, size, size, 0, 0, outputSize, outputSize);
    const capturedAt = Date.now();
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) throw new Error("Could not capture this image.");
    return { blob, capturedAt };
  }

  function scheduleRecognition(delay) {
    clearTimeout(state.recognitionTimeout);
    state.recognitionTimeout = null;
    if (document.hidden || state.source !== "camera" || !ui.autoRecognize.checked || state.model.state !== "ready" || state.recognitionBusy || state.modelSwitchBusy) return;
    const interval = cloudRecognition() ? 5000 : 750;
    const requestedDelay = delay === undefined ? interval : cloudRecognition() ? Math.max(interval, delay) : delay;
    state.recognitionTimeout = setTimeout(() => {
      state.recognitionTimeout = null;
      recognize(true);
    }, Math.max(requestedDelay, recognitionWait()));
  }

  async function recognize(automatic = false) {
    if (state.recognitionBusy || state.modelSwitchBusy || !state.source) return;
    if (recognitionWait() > 0) {
      updateSourceUI();
      if (automatic) scheduleRecognition();
      return;
    }
    if (state.model.state !== "ready") {
      if (!automatic) notice(state.model.message || "The recognition model is not ready. Manual ingredient selection and timers still work.");
      return;
    }
    const generation = state.generation;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 45000);
    state.recognitionBusy = true;
    state.recognitionAbort = controller;
    ui.recognizeOnce.disabled = true;
    ui.inferenceLabel.textContent = "RECOGNIZING…";
    try {
      const { blob, capturedAt } = await captureCrop();
      if (generation !== state.generation) return;
      const form = new FormData();
      form.append("image", blob, "frame.jpg");
      form.append("captured_at", String(capturedAt));
      state.recognitionStartedAt = Date.now();
      const prediction = await api("/api/recognize", { method: "POST", body: form, signal: controller.signal });
      if (generation !== state.generation) return;
      if (prediction.usage) {
        state.model.usage = prediction.usage;
        if (Number.isFinite(state.model.request_limit) && Number.isFinite(prediction.usage.attempts)) {
          state.model.attempts_remaining = Math.max(0, state.model.request_limit - prediction.usage.attempts);
        }
        renderUsage();
      }
      showPrediction(prediction, automatic);
      ui.inferenceLabel.textContent = Number.isFinite(prediction.inference_ms) ? `${Math.round(prediction.inference_ms)} MS` : "FRAME RECOGNIZED";
    } catch (error) {
      if (generation !== state.generation) return;
      ui.inferenceLabel.textContent = "RECOGNITION ERROR";
      ui.confidenceNote.hidden = true;
      ui.confidenceBadge.textContent = "RECOGNITION ERROR";
      ui.confidenceBadge.className = "confidence-badge uncertain";
      ui.confidenceBadge.title = "";
      if (error.name !== "AbortError") {
        ui.predictionDetail.textContent = error.message;
        ui.frameResultName.textContent = "Recognition unavailable.";
        ui.frameResultState.textContent = "TRY AGAIN OR CHOOSE MANUALLY";
        ui.frameResultScore.textContent = "";
        ui.frameResult.classList.remove("confident", "uncertain");
        if (!automatic) notice(error.message);
      } else if (!automatic) notice("Recognition timed out. Try again or check the local server.");
      if (cloudRecognition()) {
        // A retry can create another billable request; require an explicit retry.
        ui.autoRecognize.checked = false;
        clearTimeout(state.recognitionTimeout);
        state.recognitionTimeout = null;
        if (automatic) notice(`${error.name === "AbortError" ? "Recognition timed out." : error.message} Live recognition paused. Use “Recognize frame” to try again.`);
        loadModel().catch(() => {});
      } else if (automatic) {
        // A failed model call is retried with a delay rather than flooding the server.
        clearTimeout(state.recognitionTimeout);
        state.recognitionTimeout = setTimeout(() => {
          state.recognitionTimeout = null;
          loadModel().catch(() => {});
          scheduleRecognition(2500);
        }, 2500);
      }
    } finally {
      clearTimeout(timeout);
      state.recognitionBusy = false;
      if (state.recognitionAbort === controller) state.recognitionAbort = null;
      updateSourceUI();
      if (generation !== state.generation && state.source === "upload" && state.model.state === "ready" && !cloudRecognition()) setTimeout(() => recognize(false), 0);
      else if (!state.recognitionTimeout) scheduleRecognition();
    }
  }

  function showPrediction(prediction, smooth) {
    const rawScores = Array.isArray(prediction.scores) ? prediction.scores.filter((score) => (ingredient(score.id) || score.id === "background") && Number.isFinite(score.score)) : [];
    state.predictionHistory.push(rawScores);
    state.predictionHistory = state.predictionHistory.slice(-3);
    const histories = smooth ? state.predictionHistory : [rawScores];
    const scores = [...state.ingredients, { id: "background" }].map((item) => ({
      id: item.id,
      score: histories.reduce((total, frame) => total + (frame.find((score) => score.id === item.id)?.score || 0), 0) / histories.length,
    })).sort((a, b) => b.score - a.score);
    const returnedItem = ingredient(prediction.ingredient_id);
    const topRaw = [...rawScores].sort((a, b) => b.score - a.score)[0];
    const isBackground = !returnedItem && topRaw?.id === "background";
    const item = returnedItem || ingredient(topRaw?.id);
    const confidenceLevel = cloudRecognition() && ["high", "medium", "low"].includes(prediction.confidence_level)
      ? prediction.confidence_level : null;
    const uncertain = prediction.uncertain === true || !returnedItem || confidenceLevel === "low";
    const confidence = cloudRecognition() ? null : Number.isFinite(prediction.confidence) ? prediction.confidence : rawScores[0]?.score;
    const unknown = cloudRecognition() && !returnedItem;
    const ambiguous = prediction.decision === "ambiguous" || prediction.status === "ambiguous"
      || (cloudRecognition() && Boolean(returnedItem) && uncertain);
    ui.predictionName.textContent = item ? uncertain ? `Maybe ${item.name}?` : item.name : isBackground ? "No ingredient recognized." : "Not sure what this is.";
    ui.predictionDetail.textContent = item
      ? `${item.en} · ${uncertain ? "Choose the correct ingredient below, or try another angle." : "Press “Added to pot” when you put it in."}`
      : isBackground ? "Background, a hand, or another food. Hold one supported ingredient inside the guide."
      : "Try one ingredient inside the guide, or choose it manually.";
    if (typeof prediction.reason === "string" && prediction.reason.trim()) {
      ui.predictionDetail.textContent = `${prediction.reason.trim()} ${uncertain ? "Choose manually or try another angle." : "Press “Added to pot” when you put it in."}`;
    }
    ui.confidenceBadge.textContent = confidenceLevel ? `${confidenceLevel.toUpperCase()} · GPT ESTIMATE`
      : ambiguous ? "AMBIGUOUS" : unknown ? "UNKNOWN" : uncertain ? "UNCERTAIN" : Number.isFinite(confidence) ? `${confidence.toFixed(3)} MATCH SCORE` : "RECOGNIZED";
    ui.confidenceBadge.className = `confidence-badge ${uncertain ? "uncertain" : "confident"}`;
    ui.confidenceBadge.title = confidenceLevel ? "GPT's own uncalibrated assessment, not a validated probability." : "";
    ui.confidenceNote.hidden = !confidenceLevel;
    ui.frameResultName.textContent = ui.predictionName.textContent;
    ui.frameResultState.textContent = ambiguous ? "AMBIGUOUS · 请确认食材" : unknown ? "UNKNOWN · 无法识别" : uncertain ? "UNCERTAIN · 请确认食材" : "RECOGNIZED · 已识别";
    ui.frameResultScore.textContent = confidenceLevel ? `${confidenceLevel.toUpperCase()} GPT estimate`
      : !uncertain && Number.isFinite(confidence) ? `${confidence.toFixed(3)} match score` : "";
    ui.frameResultScore.title = confidenceLevel ? ui.confidenceBadge.title : "";
    ui.frameResult.classList.toggle("confident", !uncertain);
    ui.frameResult.classList.toggle("uncertain", uncertain);
    ui.scoreBars.replaceChildren();
    for (const score of scores.slice(0, 3)) {
      const scoreItem = element("div", "score-item");
      const label = element("div", "score-item-label");
      label.append(element("span", "", score.id === "background" ? "Background" : ingredient(score.id).name), element("span", "", score.score.toFixed(3)));
      const track = element("div", "score-track");
      const fill = element("div", "score-fill");
      fill.style.width = `${Math.min(100, Math.max(0, score.score * 100))}%`;
      track.append(fill);
      scoreItem.append(label, track);
      ui.scoreBars.append(scoreItem);
    }
    ui.scoreBars.hidden = cloudRecognition() || !rawScores.length;
    const stable = cloudRecognition() || !smooth || histories.length < 2 || scores[0]?.id === returnedItem?.id;
    if (!uncertain && returnedItem && stable && !state.manualSelection && Date.now() >= state.selectionLockUntil
      && document.activeElement !== ui.durationInput) selectIngredient(returnedItem.id);
  }

  function renderModel() {
    const model = state.model;
    const cloud = cloudRecognition();
    const selectionKey = cloud ? `openai:${model.model || "gpt-5.4"}` : "local";
    if (selectionKey !== state.modelSelectionKey) {
      state.modelSelectionKey = selectionKey;
      ui.autoRecognize.checked = !cloud;
      clearTimeout(state.recognitionTimeout);
      state.recognitionTimeout = null;
      ui.confidenceBadge.textContent = "WAITING";
      ui.confidenceBadge.className = "confidence-badge";
      ui.confidenceBadge.title = "";
      ui.confidenceNote.hidden = true;
      ui.predictionName.textContent = "Ready to recognize an ingredient.";
      ui.predictionDetail.textContent = cloud ? "Click “Recognize frame” to send this guide crop to OpenAI." : "The crop inside your camera guide is sent to the local model.";
      ui.scoreBars.hidden = true;
      ui.scoreBars.replaceChildren();
      ui.frameResultScore.textContent = "";
    }
    ui.recognitionModel.value = selectionKey;
    ui.recognitionModel.disabled = state.modelSwitchBusy || model.state === "loading";
    ui.prototypeTag.textContent = cloud ? "GPT VISION PROTOTYPE" : "LOCAL PROTOTYPE";
    ui.sourceHelp.textContent = cloud ? "Cropped images are sent to OpenAI when you recognize them." : "Recognition runs on your laptop.";
    ui.liveRecognitionLabel.textContent = cloud ? "Live recognition · every 5s" : "Live recognition";
    ui.recognitionCaption.textContent = cloud ? "Sends cropped frames to OpenAI. API billed separately. Click “Recognize frame” to test; live mode is optional." : "Only the crop inside the guide is sent to your local model. No API charges.";
    ui.modelStatusDot.className = `status-dot ${model.state}`;
    ui.modelLabel.textContent = model.label || (model.state === "ready" ? "Recognition model ready" : model.state === "loading" ? "Loading recognition model…" : "Recognition model unavailable");
    ui.modelMessage.textContent = model.message || (model.state !== "ready" ? "You can select ingredients manually and use all timers while the model is unavailable." : "");
    ui.reloadModel.disabled = model.state === "loading";
    renderUsage();
    if (model.state !== "ready") {
      ui.confidenceBadge.textContent = model.state === "loading" ? "MODEL LOADING" : "MANUAL MODE AVAILABLE";
      ui.confidenceBadge.className = "confidence-badge";
      ui.confidenceBadge.title = "";
      ui.confidenceNote.hidden = true;
      ui.predictionName.textContent = model.state === "loading" ? "Getting the model ready." : "Recognition needs a model.";
      ui.predictionDetail.textContent = "Choose any ingredient on the right to use timers now.";
      ui.scoreBars.hidden = true;
      ui.frameResultName.textContent = model.state === "loading" ? "Getting the model ready." : "Choose an ingredient manually.";
      ui.frameResultState.textContent = model.state === "loading" ? "MODEL LOADING" : "RECOGNITION UNAVAILABLE";
      ui.frameResultScore.textContent = "";
      ui.frameResult.classList.remove("confident", "uncertain");
    }
    updateSourceUI();
  }

  function renderUsage() {
    ui.modelUsage.hidden = !cloudRecognition();
    if (!cloudRecognition()) return;
    const usage = state.model.usage || {};
    const attempts = Number.isFinite(usage.attempts) ? usage.attempts : 0;
    const cap = Number.isFinite(state.model.request_limit) ? ` / ${state.model.request_limit}` : "";
    const estimate = Number.isFinite(usage.estimated_usd) ? `$${usage.estimated_usd.toFixed(4)} estimated` : "Cost estimate unavailable";
    ui.modelUsage.textContent = `${attempts}${cap} requests attempted · ${estimate} this server session. ${usage.price_note || ""}`.trim();
  }

  async function changeRecognitionModel() {
    const choice = ui.recognitionModel.value;
    if (state.modelSwitchBusy || choice === state.modelSelectionKey) return;
    invalidateRecognition(true);
    state.modelSwitchBusy = true;
    ui.autoRecognize.checked = false;
    const cloud = choice.startsWith("openai:");
    try {
      ui.recognitionModel.disabled = true;
      ui.recognizeOnce.disabled = true;
      const payload = cloud ? { provider: "openai", model: choice.slice("openai:".length) } : { provider: "local" };
      state.model = await api("/api/model/provider", {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      renderModel();
      await loadModel();
    } catch (error) {
      notice(error.message);
      await loadModel().catch(() => {});
    } finally {
      state.modelSwitchBusy = false;
      renderModel();
      scheduleRecognition(100);
    }
  }

  async function loadModel() {
    state.model = await api("/api/model");
    renderModel();
    clearTimeout(state.modelPoll);
    if (state.model.state === "loading") state.modelPoll = setTimeout(() => loadModel().catch(() => {}), 5000);
    if (state.model.state === "ready") scheduleRecognition(100);
  }

  async function reloadModel() {
    invalidateRecognition();
    state.model = { ...state.model, state: "loading", label: "Reloading recognition model…" };
    renderModel();
    ui.reloadModel.disabled = true;
    ui.modelLabel.textContent = "Reloading recognition model…";
    try {
      state.model = await api("/api/model/reload", { method: "POST" });
      renderModel();
      await loadModel();
      if (state.source === "upload" && state.model.state === "ready" && !cloudRecognition()) recognize(false);
    } catch (error) {
      notice(error.message);
      ui.reloadModel.disabled = false;
      await loadModel().catch(() => {});
    }
  }

  function updateCaptureControl() {
    ui.captureSample.disabled = !state.source || !ui.sampleLabel.value || !ui.sessionId.value.trim() || state.sampleBusy;
    ui.captureSample.firstChild.textContent = state.sampleBusy ? "Saving crop… " : "Save labeled crop ";
    if (!state.source) ui.sampleMessage.textContent = "Camera or image needed.";
    else if (!ui.sampleLabel.value) ui.sampleMessage.textContent = "Choose its correct label.";
    else if (!ui.sessionId.value.trim()) ui.sampleMessage.textContent = "Give this recording session a name.";
    else if (!state.sampleBusy && !ui.sampleMessage.dataset.saved) ui.sampleMessage.textContent = "Saves the crop inside the guide.";
  }

  function renderDatasetCounts() {
    ui.datasetCounts.replaceChildren();
    for (const item of state.ingredients) {
      const chip = element("span", "dataset-count-chip", item.name);
      chip.append(element("strong", "", String(state.counts[item.id] || 0)));
      ui.datasetCounts.append(chip);
    }
    const background = element("span", "dataset-count-chip", "Background");
    background.append(element("strong", "", String(state.counts.background || 0)));
    ui.datasetCounts.append(background);
  }

  async function loadDataset() {
    const dataset = await api("/api/dataset");
    state.counts = dataset.counts || {};
    renderDatasetCounts();
  }

  async function captureSample() {
    const label = ui.sampleLabel.value;
    const session = ui.sessionId.value.trim();
    if (!state.source || (!ingredient(label) && label !== "background") || !session || state.sampleBusy) return;
    state.sampleBusy = true;
    updateCaptureControl();
    try {
      const { blob, capturedAt } = await captureCrop();
      const form = new FormData();
      form.append("image", blob, "sample.jpg");
      form.append("label", label);
      form.append("session_id", session);
      form.append("captured_at", String(capturedAt));
      const result = await api("/api/samples", { method: "POST", body: form });
      state.counts = result.counts || state.counts;
      renderDatasetCounts();
      ui.sampleMessage.textContent = `Saved ${ingredient(label)?.name || "background"} · ${state.counts[label] || "one more"} samples`;
      ui.sampleMessage.dataset.saved = "true";
      persist();
      // Some backends rebuild a few-shot index after sample capture; read its real status.
      loadModel().catch(() => {});
    } catch (error) {
      ui.sampleMessage.textContent = error.message;
      notice(error.message);
    } finally {
      state.sampleBusy = false;
      updateCaptureControl();
    }
  }

  ui.startCamera.addEventListener("click", startCamera);
  ui.emptyStartCamera.addEventListener("click", startCamera);
  ui.stopCamera.addEventListener("click", stopCamera);
  ui.imageUpload.addEventListener("change", () => uploadImage(ui.imageUpload.files[0]));
  ui.autoRecognize.addEventListener("change", () => {
    if (ui.autoRecognize.checked) scheduleRecognition(100);
    else { clearTimeout(state.recognitionTimeout); state.recognitionTimeout = null; }
  });
  ui.recognizeOnce.addEventListener("click", () => recognize(false));
  ui.recognitionModel.addEventListener("change", changeRecognitionModel);
  ui.reloadModel.addEventListener("click", reloadModel);
  ui.durationInput.addEventListener("input", changeDuration);
  ui.durationInput.addEventListener("focus", () => { state.selectionLockUntil = Date.now() + 3500; });
  ui.startTimer.addEventListener("click", startTimer);
  ui.sampleLabel.addEventListener("change", () => {
    state.sampleLabelManuallySet = Boolean(ui.sampleLabel.value);
    delete ui.sampleMessage.dataset.saved;
    updateCaptureControl();
  });
  ui.sessionId.addEventListener("input", () => {
    persist();
    delete ui.sampleMessage.dataset.saved;
    updateCaptureControl();
  });
  ui.captureSample.addEventListener("click", captureSample);
  for (const checkbox of [ui.soundNotifications, ui.voiceNotifications]) {
    checkbox.addEventListener("change", () => { unlockAudio(); persist(); });
  }
  ui.testAlert.addEventListener("click", () => {
    if (!ui.soundNotifications.checked && !ui.voiceNotifications.checked) {
      notice("Enable sound or spoken reminders, then test the alert.");
      return;
    }
    unlockAudio();
    notifyTimer(ingredient(state.selected) || state.ingredients[0]);
  });
  document.addEventListener("keydown", (event) => {
    if (event.code !== "Space" || event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
    const target = event.target;
    if (target instanceof HTMLElement && (target.closest("input, select, textarea, button, a") || target.isContentEditable)) return;
    event.preventDefault();
    startTimer();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { clearTimeout(state.recognitionTimeout); state.recognitionTimeout = null; }
    else { tickTimers(); scheduleRecognition(100); }
  });
  window.addEventListener("beforeunload", () => {
    state.recognitionAbort?.abort();
    releaseCamera();
    if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  });
  navigator.mediaDevices?.addEventListener("devicechange", () => listCameras().catch(() => {}));

  renderIngredients();
  renderTimers();
  updateStartControl();
  updateSourceUI();
  setInterval(tickTimers, 250);
  Promise.allSettled([loadConfig(), loadModel(), loadDataset(), listCameras()]).then((results) => {
    for (let index = 0; index < results.length; index++) {
      const result = results[index];
      if (result.status !== "rejected") continue;
      if (index === 1) {
        state.model = { state: "unavailable", mode: "unavailable", label: "Recognition model unavailable", message: result.reason.message };
        renderModel();
      }
      if (index < 3) notice(result.reason.message);
    }
    updateCaptureControl();
  });
})();
