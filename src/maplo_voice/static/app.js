/**
 * Maplo Voice browser client.
 *
 * Mic -> AudioWorklet (16 kHz PCM16, 20 ms frames) -> WebSocket -> server VAD/ASR/LLM/TTS
 * Server PCM16 @ 24 kHz -> playback AudioWorklet -> speakers.
 *
 * Security notes: all server text is rendered with textContent (never innerHTML);
 * the token travels as a WebSocket sub-protocol and is never persisted.
 */

const PROTOCOL = "maplo.v1";
const $ = (id) => document.getElementById(id);

const ui = {
  status: $("status"), start: $("start"), commit: $("commit"), level: $("level"),
  log: $("log"), taskField: $("task-field"), task: $("task"), taskPrompt: $("task-prompt"),
  language: $("language"), tenant: $("tenant"), token: $("token"),
  metrics: $("metrics"), sources: $("sources"), assessment: $("assessment"),
  tpl: $("tpl-turn"),
};

const state = {
  ws: null, captureCtx: null, playbackCtx: null, micStream: null,
  captureNode: null, playbackNode: null,
  turns: new Map(),   // turn_index -> { user: li, assistant: li }
  connected: false,
};

// ------------------------------------------------------------------ helpers
function setStatus(text, kind = "idle") {
  ui.status.textContent = text;
  ui.status.dataset.state = kind;
}

function mode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function authHeaders() {
  const headers = { "X-Tenant-ID": ui.tenant.value.trim() || "default" };
  const token = ui.token.value.trim();
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

function addBubble(role, text, extraClass = "") {
  const node = ui.tpl.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  if (extraClass) node.classList.add(extraClass);
  node.querySelector(".who").textContent = role === "user" ? "You" : role === "assistant" ? "Maplo" : "Error";
  node.querySelector(".text").textContent = text;
  ui.log.appendChild(node);
  ui.log.scrollTop = ui.log.scrollHeight;
  return node;
}

function turn(index) {
  if (!state.turns.has(index)) state.turns.set(index, {});
  return state.turns.get(index);
}

function formatMs(value) {
  return value === null || value === undefined ? "–" : `${value} ms`;
}

// ------------------------------------------------------------------ tasks
async function loadTasks() {
  try {
    const res = await fetch("/v1/assessment/tasks", { headers: authHeaders() });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const tasks = await res.json();
    ui.task.replaceChildren(
      ...tasks.map((t) => {
        const opt = document.createElement("option");
        opt.value = t.id;
        opt.textContent = `${t.title} (${t.target_level})`;
        return opt;
      }),
    );
  } catch (err) {
    console.warn("Could not load tasks", err);
  }
}

document.querySelectorAll('input[name="mode"]').forEach((el) =>
  el.addEventListener("change", () => {
    const assessment = mode() === "assessment";
    ui.taskField.hidden = !assessment;
    if (assessment && ui.task.options.length === 0) loadTasks();
  }),
);

// ------------------------------------------------------------------ audio
async function startAudio(inputRate, outputRate) {
  state.micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });

  // Capture at the device's native rate; the worklet resamples (portable across browsers).
  state.captureCtx = new AudioContext();
  await state.captureCtx.audioWorklet.addModule("/static/audio-worklets.js");
  const source = state.captureCtx.createMediaStreamSource(state.micStream);
  state.captureNode = new AudioWorkletNode(state.captureCtx, "capture-processor", {
    processorOptions: { targetRate: inputRate, frameMs: 20 },
  });
  state.captureNode.port.onmessage = ({ data }) => {
    if (data.type === "frame" && state.ws?.readyState === WebSocket.OPEN) {
      state.ws.send(data.buffer);
    } else if (data.type === "level") {
      ui.level.style.width = `${Math.min(100, Math.round(data.rms * 400))}%`;
    }
  };
  source.connect(state.captureNode);

  // Playback context at the TTS rate (no resampling of server audio needed).
  state.playbackCtx = new AudioContext({ sampleRate: outputRate });
  await state.playbackCtx.audioWorklet.addModule("/static/audio-worklets.js");
  state.playbackNode = new AudioWorkletNode(state.playbackCtx, "playback-processor", {
    outputChannelCount: [1],
  });
  state.playbackNode.port.onmessage = ({ data }) => {
    if (data.type === "started") setStatus("Speaking…", "speaking");
    if (data.type === "drained" && state.connected) setStatus("Listening", "connected");
  };
  state.playbackNode.connect(state.playbackCtx.destination);
}

function playChunk(buffer) {
  state.playbackNode?.port.postMessage({ type: "chunk", buffer }, [buffer]);
}

function stopPlayback() {
  state.playbackNode?.port.postMessage({ type: "clear" });
}

async function stopAudio() {
  state.micStream?.getTracks().forEach((t) => t.stop());
  await state.captureCtx?.close().catch(() => {});
  await state.playbackCtx?.close().catch(() => {});
  Object.assign(state, { micStream: null, captureCtx: null, playbackCtx: null, captureNode: null, playbackNode: null });
  ui.level.style.width = "0";
}

// ------------------------------------------------------------------ server events
const handlers = {
  "session.ready": async (ev) => {
    await startAudio(ev.input_sample_rate, ev.output_sample_rate);
    state.connected = true;
    setStatus("Listening", "connected");
    ui.commit.disabled = false;
    if (ev.task_prompt) {
      ui.taskPrompt.textContent = ev.task_prompt;
      ui.taskPrompt.hidden = false;
    }
  },
  "vad.speech_started": () => {
    stopPlayback(); // client-side barge-in: silence stale audio immediately
    setStatus("Hearing you…", "connected");
  },
  "vad.speech_stopped": () => setStatus("Thinking…", "connected"),
  "transcript.partial": (ev) => {
    const t = turn(ev.turn_index);
    t.user ??= addBubble("user", "", "partial");
    t.user.querySelector(".text").textContent = ev.text;
  },
  "transcript.final": (ev) => {
    const t = turn(ev.turn_index);
    if (!ev.text) {
      t.user?.remove();
      setStatus("Listening", "connected");
      return;
    }
    t.user ??= addBubble("user", "");
    t.user.classList.remove("partial");
    t.user.querySelector(".text").textContent = ev.text;
  },
  "response.text.delta": (ev) => {
    const t = turn(ev.turn_index);
    t.assistant ??= addBubble("assistant", "");
    t.assistant.querySelector(".text").textContent += ev.delta;
    ui.log.scrollTop = ui.log.scrollHeight;
  },
  "response.interrupted": (ev) => {
    stopPlayback();
    turn(ev.turn_index).assistant?.classList.add("interrupted");
  },
  "response.done": (ev) => {
    for (const dd of ui.metrics.querySelectorAll("dd")) dd.textContent = formatMs(ev.metrics[dd.dataset.key]);
    ui.sources.hidden = ev.sources.length === 0;
    ui.sources.textContent = ev.sources.length ? `Sources: ${ev.sources.join(", ")}` : "";
  },
  "assessment.result": (ev) => renderAssessment(ev.turn_index, ev.result),
  "error": (ev) => {
    addBubble("error", ev.message, "error");
    if (!ev.retryable && ["invalid_start", "mode_unavailable"].includes(ev.code)) setStatus("Error", "error");
  },
};

function renderAssessment(turnIndex, result) {
  const t = turn(turnIndex);
  t.assistant ??= addBubble("assistant", "");
  if (result.status !== "scored") {
    ui.assessment.hidden = true;
    return;
  }
  ui.assessment.hidden = false;
  $("cefr-level").textContent = result.cefr_level;
  $("cefr-score").textContent = `${result.overall_score} / 100`;
  $("criteria").replaceChildren(
    ...Object.entries(result.scores).map(([name, score]) => {
      const li = document.createElement("li");
      const label = document.createElement("div");
      label.textContent = `${name[0].toUpperCase()}${name.slice(1)} · ${score}`;
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("i");
      fill.style.width = `${score}%`;
      bar.appendChild(fill);
      li.append(label, bar);
      return li;
    }),
  );
  $("wpm").textContent = `${Math.round(result.words_per_minute)} words per minute · ${(result.speech_duration_ms / 1000).toFixed(0)} s of speech`;
  const list = (id, items) =>
    $(id).replaceChildren(...items.map((text) => Object.assign(document.createElement("li"), { textContent: text })));
  list("strengths", result.feedback.strengths || []);
  list("improvements", result.feedback.improvements || []);
  $("corrections").replaceChildren(
    ...(result.feedback.corrections || []).map((c) => {
      const li = document.createElement("li");
      const del = Object.assign(document.createElement("del"), { textContent: c.original });
      const ins = Object.assign(document.createElement("ins"), { textContent: c.corrected });
      li.append(del, " → ", ins);
      return li;
    }),
  );
}

// ------------------------------------------------------------------ connection
function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const tenant = encodeURIComponent(ui.tenant.value.trim() || "default");
  const protocols = [PROTOCOL];
  const token = ui.token.value.trim();
  if (token) protocols.push(`bearer.${token}`);

  const ws = new WebSocket(`${scheme}://${location.host}/ws/voice?tenant=${tenant}`, protocols);
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  setStatus("Connecting…");

  ws.onopen = () => {
    const start = { type: "session.start", mode: mode() };
    if (ui.language.value) start.language = ui.language.value;
    if (mode() === "assessment" && ui.task.value) start.task_id = ui.task.value;
    ws.send(JSON.stringify(start));
  };
  ws.onmessage = async (msg) => {
    if (msg.data instanceof ArrayBuffer) {
      playChunk(msg.data);
      return;
    }
    const ev = JSON.parse(msg.data);
    try {
      await handlers[ev.type]?.(ev);
    } catch (err) {
      console.error(err);
      addBubble("error", err.message || String(err), "error");
      disconnect();
    }
  };
  ws.onclose = (ev) => {
    const wasConnected = state.connected;
    state.connected = false;
    stopAudio();
    ui.start.textContent = "Start talking";
    ui.commit.disabled = true;
    if (ev.code === 1008 && !wasConnected) setStatus("Unauthorized or not allowed", "error");
    else if (ev.code === 1013) setStatus("Server busy — try again", "error");
    else setStatus("Disconnected");
  };
  ws.onerror = () => setStatus("Connection error", "error");
}

function disconnect() {
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: "session.end" }));
  state.ws?.close();
}

ui.start.addEventListener("click", () => {
  if (state.ws && state.ws.readyState <= WebSocket.OPEN) {
    disconnect();
    return;
  }
  state.turns.clear();
  ui.log.replaceChildren();
  ui.assessment.hidden = true;
  ui.taskPrompt.hidden = true;
  ui.start.textContent = "Stop";
  connect();
});

ui.commit.addEventListener("click", () => {
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: "input.commit" }));
});

window.addEventListener("beforeunload", disconnect);
