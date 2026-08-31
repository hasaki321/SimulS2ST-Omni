/**
 * Offscreen document: tab capture → WS uplink @16k → play downlink @24k.
 *
 * Playback notes (Chrome MV3):
 * - AudioContext often starts "suspended"; must resume() after Start.
 * - Play buffers at context.sampleRate (upsample from 24 kHz) for reliability.
 */

import {
  TYPE_JSON,
  TYPE_PCM,
  TYPE_TEXT,
  SAMPLE_RATE_IN,
  SAMPLE_RATE_OUT,
  packJson,
  packPcmI16,
  packEos,
  unpackFrame,
  unpackJson,
  floatToInt16,
  downsample,
} from "./protocol.js";

const FRAME_MS = 1000;

let mediaStream = null;
let audioCtx = null;
let workletNode = null;
let sourceNode = null;
let playGain = null;
let ws = null;
let running = false;
let sendTimer = null;
let pendingFloat = new Float32Array(0);
let playTime = 0;
let audioChunksRecv = 0;

function post(type, payload = {}) {
  chrome.runtime.sendMessage({ type, ...payload }).catch(() => {});
}

function appendFloat(chunk) {
  const merged = new Float32Array(pendingFloat.length + chunk.length);
  merged.set(pendingFloat, 0);
  merged.set(chunk, pendingFloat.length);
  pendingFloat = merged;
}

function takeFrameSamples(n) {
  if (pendingFloat.length < n) return null;
  const frame = pendingFloat.slice(0, n);
  pendingFloat = pendingFloat.subarray(n);
  // Copy remaining into a fresh buffer so we don't retain a huge parent array.
  pendingFloat = pendingFloat.slice();
  return frame;
}

async function ensureRunningCtx() {
  if (!audioCtx) return false;
  if (audioCtx.state === "suspended") {
    try {
      await audioCtx.resume();
    } catch (e) {
      post("ERROR", { error: `AudioContext.resume failed: ${e}` });
      return false;
    }
  }
  return audioCtx.state === "running";
}

/** Upsample float32 from fromRate → toRate (linear). */
function upsample(float32, fromRate, toRate) {
  if (fromRate === toRate) return float32;
  const ratio = toRate / fromRate;
  const outLen = Math.max(1, Math.floor(float32.length * ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const src = i / ratio;
    const i0 = Math.floor(src);
    const i1 = Math.min(float32.length - 1, i0 + 1);
    const t = src - i0;
    out[i] = float32[i0] * (1 - t) + float32[i1] * t;
  }
  return out;
}

function playFloat32AtCtxRate(f32at24k) {
  if (!audioCtx || !playGain) return;
  const f32 = upsample(f32at24k, SAMPLE_RATE_OUT, audioCtx.sampleRate);
  const buf = audioCtx.createBuffer(1, f32.length, audioCtx.sampleRate);
  buf.copyToChannel(f32, 0);
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(playGain);
  const now = audioCtx.currentTime;
  if (playTime < now + 0.01) playTime = now + 0.05;
  src.start(playTime);
  playTime += buf.duration;
}

function playPcm16Bytes(payloadU8) {
  // Copy into a fresh Int16Array (avoid SharedArrayBuffer / odd offset issues).
  const n = Math.floor(payloadU8.byteLength / 2);
  if (n <= 0) return;
  const copy = new ArrayBuffer(n * 2);
  new Uint8Array(copy).set(payloadU8.subarray(0, n * 2));
  const int16 = new Int16Array(copy);
  const f32 = new Float32Array(n);
  for (let i = 0; i < n; i++) f32[i] = int16[i] / 32768;
  playFloat32AtCtxRate(f32);
}

/** Short beep so user can verify speakers / AudioContext. */
function playTestBeep() {
  if (!audioCtx || !playGain) return;
  const dur = 0.15;
  const n = Math.floor(audioCtx.sampleRate * dur);
  const buf = audioCtx.createBuffer(1, n, audioCtx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < n; i++) {
    data[i] = 0.25 * Math.sin((2 * Math.PI * 880 * i) / audioCtx.sampleRate);
  }
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(playGain);
  const t = audioCtx.currentTime + 0.02;
  src.start(t);
  playTime = Math.max(playTime, t + dur + 0.05);
}

async function startSession({ streamId, wsUrl, direction, latency, muteTab }) {
  if (running) await stopSession();
  running = true;
  playTime = 0;
  audioChunksRecv = 0;
  pendingFloat = new Float32Array(0);

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: "tab",
        chromeMediaSourceId: streamId,
      },
    },
    video: false,
  });

  audioCtx = new AudioContext();
  await ensureRunningCtx();

  playGain = audioCtx.createGain();
  playGain.gain.value = 1.0;
  playGain.connect(audioCtx.destination);

  // Prove speakers work right after Start (user gesture chain).
  playTestBeep();
  post("STATUS", {
    status: {
      phase: "audio_ctx",
      state: audioCtx.state,
      sampleRate: audioCtx.sampleRate,
      beep: true,
    },
  });

  sourceNode = audioCtx.createMediaStreamSource(mediaStream);

  // Optionally also hear the original tab (mixed under translation).
  if (!muteTab) {
    const tabGain = audioCtx.createGain();
    tabGain.gain.value = 0.35;
    sourceNode.connect(tabGain);
    tabGain.connect(audioCtx.destination);
  }

  await audioCtx.audioWorklet.addModule(
    chrome.runtime.getURL("capture-processor.js")
  );
  workletNode = new AudioWorkletNode(audioCtx, "capture-processor", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    channelCount: 1,
  });
  workletNode.port.onmessage = (ev) => {
    if (!running) return;
    const input = ev.data;
    const at16k = downsample(input, audioCtx.sampleRate, SAMPLE_RATE_IN);
    appendFloat(at16k);
  };
  sourceNode.connect(workletNode);
  // Keep the graph alive without audible bleed from the capture path.
  const silent = audioCtx.createGain();
  silent.gain.value = 0;
  workletNode.connect(silent);
  silent.connect(audioCtx.destination);

  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  await new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error("WebSocket connect timeout")), 15000);
    ws.onopen = () => {
      clearTimeout(t);
      resolve();
    };
    ws.onerror = () => {
      clearTimeout(t);
      reject(new Error("WebSocket error"));
    };
  });

  ws.send(
    packJson({
      direction,
      latency_multiplier: Number(latency),
      sample_rate: SAMPLE_RATE_IN,
    })
  );

  ws.onmessage = (ev) => {
    const { type, payload } = unpackFrame(ev.data);
    if (type === TYPE_JSON) {
      const obj = unpackJson(payload);
      post("STATUS", { status: obj });
      return;
    }
    if (type === TYPE_TEXT) {
      const text = new TextDecoder().decode(payload);
      post("TEXT", { text });
      return;
    }
    if (type === TYPE_PCM) {
      audioChunksRecv += 1;
      const sec = payload.byteLength / 2 / SAMPLE_RATE_OUT;
      ensureRunningCtx().then((ok) => {
        if (!ok) {
          post("ERROR", { error: "AudioContext not running; click Start again" });
          return;
        }
        playPcm16Bytes(payload);
        post("STATUS", {
          status: {
            phase: "audio_recv",
            chunk: audioChunksRecv,
            sec: Number(sec.toFixed(2)),
            ctx: audioCtx?.state,
          },
        });
      });
    }
  };

  ws.onclose = () => {
    post("STATUS", { status: { phase: "ws_closed", audioChunksRecv } });
  };

  const frameSamples = Math.floor((SAMPLE_RATE_IN * FRAME_MS) / 1000);
  sendTimer = setInterval(() => {
    if (!running || !ws || ws.readyState !== WebSocket.OPEN) return;
    const frame = takeFrameSamples(frameSamples);
    if (!frame) return;
    ws.send(packPcmI16(floatToInt16(frame)));
  }, FRAME_MS);

  // Keep trying to resume in case Chrome suspends later.
  setInterval(() => {
    if (running) ensureRunningCtx();
  }, 2000);

  post("STATUS", {
    status: {
      phase: "running",
      wsUrl,
      direction,
      latency,
      ctx: audioCtx.state,
      tip: "You should hear a short beep now; translation audio follows after ~L seconds",
    },
  });
}

async function stopSession() {
  running = false;
  if (sendTimer) {
    clearInterval(sendTimer);
    sendTimer = null;
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      ws.send(packEos());
    } catch (_) {}
    try {
      ws.close();
    } catch (_) {}
  }
  ws = null;
  if (workletNode) {
    try {
      workletNode.port.onmessage = null;
      workletNode.disconnect();
    } catch (_) {}
    workletNode = null;
  }
  if (sourceNode) {
    try {
      sourceNode.disconnect();
    } catch (_) {}
    sourceNode = null;
  }
  playGain = null;
  if (mediaStream) {
    for (const t of mediaStream.getTracks()) t.stop();
    mediaStream = null;
  }
  if (audioCtx) {
    try {
      await audioCtx.close();
    } catch (_) {}
    audioCtx = null;
  }
  pendingFloat = new Float32Array(0);
  post("STATUS", { status: { phase: "stopped", audioChunksRecv } });
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  (async () => {
    if (msg.type === "OFFSCREEN_START") {
      await startSession(msg);
      sendResponse({ ok: true });
      return;
    }
    if (msg.type === "OFFSCREEN_STOP") {
      await stopSession();
      sendResponse({ ok: true });
      return;
    }
    if (msg.type === "OFFSCREEN_RESUME") {
      const ok = await ensureRunningCtx();
      if (ok) playTestBeep();
      sendResponse({ ok, state: audioCtx?.state });
      return;
    }
  })().catch((err) => {
    post("ERROR", { error: String(err?.message || err) });
    sendResponse({ ok: false, error: String(err?.message || err) });
  });
  return true;
});
