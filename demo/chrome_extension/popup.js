const $ = (id) => document.getElementById(id);

async function loadSettings() {
  const st = await chrome.storage.local.get({
    wsUrl: "ws://127.0.0.1:8765",
    direction: "en2zh",
    latency: "2",
  });
  $("wsUrl").value = st.wsUrl;
  $("direction").value = st.direction;
  $("latency").value = String(st.latency);
}

async function saveSettings() {
  await chrome.storage.local.set({
    wsUrl: $("wsUrl").value.trim(),
    direction: $("direction").value,
    latency: $("latency").value,
  });
}

function setRunning(running) {
  $("start").disabled = running;
  $("stop").disabled = !running;
}

function setStatus(text) {
  $("status").textContent = text;
}

const CAPTION_TTL_MS = 20000;
/** @type {{ text: string, t: number }[]} */
let captionSegments = [];
let captionPruneTimer = null;

function renderCaptions() {
  const el = $("captions");
  const cutoff = Date.now() - CAPTION_TTL_MS;
  captionSegments = captionSegments.filter((s) => s.t >= cutoff);
  el.textContent = captionSegments.length
    ? captionSegments.map((s) => s.text).join("")
    : "(captions — also shown as a floating bar on the tab)";
  el.scrollTop = el.scrollHeight;
  if (captionSegments.length === 0 && captionPruneTimer) {
    clearInterval(captionPruneTimer);
    captionPruneTimer = null;
  }
}

function appendCaption(text) {
  if (!text) return;
  captionSegments.push({ text, t: Date.now() });
  if (!captionPruneTimer) {
    captionPruneTimer = setInterval(renderCaptions, 250);
  }
  renderCaptions();
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "STATUS") {
    setStatus(JSON.stringify(msg.status, null, 0));
    if (msg.status?.phase === "stopped" || msg.status?.phase === "ws_closed") {
      setRunning(false);
    }
    if (msg.status?.phase === "running") setRunning(true);
  }
  if (msg.type === "TEXT" && msg.text) appendCaption(msg.text);
  if (msg.type === "ERROR") {
    setStatus(`ERROR: ${msg.error}`);
    setRunning(false);
  }
});

$("start").addEventListener("click", async () => {
  await saveSettings();
  captionSegments = [];
  if (captionPruneTimer) {
    clearInterval(captionPruneTimer);
    captionPruneTimer = null;
  }
  $("captions").textContent = "(captions — also shown as a floating bar on the tab)";
  setStatus("Starting… (expect a short beep)");
  setRunning(true);
  const res = await chrome.runtime.sendMessage({
    type: "START",
    wsUrl: $("wsUrl").value.trim(),
    direction: $("direction").value,
    latency: Number($("latency").value),
    // muteTab=true => translation only; false => also hear original quietly
    muteTab: !$("hearTab").checked,
  });
  if (!res?.ok) {
    setStatus(`Failed: ${res?.error || "unknown"}`);
    setRunning(false);
    return;
  }
  setStatus(`Running on tab ${res.tabId}. Keep this popup open a moment; wait for audio_recv.`);
});

$("stop").addEventListener("click", async () => {
  setStatus("Stopping…");
  const res = await chrome.runtime.sendMessage({ type: "STOP" });
  setStatus(res?.ok ? "Stopped" : `Stop failed: ${res?.error}`);
  setRunning(false);
});

$("beep").addEventListener("click", async () => {
  const res = await chrome.runtime.sendMessage({ type: "RESUME_AUDIO" });
  setStatus(res?.ok ? `Audio resumed (${res.state}); beep sent` : `Resume failed: ${res?.error}`);
});

(async () => {
  await loadSettings();
  const st = await chrome.runtime.sendMessage({ type: "GET_STATUS" });
  setRunning(!!st?.running);
  setStatus(st?.running ? `Running on tab ${st.tabId}` : "Idle");
})();
