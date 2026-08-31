/**
 * Service worker: start/stop tab capture session via offscreen document.
 */

const OFFSCREEN_URL = "offscreen.html";

async function hasOffscreen() {
  if (chrome.runtime.getContexts) {
    const existing = await chrome.runtime.getContexts({
      contextTypes: ["OFFSCREEN_DOCUMENT"],
      documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)],
    });
    return existing.length > 0;
  }
  // Older Chrome fallback
  const clients = await self.clients.matchAll({ type: "window" });
  const url = chrome.runtime.getURL(OFFSCREEN_URL);
  return clients.some((c) => c.url === url);
}

async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ["USER_MEDIA", "AUDIO_PLAYBACK"],
    justification: "Capture tab audio and play OmniTalker translation output",
  });
}

async function closeOffscreen() {
  if (!(await hasOffscreen())) return;
  await chrome.offscreen.closeDocument();
}

async function ensureCaptionOverlay(tabId) {
  await chrome.scripting.executeScript({
    target: { tabId },
    files: ["content_overlay.js"],
  });
  try {
    await chrome.tabs.sendMessage(tabId, { type: "CAPTION_CLEAR" });
    await chrome.tabs.sendMessage(tabId, { type: "CAPTION_SHOW" });
  } catch (_) {
    /* first inject: listener may not be ready for a tick */
    await new Promise((r) => setTimeout(r, 50));
    await chrome.tabs.sendMessage(tabId, { type: "CAPTION_CLEAR" });
    await chrome.tabs.sendMessage(tabId, { type: "CAPTION_SHOW" });
  }
}

async function forwardCaption(tabId, text) {
  try {
    await chrome.tabs.sendMessage(tabId, { type: "CAPTION_APPEND", text });
  } catch (_) {
    await ensureCaptionOverlay(tabId);
    await chrome.tabs.sendMessage(tabId, { type: "CAPTION_APPEND", text });
  }
}

async function hideCaptionOverlay(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { type: "CAPTION_HIDE" });
  } catch (_) {
    /* tab gone or not injected */
  }
}

/** Release any leftover tabCapture / offscreen stream before a new Start. */
async function forceReleaseCapture() {
  const st = await chrome.storage.session.get(["tabId"]);
  try {
    if (await hasOffscreen()) {
      await chrome.runtime.sendMessage({ type: "OFFSCREEN_STOP" });
    }
  } catch (_) {
    /* ignore */
  }
  await closeOffscreen();
  if (st.tabId != null) {
    await hideCaptionOverlay(st.tabId);
  }
  await chrome.storage.session.set({ running: false });
  // Give Chrome a beat to drop the previous MediaStream on the tab.
  await new Promise((r) => setTimeout(r, 200));
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  // Offscreen-targeted messages: do not claim async response here.
  if (
    msg.type === "OFFSCREEN_START" ||
    msg.type === "OFFSCREEN_STOP" ||
    msg.type === "OFFSCREEN_RESUME"
  ) {
    return false;
  }

  (async () => {
    if (msg.type === "START") {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id) throw new Error("No active tab");
      if (tab.url?.startsWith("chrome://") || tab.url?.startsWith("chrome-extension://")) {
        throw new Error("Cannot capture chrome:// pages; open a normal site (e.g. Bilibili)");
      }

      // Chrome allows only one tabCapture stream per tab; clear leftovers first.
      await forceReleaseCapture();

      await ensureOffscreen();
      // Small delay so offscreen listeners attach.
      await new Promise((r) => setTimeout(r, 100));

      let streamId;
      try {
        streamId = await chrome.tabCapture.getMediaStreamId({
          targetTabId: tab.id,
        });
      } catch (err) {
        const detail = String(err?.message || err);
        throw new Error(
          `${detail}. Click Stop, or reload this tab / the extension, then Start again.`
        );
      }

      const result = await chrome.runtime.sendMessage({
        type: "OFFSCREEN_START",
        streamId,
        wsUrl: msg.wsUrl,
        direction: msg.direction,
        latency: msg.latency,
        muteTab: !!msg.muteTab,
      });
      if (!result?.ok) {
        await closeOffscreen();
        throw new Error(result?.error || "offscreen start failed");
      }
      await chrome.storage.session.set({ running: true, tabId: tab.id });
      await ensureCaptionOverlay(tab.id);
      sendResponse({ ok: true, tabId: tab.id });
      return;
    }

    if (msg.type === "STOP") {
      const st = await chrome.storage.session.get(["tabId"]);
      try {
        await chrome.runtime.sendMessage({ type: "OFFSCREEN_STOP" });
      } catch (_) {
        /* offscreen may already be gone */
      }
      await closeOffscreen();
      if (st.tabId != null) {
        await hideCaptionOverlay(st.tabId);
      }
      await chrome.storage.session.set({ running: false });
      sendResponse({ ok: true });
      return;
    }

    if (msg.type === "GET_STATUS") {
      const st = await chrome.storage.session.get(["running", "tabId"]);
      sendResponse({ ok: true, running: !!st.running, tabId: st.tabId ?? null });
      return;
    }

    if (msg.type === "RESUME_AUDIO") {
      if (!(await hasOffscreen())) {
        sendResponse({ ok: false, error: "not running — press Start first" });
        return;
      }
      const result = await chrome.runtime.sendMessage({ type: "OFFSCREEN_RESUME" });
      sendResponse(result || { ok: false, error: "no response" });
      return;
    }

    // STATUS / TEXT / ERROR from offscreen — popup also listens; forward captions to tab.
    if (msg.type === "STATUS" || msg.type === "TEXT" || msg.type === "ERROR") {
      if (msg.type === "STATUS" && (msg.status?.phase === "stopped" || msg.status?.phase === "ws_closed")) {
        const st = await chrome.storage.session.get(["tabId"]);
        await chrome.storage.session.set({ running: false });
        if (st.tabId != null) {
          await hideCaptionOverlay(st.tabId);
        }
      }
      if (msg.type === "TEXT" && msg.text) {
        const st = await chrome.storage.session.get(["tabId"]);
        if (st.tabId != null) {
          await forwardCaption(st.tabId, msg.text);
        }
      }
      sendResponse({ ok: true });
      return;
    }

    sendResponse({ ok: false, error: `unknown message ${msg.type}` });
  })().catch((err) => {
    sendResponse({ ok: false, error: String(err?.message || err) });
  });
  return true;
});
