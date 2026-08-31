/**
 * Floating caption bar at the bottom of the active tab.
 * Injected on Start; receives CAPTION_* messages from the service worker.
 * Caption segments older than CAPTION_TTL_MS are dropped.
 */
(() => {
  const HOST_ID = "omnitalker-s2st-caption-host";
  // Keep recent lines longer so the floating bar stays readable while watching.
  const CAPTION_TTL_MS = 20000;

  function mount() {
    let host = document.getElementById(HOST_ID);
    if (!host) {
      host = document.createElement("div");
      host.id = HOST_ID;
      document.documentElement.appendChild(host);
    }

    // Re-inject after extension reload: rebuild shadow so new TTL logic applies.
    if (host.shadowRoot) {
      host.innerHTML = "";
      // Detach old shadow by replacing the host node.
      const next = document.createElement("div");
      next.id = HOST_ID;
      host.replaceWith(next);
      host = next;
    }

    const shadow = host.attachShadow({ mode: "open" });
    shadow.innerHTML = `
      <style>
        :host { all: initial; }
        #bar {
          position: fixed;
          left: 50%;
          bottom: 48px;
          transform: translateX(-50%);
          z-index: 2147483646;
          width: min(920px, calc(100vw - 48px));
          max-height: 28vh;
          overflow: hidden;
          pointer-events: none;
          font-family: "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif;
        }
        #bar.hidden { display: none; }
        #panel {
          background: rgba(12, 12, 14, 0.78);
          color: #f5f5f5;
          border-radius: 12px;
          padding: 12px 18px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
          backdrop-filter: blur(8px);
          -webkit-backdrop-filter: blur(8px);
          border: 1px solid rgba(255, 255, 255, 0.12);
        }
        #text {
          font-size: 20px;
          line-height: 1.45;
          letter-spacing: 0.01em;
          text-align: center;
          white-space: pre-wrap;
          word-break: break-word;
          max-height: calc(28vh - 24px);
          overflow-y: auto;
          text-shadow: 0 1px 2px rgba(0, 0, 0, 0.55);
        }
        #text:empty::before {
          content: "…";
          opacity: 0.35;
        }
      </style>
      <div id="bar">
        <div id="panel"><div id="text"></div></div>
      </div>
    `;

    const bar = shadow.getElementById("bar");
    const textEl = shadow.getElementById("text");
    /** @type {{ text: string, t: number }[]} */
    let segments = [];
    let pruneTimer = null;

    function show() {
      bar.classList.remove("hidden");
    }

    function hide() {
      bar.classList.add("hidden");
    }

    function render() {
      const cutoff = Date.now() - CAPTION_TTL_MS;
      segments = segments.filter((s) => s.t >= cutoff);
      textEl.textContent = segments.map((s) => s.text).join("");
      textEl.scrollTop = textEl.scrollHeight;
      if (segments.length === 0 && pruneTimer) {
        clearInterval(pruneTimer);
        pruneTimer = null;
      }
    }

    function ensurePruneTimer() {
      if (pruneTimer) return;
      pruneTimer = setInterval(render, 250);
    }

    function clear() {
      segments = [];
      textEl.textContent = "";
      if (pruneTimer) {
        clearInterval(pruneTimer);
        pruneTimer = null;
      }
    }

    function append(chunk) {
      if (!chunk) return;
      show();
      segments.push({ text: chunk, t: Date.now() });
      ensurePruneTimer();
      render();
    }

    if (window.__omnitalkerCaptionListener) {
      chrome.runtime.onMessage.removeListener(window.__omnitalkerCaptionListener);
    }

    const listener = (msg, _sender, sendResponse) => {
      if (msg?.type === "CAPTION_APPEND") {
        append(msg.text || "");
        sendResponse({ ok: true });
        return true;
      }
      if (msg?.type === "CAPTION_CLEAR") {
        clear();
        show();
        sendResponse({ ok: true });
        return true;
      }
      if (msg?.type === "CAPTION_SHOW") {
        show();
        sendResponse({ ok: true });
        return true;
      }
      if (msg?.type === "CAPTION_HIDE") {
        hide();
        sendResponse({ ok: true });
        return true;
      }
      return false;
    };
    window.__omnitalkerCaptionListener = listener;
    chrome.runtime.onMessage.addListener(listener);

    window.__omnitalkerCaptionApi = { show, hide, clear, append };
    window.__omnitalkerCaptionMounted = true;
    show();
  }

  mount();
})();
