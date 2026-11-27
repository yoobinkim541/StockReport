(() => {
  const fetchOriginal = window.fetch.bind(window);
  const paused = new Promise(() => {});
  let expired = false;
  let failures = 0;
  let timer;
  let probing = false;

  function showExpired() {
    const dialog = document.createElement("dialog");
    dialog.lang = "ko";
    dialog.setAttribute("aria-labelledby", "tn-session-title");
    dialog.style.cssText = "box-sizing:border-box;width:min(90%,26rem);padding:1.5rem;"
      + "border:1px solid GrayText;border-radius:1rem;color-scheme:light dark;"
      + "background:Canvas;color:CanvasText;font:1rem/1.5 system-ui,sans-serif;";
    dialog.innerHTML = '<h1 id="tn-session-title" style="font-size:1.25rem;margin:0">로그인이 필요합니다</h1>'
      + '<p>접근 인증이 만료되었거나 거부되어 자동 재접속을 중단했습니다.</p>'
      + '<button type="button" autofocus style="padding:.65rem 1rem;font:inherit">다시 로그인</button>';
    dialog.addEventListener("cancel", (event) => event.preventDefault());
    dialog.querySelector("button").addEventListener("click", () => window.location.reload());
    document.body.append(dialog);
    dialog.showModal();
  }

  async function connectionFetch(input, options) {
    // Do not follow Access's cross-origin login redirect on a background request.
    const response = await fetchOriginal(input, {...options, redirect: "manual", cache: "no-store"});
    if (!expired && (response.type === "opaqueredirect" || response.status === 401 || response.status === 403)) {
      expired = true;
      clearTimeout(timer);
      if (document.body) showExpired();
      else document.addEventListener("DOMContentLoaded", showExpired, {once: true});
    }
    return response;
  }

  window.fetch = async (input, options) => {
    const request = input instanceof Request ? input : null;
    const method = options?.method ?? request?.method ?? "GET";
    if (method.toUpperCase() === "GET") {
      const url = new URL(request?.url ?? input, document.baseURI);
      if (url.origin === window.location.origin && /\/_stcore\/(health|host-config)$/.test(url.pathname)) {
        if (expired) return paused;
        const response = await connectionFetch(input, options);
        // Streamlit retries both failed pings and rejected fetches; park its ping
        // until the user discards this document, without faking a success.
        return expired ? paused : response;
      }
    }
    return fetchOriginal(input, options);
  };

  async function probe() {
    if (expired || document.hidden || probing) return;
    probing = true;
    const controller = new AbortController();
    const deadline = setTimeout(() => controller.abort(), 2500);
    try {
      const response = await connectionFetch("/_stcore/health", {signal: controller.signal});
      if (expired) return;
      if (response.ok) {
        if (failures >= 3) {
          if (!document.hidden) {
            window.location.reload();
            return;
          }
        } else {
          failures = 0;
        }
      } else {
        failures++;
      }
    } catch {
      failures++;
    } finally {
      clearTimeout(deadline);
      probing = false;
    }
    if (!document.hidden && !expired) timer = setTimeout(probe, 15000);
  }

  document.addEventListener("visibilitychange", () => {
    clearTimeout(timer);
    if (!document.hidden) void probe();
  });
  if (!document.hidden) timer = setTimeout(probe, 15000);
})();
