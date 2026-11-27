"""Connection retries should stop on Access denial, not on a transient outage."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed for browser script tests")
def test_reconnect_guard():
    guard = Path(__file__).resolve().parents[1] / "dashboard" / "reconnect.js"
    result = subprocess.run(["node", "-e", _HARNESS, str(guard)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


_HARNESS = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const script = fs.readFileSync(process.argv[1], "utf8");
const root = "https://terminal.example/";
const health = "/_stcore/health";
const config = "/_stcore/host-config";
const ok = {ok: true, status: 200};
const unavailable = {ok: false, status: 503};

function browser() {
  const timers = new Map(), events = {}, calls = [], outcomes = [];
  let id = 0, dialogs = 0, reloads = 0, click;
  const window = {
    location: {origin: new URL(root).origin, reload() { reloads++; }},
    async fetch(url, options) {
      calls.push([url, options?.redirect]);
      const outcome = outcomes.shift() ?? ok;
      if (outcome instanceof Error) throw outcome;
      return outcome;
    },
  };
  const document = {
    baseURI: root + "portfolio?tab=positions",
    body: {append() {}}, hidden: false,
    createElement() {
      return {style: {}, setAttribute() {}, addEventListener() {}, showModal() { dialogs++; },
              querySelector() { return {addEventListener(_, fn) { click = fn; }}; }};
    },
    addEventListener(event, fn) { events[event] = fn; },
  };
  vm.runInNewContext(script, {window, document, Request, URL, AbortController,
    setTimeout(fn, ms) { timers.set(++id, [fn, ms]); return id; },
    clearTimeout(id) { timers.delete(id); },
  });
  return {window, calls, outcomes, timers,
    get dialogs() { return dialogs; }, get reloads() { return reloads; },
    click() { click(); },
    setHidden(hidden) { document.hidden = hidden; events.visibilitychange(); },
    async tick() {
      const entry = [...timers].find(([_, [, ms]]) => ms === 15000);
      assert.ok(entry, "expected one scheduled health probe");
      timers.delete(entry[0]);
      await entry[1][0]();
    },
  };
}

async function check() {
  for (const denied of [{type: "opaqueredirect", status: 0}, {status: 401}, {status: 403}]) {
    const b = browser();
    b.outcomes.push(denied, ok);
    let settled = false;
    void Promise.all([b.window.fetch(health), b.window.fetch(config)]).then(() => { settled = true; });
    await new Promise(setImmediate);
    assert.deepEqual(b.calls, [[health, "manual"], [config, "manual"]]);
    assert.equal(b.dialogs, 1);
    assert.equal(b.reloads, 0);
    assert.equal(settled, false);
    assert.equal(b.timers.size, 0);
    b.setHidden(true);
    b.setHidden(false);
    void b.window.fetch(new Request(root + "_stcore/health"));
    void b.window.fetch(config);
    await Promise.resolve();
    assert.equal(b.calls.length, 2, "paused probes must never hit the network again");
    await b.window.fetch("https://other.example/_stcore/health");
    assert.deepEqual(b.calls.at(-1), ["https://other.example/_stcore/health", undefined]);
    b.click();
    assert.equal(b.reloads, 1);
  }

  const b = browser();
  b.outcomes.push(unavailable, ok);
  assert.equal(await b.window.fetch(config), unavailable);
  assert.equal(await b.window.fetch(config), ok);
  assert.equal(b.dialogs, 0);
  b.setHidden(true);
  assert.equal(b.timers.size, 0, "no background watchdog poll");
  b.setHidden(false);
  await new Promise(setImmediate);
  assert.deepEqual(b.calls.at(-1), [health, "manual"]);
  b.outcomes.push(new TypeError("network reset"), unavailable, ok);
  for (let i = 0; i < 3; i++) await b.tick();
  assert.equal(b.reloads, 0, "two transient failures must not reload");
  b.outcomes.push(unavailable, unavailable, unavailable, ok);
  for (let i = 0; i < 4; i++) await b.tick();
  assert.equal(b.reloads, 1);
  assert.equal(b.dialogs, 0);
  assert.equal(b.timers.size, 0);

  const c = browser();
  c.outcomes.push({type: "opaqueredirect", status: 0});
  await c.tick();
  assert.equal(c.dialogs, 1, "expiry is detected even if the original socket stays open");
  assert.deepEqual(c.calls, [[health, "manual"]]);
  assert.equal(c.timers.size, 0);
  assert.equal(c.reloads, 0);
}
check().catch((error) => { console.error(error); process.exitCode = 1; });
"""
