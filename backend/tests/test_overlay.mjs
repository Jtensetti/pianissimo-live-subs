// Development-only: node --test backend/tests/test_overlay.mjs
// Runs the real inline script with a deterministic DOM/timer/WebSocket harness.
// Layout uses a simulated measured height; this does not replace a browser check.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const html = readFileSync(new URL("../overlay.html", import.meta.url), "utf8");
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function overlay(columns = 100) {
  let now = 100000;
  let timerId = 0;
  const timers = new Map();
  const sockets = [];
  const resizeObservers = [];
  const listeners = new Map();
  const caption = {
    hidden: true,
    textContent: "",
    get scrollHeight() { return Math.ceil(this.textContent.length / columns) * 20; },
  };
  class Socket {
    constructor(url) { this.url = url; sockets.push(this); }
    close() { this.onclose?.(); }
  }
  class Observer {
    constructor(callback) { this.callback = callback; resizeObservers.push(this); }
    observe() {}
    disconnect() {}
  }
  vm.runInNewContext(source, {
    document: { getElementById: () => caption, documentElement: {} },
    window: { addEventListener: (name, callback) => listeners.set(name, callback) },
    location: { protocol: "http:", host: "localhost:8000" },
    WebSocket: Socket,
    ResizeObserver: Observer,
    getComputedStyle: () => ({ lineHeight: "20px" }),
    Date: class extends Date { static now() { return now; } },
    setTimeout: (callback, delay) => {
      timers.set(++timerId, { callback, at: now + delay });
      return timerId;
    },
    clearTimeout: (id) => timers.delete(id),
  });
  function message(event) { sockets.at(-1).onmessage({ data: JSON.stringify(event) }); }
  function say(id, text, mode = "instant", lifetime = 8000) {
    message({ type: "transcription", mode, data: { id, text }, expires_in_ms: lifetime });
  }
  function advance(milliseconds) {
    const end = now + milliseconds;
    for (let iteration = 0; iteration < 1000; iteration++) {
      const pending = [...timers].filter(([, timer]) => timer.at <= end).sort((a, b) => a[1].at - b[1].at);
      if (!pending.length) { now = end; return; }
      const [id, timer] = pending[0];
      now = timer.at;
      timers.delete(id);
      timer.callback();
    }
    throw new Error("Timer failed to advance");
  }
  return { caption, say, message, advance, sockets, listeners,
    resize: (newColumns) => { columns = newColumns; resizeObservers.forEach((observer) => observer.callback()); } };
}

test("a new partial keeps the previous final visible and appears without a display delay", () => {
  const view = overlay();
  view.say("one", "Vi bygger svensk livetext.", "final");
  view.say("two", "Nu testar");
  assert.equal(view.caption.textContent, "Vi bygger svensk livetext. Nu testar");
  view.say("two", "Nu testar vi mikrofonen.");
  assert.equal(view.caption.textContent, "Vi bygger svensk livetext. Nu testar vi mikrofonen.");
  view.say("two", "Nu testar vi mikrofonen!", "final");
  assert.equal(view.caption.textContent, "Vi bygger svensk livetext. Nu testar vi mikrofonen!");
});

test("a correction to the preceding segment replaces in place without changing chronological order", () => {
  const view = overlay();
  view.say("one", "Det var rött.", "final");
  view.say("two", "Sedan kom nästa");
  view.say("one", "Det var rätt.", "final");
  assert.equal(view.caption.textContent, "Det var rätt. Sedan kom nästa");
  view.say("three", "Och sist detta.");
  assert.equal(view.caption.textContent, "Sedan kom nästa Och sist detta.");
});

test("an empty final retracts only its own partial; an empty partial does not erase text", () => {
  const view = overlay();
  view.say("one", "Den färdiga frasen.", "final");
  view.say("two", "Brus blev ord");
  view.say("two", "");
  assert.equal(view.caption.textContent, "Den färdiga frasen. Brus blev ord");
  view.say("two", "   ", "final");
  assert.equal(view.caption.textContent, "Den färdiga frasen.");
  assert.equal(view.caption.hidden, false);
  view.say("other", "", "final");
  assert.equal(view.caption.textContent, "Den färdiga frasen.");
});

test("segment expiry is independent and new partials do not keep old finals alive", () => {
  const view = overlay();
  view.say("one", "Förra frasen.", "final", 2000);
  view.advance(1000);
  view.say("two", "Aktuell fras.", "instant", 8000);
  view.advance(1000);
  assert.equal(view.caption.textContent, "Aktuell fras.");
  view.advance(6999);
  assert.equal(view.caption.hidden, false);
  view.advance(1);
  assert.equal(view.caption.hidden, true);
});

test("an expired snapshot removes only its ID and reconnect snapshots retain both IDs", () => {
  const view = overlay();
  view.say("one", "Tidigare slut.", "final", 1000);
  view.say("two", "Färska ord", "instant", 6000);
  assert.equal(view.caption.textContent, "Tidigare slut. Färska ord");
  view.say("one", "Tidigare slut.", "final", 0);
  assert.equal(view.caption.textContent, "Färska ord");
});

test("fit retains the newest words within two measured lines and refits on resize", () => {
  const view = overlay(25);
  view.say("one", "En ganska lång föregående mening som redan har avslutats.", "final");
  view.say("two", "Och här kommer nya orden som ska synas allra sist.");
  assert.ok(view.caption.textContent.startsWith("… "));
  assert.ok(view.caption.textContent.endsWith("synas allra sist."));
  assert.ok(view.caption.scrollHeight <= 40);
  view.resize(13);
  assert.ok(view.caption.textContent.endsWith("allra sist."));
  assert.ok(view.caption.scrollHeight <= 40);
});

test("even a single oversized word retains its ending within two lines", () => {
  const view = overlay(12);
  view.say("one", "väldigtlångtsammansattsvensktordmedslut");
  assert.ok(view.caption.textContent.endsWith("medslut"));
  assert.ok(view.caption.scrollHeight <= 40);
});

test("clear and disconnect immediately remove all context; reconnect does not revive it", () => {
  const view = overlay();
  view.say("one", "Första frasen.", "final");
  view.say("two", "Andra frasen.");
  view.message({ type: "clear" });
  assert.equal(view.caption.textContent, "");
  assert.equal(view.caption.hidden, true);
  view.say("three", "Bara ny text.");
  assert.equal(view.caption.textContent, "Bara ny text.");
  view.sockets[0].close();
  assert.equal(view.caption.hidden, true);
  view.advance(1000);
  assert.equal(view.sockets.length, 2);
  assert.equal(view.sockets[1].url, "ws://localhost:8000/ws/subtitles");
  assert.equal(view.caption.textContent, "");
  view.say("four", "Efter återanslutning.");
  assert.equal(view.caption.textContent, "Efter återanslutning.");
});
