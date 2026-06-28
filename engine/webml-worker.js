// Module worker that runs a webml-community engine (Gemma4Mobile / Lfm2Mobile) on
// a dedicated thread — the way LocalMind drives them. On the main thread these
// fast engines lose ~2× because their per-token control code competes with the
// page event loop; in a worker they dispatch to the GPU unimpeded. tok/s is
// measured here (at the source) so postMessage overhead doesn't taint it.
//
// Robustness: a lost WebGPU device (GPU process killed by contention, or a prior
// device destroyed) otherwise makes load()/generate() hang FOREVER with no signal
// — the lane just sits at "weights 100%". We make that fail fast two ways:
//   1. a probe device whose `.lost` promise we surface as a `device-lost` message
//      (the only reliable async signal that the shared GPU process died), and
//   2. hard timeouts on load and on each generation step, so a wedged GPU rejects
//      with a clear message instead of hanging the lane.
let engine = null;
let probeDevice = null;

const LOAD_TIMEOUT_MS = 120000;   // weights + first-run pipeline compile (generous; contended Metal is slow)
const FIRST_TOKEN_MS  = 45000;    // prefill + lazy pipeline compile on the first token
const STEP_TIMEOUT_MS = 15000;    // a healthy decode step is << 1s; 15s of silence = wedged

function withTimeout(promise, ms, label) {
  let t;
  const timeout = new Promise((_, rej) => { t = setTimeout(() => rej(new Error(label + ' (' + (ms / 1000) + 's)')), ms); });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(t));
}

// A probe device shares the GPU process with the engine's device. When that process
// dies (contention OOM, a prior destroyed device), the probe's `.lost` promise
// resolves — our one reliable async signal that the engine is wedged. A `destroyed`
// reason is what our own dispose() raises, so we ignore that and report only real loss.
async function watchDeviceLoss() {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    probeDevice = await adapter.requestDevice();
    probeDevice.lost.then((info) => {
      if (info && info.reason !== 'destroyed') {
        self.postMessage({ type: 'device-lost', reason: info.reason || 'unknown', message: info.message || '' });
      }
    });
  } catch { /* probe is best-effort; timeouts below are the backstop */ }
}

self.onmessage = async (e) => {
  const d = e.data;
  try {
    if (d.cmd === 'load') {
      await watchDeviceLoss();
      const mod = await import(d.file);
      const Mobile = mod[d.cls];
      engine = await withTimeout(Mobile.load(null, {
        onProgress: (ev) => {
          if (ev && ev.status === 'weights' && ev.total) {
            self.postMessage({ type: 'progress', pct: (100 * ev.loaded) / ev.total });
          }
        },
      }), LOAD_TIMEOUT_MS, 'engine load timed out — GPU likely wedged');
      self.postMessage({ type: 'loaded' });
    } else if (d.cmd === 'run') {
      const msgs = [{ role: 'user', content: d.prompt }];
      let n = 0, firstMs = 0, prev = '';
      const t0 = performance.now();
      // Drive the async iterator by hand so each step gets a watchdog timeout — a
      // `for await` would have no way to bail a step that never settles.
      const it = engine.generate(msgs, { maxNewTokens: d.maxTok })[Symbol.asyncIterator]();
      for (;;) {
        const res = await withTimeout(
          it.next(),
          n === 0 ? FIRST_TOKEN_MS : STEP_TIMEOUT_MS,
          n === 0 ? 'no first token — GPU likely wedged' : 'generation stalled — GPU likely lost',
        );
        if (res.done) break;
        const out = res.value;
        const full = out && typeof out.text === 'string' ? out.text : '';
        if (full.length > prev.length) {
          if (!firstMs) firstMs = performance.now() - t0;
          n++;
          self.postMessage({ type: 'token', delta: full.slice(prev.length) });  // delta, not cumulative
          prev = full;
        }
      }
      const ms = performance.now() - t0;
      const tps = n > 1 ? (n - 1) / ((ms - firstMs) / 1000) : n / (ms / 1000);
      self.postMessage({ type: 'done', tps, ttft: firstMs / 1000, n });
    }
  } catch (err) {
    self.postMessage({ type: 'error', error: String((err && err.message) || err) });
  }
};
