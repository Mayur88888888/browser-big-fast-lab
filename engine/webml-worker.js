// Module worker that runs a webml-community engine (Gemma4Mobile / Lfm2Mobile) on
// a dedicated thread — the way LocalMind drives them. On the main thread these
// fast engines lose ~2× because their per-token control code competes with the
// page event loop; in a worker they dispatch to the GPU unimpeded. tok/s is
// measured here (at the source) so postMessage overhead doesn't taint it.
let engine = null;

self.onmessage = async (e) => {
  const d = e.data;
  try {
    if (d.cmd === 'load') {
      const mod = await import(d.file);
      const Mobile = mod[d.cls];
      engine = await Mobile.load(null, {
        onProgress: (ev) => {
          if (ev && ev.status === 'weights' && ev.total) {
            self.postMessage({ type: 'progress', pct: (100 * ev.loaded) / ev.total });
          }
        },
      });
      self.postMessage({ type: 'loaded' });
    } else if (d.cmd === 'run') {
      const msgs = [{ role: 'user', content: d.prompt }];
      let n = 0, firstMs = 0, prev = '';
      const t0 = performance.now();
      const stream = engine.generate(msgs, { maxNewTokens: d.maxTok });
      for await (const out of stream) {
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
