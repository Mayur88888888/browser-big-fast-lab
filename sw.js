// browserlab model cache — intercepts Hugging Face (+ jsDelivr) GETs and serves
// them from CacheStorage on re-load, so models don't re-download between loads.
// Helps every lane uniformly: transformers.js / raw-ORT (full ONNX files), our
// engine (ranged GGUF), webml (its own ranged GGUF). Range (206) responses are
// stored as 200 because the Cache API rejects 206; the 206 status + Content-Range
// are reconstructed on a hit, keyed by url+range so deterministic ranges hit.
const CACHE = 'browserlab-models-v1';
const HOSTS = ['huggingface.co', 'hf.co', 'cdn.jsdelivr.net'];

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  let url;
  try { url = new URL(req.url); } catch { return; }
  if (!HOSTS.some((h) => url.hostname === h || url.hostname.endsWith('.' + h) || url.hostname.includes(h))) return;
  event.respondWith(serve(req));
});

async function serve(req) {
  const range = req.headers.get('range') || '';
  let cache;
  try { cache = await caches.open(CACHE); } catch { return fetch(req); }
  const key = new Request(req.url + '\n' + range);

  const hit = await cache.match(key);
  if (hit) {
    return new Response(hit.body, {
      status: range ? 206 : 200,
      statusText: range ? 'Partial Content' : 'OK',
      headers: hit.headers,
    });
  }

  let resp;
  try { resp = await fetch(req); } catch { return new Response('offline (not cached)', { status: 504 }); }
  if (resp && (resp.status === 200 || resp.status === 206) && resp.body) {
    try {
      const [a, b] = resp.body.tee();
      // Store as 200 (Cache API forbids caching 206); status is reconstructed on hit.
      cache.put(key, new Response(a, { status: 200, headers: resp.headers })).catch(() => {});
      return new Response(b, { status: resp.status, statusText: resp.statusText, headers: resp.headers });
    } catch {
      return resp;
    }
  }
  return resp;
}
