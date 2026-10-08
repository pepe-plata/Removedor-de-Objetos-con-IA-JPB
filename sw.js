/* ═══════════════════════════════════════════════════════════════
   sw.js — Service Worker
   - Cache-first con fallback a red
   - Share Target (POST a ?shared=1)
   - No interceptar .onnx, jsdelivr, huggingface
   ═══════════════════════════════════════════════════════════════ */

const CACHE_NAME = 'jpb-obj-remover-v1';
const SHARED_CACHE = 'jpb-shared';

// Archivos a precachear
const PRECACHE = [
  './',
  './index.html',
  './styles.css',
  './app.js',
  './db.js',
  './model-worker.js',
  './manifest.json',
  './icons/icon-64.png',
  './icons/icon-192.png',
  './icons/icon-512.png'
];

// Install: precachear
self.addEventListener('install', (ev) => {
  ev.waitUntil(
    caches.open(CACHE_NAME).then((c) => c.addAll(PRECACHE)).catch(() => {})
  );
  self.skipWaiting();
});

// Activate: limpiar caches viejos
self.addEventListener('activate', (ev) => {
  ev.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter(k => k !== CACHE_NAME && k !== SHARED_CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch: cache-first, excepto dominios/extensiones excluidas
self.addEventListener('fetch', (ev) => {
  const req = ev.request;
  const url = new URL(req.url);

  // ─── Share Target: POST a ?shared=1 ───
  if (req.method === 'POST' && url.searchParams.get('shared') === '1') {
    ev.respondWith(handleShareTarget(req));
    return;
  }

  // No interceptar ONNX ni CDNs externos
  if (url.pathname.endsWith('.onnx') ||
      url.hostname.includes('cdn.jsdelivr.net') ||
      url.hostname.includes('huggingface.co') ||
      url.hostname.includes('github.com') ||
      url.hostname.includes('githubusercontent.com')) {
    return; // deja que vaya directo a red
  }

  // Solo GET, mismo origen
  if (req.method !== 'GET') return;

  ev.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req).then((resp) => {
        // Cachear respuestas válidas del mismo origen
        if (resp.ok && url.origin === location.origin) {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then((c) => c.put(req, clone)).catch(() => {});
        }
        return resp;
      }).catch(() => cached);
    })
  );
});

/* ─── Handler Share Target ─── */
async function handleShareTarget(req) {
  try {
    const formData = await req.formData();
    const files = formData.getAll('files') || formData.getAll('file') || formData.getAll('image');

    if (!files.length) {
      return Response.redirect('./?shared=0', 303);
    }

    // Convertir a base64 para guardar en cache
    const serialized = [];
    for (const f of files) {
      if (typeof f === 'string') continue;
      const buf = await f.arrayBuffer();
      const bytes = new Uint8Array(buf);
      let bin = '';
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {
        bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
      }
      serialized.push({
        name: f.name || 'imagen',
        type: f.type || 'image/png',
        data: btoa(bin)
      });
    }

    // Guardar en cache temporal
    const cache = await caches.open(SHARED_CACHE);
    await cache.put(
      'shared-files',
      new Response(JSON.stringify({ files: serialized }), {
        headers: { 'Content-Type': 'application/json' }
      })
    );

    // Redirigir a la app
    return Response.redirect('./?shared=1', 303);
  } catch (e) {
    console.error('Share target error', e);
    return Response.redirect('./?shared=0', 303);
  }
}

// Mensajes desde la app
self.addEventListener('message', (ev) => {
  if (ev.data === 'skipWaiting') self.skipWaiting();
});