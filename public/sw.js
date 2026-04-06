const CACHE_NAME = 'letter-constellations-v1';

const getAppRoot = () => self.registration.scope;

const getAppShellUrls = () => {
  const root = getAppRoot();
  return [
    root,
    new URL('index.html', root).toString(),
    new URL('manifest.webmanifest', root).toString(),
    new URL('icon.png', root).toString(),
    new URL('apple-touch-icon.png', root).toString(),
    new URL('pwa-192x192.png', root).toString(),
    new URL('pwa-512x512.png', root).toString(),
    new URL('pwa-512x512-maskable.png', root).toString(),
  ];
};

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      await cache.addAll(getAppShellUrls());
      await self.skipWaiting();
    })()
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const cacheNames = await caches.keys();
      await Promise.all(
        cacheNames
          .filter((cacheName) => cacheName !== CACHE_NAME)
          .map((cacheName) => caches.delete(cacheName))
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') {
    return;
  }

  const requestUrl = new URL(request.url);
  if (requestUrl.origin !== self.location.origin) {
    return;
  }

  if (request.mode === 'navigate') {
    event.respondWith(
      (async () => {
        try {
          const networkResponse = await fetch(request);
          const cache = await caches.open(CACHE_NAME);
          await cache.put(request, networkResponse.clone());
          return networkResponse;
        } catch {
          return (
            (await caches.match(request)) ??
            (await caches.match(getAppRoot())) ??
            Response.error()
          );
        }
      })()
    );
    return;
  }

  event.respondWith(
    (async () => {
      const cachedResponse = await caches.match(request);
      const networkFetch = fetch(request)
        .then(async (networkResponse) => {
          if (networkResponse.ok) {
            const cache = await caches.open(CACHE_NAME);
            await cache.put(request, networkResponse.clone());
          }
          return networkResponse;
        })
        .catch(() => cachedResponse);

      return cachedResponse ?? networkFetch;
    })()
  );
});
