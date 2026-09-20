/*
 * Jarvis alerts: service worker (reference file).
 *
 * The client half of jarvis_alerts/contracts.py.  The server half writes an
 * alert to a durable outbox and pushes it through a transport; this file is
 * what receives that push while the app is closed, shows it, and collapses
 * the duplicates that at-least-once delivery may produce (contracts.py,
 * design point 3: "the client side is expected to collapse duplicates by
 * alert id").
 *
 * Plain JavaScript.  No bundler, no framework, no external requests: the
 * worker never fetches anything; the whole alert arrives inside the push.
 *
 * Payload (jarvis_alerts.transports.payload_for): a JSON object with
 *   id        string, unique per alert; the dedupe key on this side
 *   kind      string, e.g. "render_done", "gpu_missing", "portal_open"
 *   title     string
 *   body      string
 *   data      object, the app's own; data.url is where a click goes
 *   priority  "low" | "normal" | "high"
 *
 * How to use: serve this file from the app's origin at or above the app's
 * path, so the registration scope covers the app (register-push.js
 * defaults to '/sw.js').  See README.md next to this file.
 */

'use strict';

// ---------------------------------------------------------------------------
// Constants.
// ---------------------------------------------------------------------------

// The Cache API bucket that holds the seen-id list.  The Cache API is used
// as a tiny key-value store because it is available to service workers on
// every browser that supports push, and it survives the worker being killed
// between pushes (which happens: a worker only lives while it has work).
const SEEN_CACHE = 'jarvis-alerts-seen-v1';

// The synthetic request URL the list is stored under.  It is never fetched;
// it is only a key.  Same origin, so cache.put accepts it.
const SEEN_KEY = '/__jarvis_alerts__/seen-ids';

// How many ids to remember.  An at-least-once repeat arrives seconds to
// minutes after the original (a server worker crashed between send and
// mark and its lease expired), so a short memory is enough; 200 covers a
// very busy day.
const SEEN_LIMIT = 200;

// Where a click goes when the alert carries no data.url.
const DEFAULT_URL = '/';

// Notification title when the payload cannot be parsed at all.
const FALLBACK_TITLE = 'Jarvis';

const PRIORITIES = ['low', 'normal', 'high'];

// ---------------------------------------------------------------------------
// Lifecycle: take over at once, so the first push after an update is
// handled by this version and not by a stale one waiting for tabs to close.
// ---------------------------------------------------------------------------

self.addEventListener('install', (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

// ---------------------------------------------------------------------------
// Seen-id memory.  Drops repeats even after the notification's tag is gone
// (the owner dismissed it, or the OS expired it), which tag collapsing
// alone cannot do.
// ---------------------------------------------------------------------------

async function readSeenIds() {
  // Step 1: open the bucket; it is created on first use.
  const cache = await caches.open(SEEN_CACHE);
  // Step 2: look the list up under its key; absent on a fresh install.
  const stored = await cache.match(SEEN_KEY);
  if (!stored) return [];
  // Step 3: parse it.  A corrupt entry counts as empty rather than failing
  // the push: showing a possible duplicate beats showing nothing.
  try {
    const ids = await stored.json();
    return Array.isArray(ids) ? ids.filter((x) => typeof x === 'string') : [];
  } catch (err) {
    return [];
  }
}

async function writeSeenIds(ids) {
  const cache = await caches.open(SEEN_CACHE);
  // A Response is the only value the Cache API stores; its body is the
  // JSON list.  put() overwrites the previous entry.
  await cache.put(
    SEEN_KEY,
    new Response(JSON.stringify(ids), { headers: { 'Content-Type': 'application/json' } })
  );
}

// Returns true if `id` was not seen before (and records it), false if it is
// a repeat.  The list is kept oldest-first and trimmed to SEEN_LIMIT.
async function rememberId(id) {
  const ids = await readSeenIds();
  if (ids.includes(id)) return false;
  ids.push(id);
  while (ids.length > SEEN_LIMIT) ids.shift();
  await writeSeenIds(ids);
  return true;
}

// ---------------------------------------------------------------------------
// Payload parsing.  Tolerant on purpose: a push whose body the worker cannot
// read still shows *something*, because a push that shows nothing is a
// "silent push", which Chrome answers with a generic notification of its
// own and, if repeated, by revoking the subscription.
// ---------------------------------------------------------------------------

function parsePayload(event) {
  // Step 1: no body at all (the server never does this) -> caller shows a
  // generic notification.
  if (!event.data) return null;
  // Step 2: JSON, as payload_for() produces.  PushMessageData.json() and
  // .text() are synchronous.
  let raw;
  try {
    raw = event.data.json();
  } catch (err) {
    // Step 2b: not JSON: show the text as the body so the owner sees it.
    return { id: null, kind: 'unknown', title: FALLBACK_TITLE, body: safeText(event), data: {}, priority: 'normal' };
  }
  if (!raw || typeof raw !== 'object') return null;
  // Step 3: take each field only if it has the expected type; defaults
  // otherwise, so a malformed field never throws mid-push.
  return {
    id: typeof raw.id === 'string' && raw.id ? raw.id : null,
    kind: typeof raw.kind === 'string' ? raw.kind : 'unknown',
    title: typeof raw.title === 'string' && raw.title ? raw.title : FALLBACK_TITLE,
    body: typeof raw.body === 'string' ? raw.body : '',
    data: raw.data && typeof raw.data === 'object' ? raw.data : {},
    priority: PRIORITIES.includes(raw.priority) ? raw.priority : 'normal',
  };
}

function safeText(event) {
  try {
    return event.data.text();
  } catch (err) {
    return '';
  }
}

// Only navigate within our own origin; anything else goes to the app root.
function sameOriginUrl(candidate) {
  if (typeof candidate !== 'string' || !candidate) return DEFAULT_URL;
  try {
    const url = new URL(candidate, self.location.origin);
    return url.origin === self.location.origin ? url.href : DEFAULT_URL;
  } catch (err) {
    return DEFAULT_URL;
  }
}

// ---------------------------------------------------------------------------
// push: parse, dedupe, show.
// ---------------------------------------------------------------------------

self.addEventListener('push', (event) => {
  // waitUntil keeps the worker alive until the promise settles; without it
  // the browser may kill the worker before showNotification has run.
  event.waitUntil(handlePush(event));
});

async function handlePush(event) {
  // Step 1: parse.  With no usable payload at all, show a generic
  // notification rather than nothing (see parsePayload on silent pushes).
  const payload = parsePayload(event);
  if (!payload) {
    await self.registration.showNotification(FALLBACK_TITLE, {
      body: 'You have a new alert.',
      data: { url: DEFAULT_URL },
    });
    return;
  }

  // Step 2: dedupe by alert id across tag expiry.  A repeat is a row the
  // server sent again after a crash between send and mark (at-least-once).
  // The first copy was already shown, so this one is dropped.  Dropping is
  // itself a silent push; repeats are rare enough that browsers tolerate
  // it.  If they ever were not, re-show with the same tag instead of
  // returning here, and the browser collapses it (step 3).
  if (payload.id !== null) {
    const fresh = await rememberId(payload.id);
    if (!fresh) return;
  }

  // Step 3: show.  tag = id makes the browser *replace* a notification with
  // the same tag instead of stacking a second one, which collapses a
  // duplicate that arrives while the first is still on screen.  That is
  // belt and braces with step 2: if the Cache API is unavailable, this
  // still collapses them.
  const url = sameOriginUrl(payload.data && payload.data.url);
  const options = {
    body: payload.body,
    tag: payload.id || undefined,
    renotify: false,                                   // a replaced notification does not buzz again
    data: { url: url, id: payload.id, kind: payload.kind, priority: payload.priority },
    requireInteraction: payload.priority === 'high',   // stays on screen until dismissed
    silent: payload.priority === 'low',                // no sound or vibration
  };
  await self.registration.showNotification(payload.title, options);
}

// ---------------------------------------------------------------------------
// notificationclick: focus an open app window, else open one.
// ---------------------------------------------------------------------------

self.addEventListener('notificationclick', (event) => {
  // Step 1: dismiss the notification; it has done its job.
  event.notification.close();
  // Step 2: the destination is data.url (checked to be same-origin when it
  // was stored), or the app root.
  const url = sameOriginUrl(event.notification.data && event.notification.data.url);
  event.waitUntil(focusOrOpen(url));
});

async function focusOrOpen(url) {
  // Step 3: find the app's open windows.  includeUncontrolled covers a tab
  // that was opened before this worker version took over.
  const windows = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
  // Prefer a window that is already on the destination, else any window.
  const exact = windows.find((client) => client.url === url);
  const client = exact || windows[0];
  if (client) {
    // Step 4a: focus it, and steer it to the destination when it is
    // somewhere else and the browser lets a worker navigate a client.
    if ('focus' in client) await client.focus();
    if (!exact && 'navigate' in client) {
      try {
        await client.navigate(url);
      } catch (err) {
        // The window stays where it was; it is focused, which is the point.
      }
    }
    return;
  }
  // Step 4b: no window is open (the app was closed): open one.
  if (self.clients.openWindow) await self.clients.openWindow(url);
}

// ---------------------------------------------------------------------------
// pushsubscriptionchange: the browser rotated or dropped the subscription.
// Re-subscribing needs the VAPID public key, which this worker does not
// hold (nothing is embedded here), so tell any open window to run
// registerPush again.  With no window open, the app re-registers on its
// next start; meanwhile the push service answers the server with 410, the
// outbox marks the old subscription gone and stores every alert published
// in between, and the re-registration (with backfill_s) queues them.
// ---------------------------------------------------------------------------

self.addEventListener('pushsubscriptionchange', (event) => {
  event.waitUntil(
    self.clients.matchAll({ type: 'window' }).then((windows) => {
      windows.forEach((client) => client.postMessage({ type: 'jarvis-alerts:resubscribe' }));
    })
  );
});
