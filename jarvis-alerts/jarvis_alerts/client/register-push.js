/*
 * Jarvis alerts: push registration (reference file).
 *
 * The client half of jarvis_alerts/contracts.py, design point 2: "Sending
 * goes through a subscription registry."  This file creates the browser's
 * push subscription and hands it to the app's server, which stores it as an
 * opaque blob (jarvis_alerts.outbox.Outbox.register, transport "webpush").
 *
 * Nothing is embedded here: the VAPID public key and the server URLs are
 * parameters.  The matching *private* key never reaches the browser or the
 * jarvis_alerts package; it lives with the app's sender, the callable the
 * app injects into jarvis_alerts.transports.WebPushTransport.
 *
 * Plain JavaScript ES module.  No bundler, no framework.
 *
 *   import { registerPush, unregisterPush, newDeviceId } from '/register-push.js';
 *
 *   enableButton.addEventListener('click', async () => {
 *     await registerPush({
 *       vapidPublicKey: config.vapidPublicKey,      // from the app's config endpoint
 *       subscribeUrl: '/api/push/subscribe',
 *       profileId: session.profileId,
 *       deviceId: localStorage.deviceId ||= newDeviceId(),
 *     });
 *   });
 *
 * USER GESTURE.  registerPush asks for notification permission, and
 * browsers grant that only from inside a user gesture: a click, a tap or a
 * key press, in the handler itself.  Called at page load, Chrome and
 * Firefox refuse without asking and Safari throws.  Call it from the
 * control the owner presses to turn alerts on, and nowhere else.
 *
 * PRIVACY.  The subscription is the owner's device data (contracts.py,
 * Subscription.blob "never logged").  It is sent to subscribeUrl and
 * returned to the caller; it is never written to the console here, and the
 * app should not do so either.
 */

const SW_URL_DEFAULT = '/sw.js';
const TRANSPORT = 'webpush';

// ---------------------------------------------------------------------------
// Key conversion.  pushManager.subscribe wants the VAPID public key as raw
// bytes (a 65-byte uncompressed P-256 point); it is handed around as
// base64url text.
// ---------------------------------------------------------------------------

export function base64UrlToUint8Array(base64url) {
  if (typeof base64url !== 'string' || !base64url) throw new Error('applicationServerKey must be a base64url string');
  // Step 1: base64url -> base64: restore the padding and the two swapped chars.
  const padding = '='.repeat((4 - (base64url.length % 4)) % 4);
  const base64 = (base64url + padding).replace(/-/g, '+').replace(/_/g, '/');
  // Step 2: decode.  atob yields a "binary string"; copy each char code out.
  const raw = atob(base64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

function bytesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// Does an existing subscription use the key we were given?  A subscription
// made with another key cannot be signed for by our sender, so it has to be
// replaced.  Browsers that do not expose options.applicationServerKey are
// given the benefit of the doubt.
function subscribedWithKey(subscription, key) {
  const current = subscription.options && subscription.options.applicationServerKey;
  if (!current) return true;
  return bytesEqual(new Uint8Array(current), key);
}

function checkSupport() {
  if (!('serviceWorker' in navigator)) throw new Error('service workers are not supported in this browser');
  if (!('PushManager' in window)) {
    throw new Error('Web Push is not available here (on iOS it needs 16.4+ and the app added to the Home Screen)');
  }
  if (!('Notification' in window)) throw new Error('notifications are not supported in this browser');
}

// ---------------------------------------------------------------------------
// registerPush
// ---------------------------------------------------------------------------

/**
 * Register the service worker, get permission, subscribe, and tell the
 * server.  Returns the PushSubscription.  Must run inside a user gesture
 * (see the file header).
 *
 * @param {object} options
 * @param {string} options.vapidPublicKey  base64url application server key
 * @param {string} options.subscribeUrl    server endpoint that stores the subscription
 * @param {string} options.profileId       the owner
 * @param {string} options.deviceId        this browser/device; stable across visits
 * @param {string} [options.serviceWorkerUrl='/sw.js']
 * @param {function} [options.fetchImpl=fetch]   injectable for tests
 */
export async function registerPush({
  vapidPublicKey,
  subscribeUrl,
  profileId,
  deviceId,
  serviceWorkerUrl = SW_URL_DEFAULT,
  fetchImpl = fetch,
}) {
  if (!vapidPublicKey) throw new Error('vapidPublicKey is required');
  if (!subscribeUrl) throw new Error('subscribeUrl is required');
  if (!profileId || !deviceId) throw new Error('profileId and deviceId are required');
  checkSupport();

  // Step 1: register the worker (idempotent: the browser reuses an existing
  // registration for the same URL) and wait until it is active, because a
  // subscription needs an active worker to deliver to.
  const registration = await navigator.serviceWorker.register(serviceWorkerUrl);
  await navigator.serviceWorker.ready;

  // Step 2: permission.  Ask only when undecided; asking again after a
  // denial is refused by browsers and annoys the owner.  This is the call
  // that needs the user gesture.
  let permission = Notification.permission;
  if (permission === 'default') permission = await Notification.requestPermission();
  if (permission !== 'granted') throw new Error(`notification permission is "${permission}"`);

  // Step 3: subscribe.  userVisibleOnly: true promises the browser that
  // every push shows a notification (sw.js keeps that promise); it is
  // required by Chrome.  Reuse an existing subscription made with the same
  // key; replace one made with a different key.
  const applicationServerKey = base64UrlToUint8Array(vapidPublicKey);
  let subscription = await registration.pushManager.getSubscription();
  if (subscription && !subscribedWithKey(subscription, applicationServerKey)) {
    await subscription.unsubscribe();
    subscription = null;
  }
  if (!subscription) {
    subscription = await registration.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey });
  }

  // Step 4: hand it to the server.  JSON.stringify(subscription) calls
  // PushSubscription.toJSON(): { endpoint, expirationTime, keys: { p256dh,
  // auth } }, exactly the blob WebPushTransport reads the endpoint from and
  // the app's sender reads the keys from.  The server stores it with
  // Outbox.register(Subscription(profileId, deviceId, "webpush", blob, now)).
  const response = await fetchImpl(subscribeUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({
      profileId,
      deviceId,
      transport: TRANSPORT,
      blob: JSON.stringify(subscription),
    }),
  });
  if (!response.ok) throw new Error(`subscribe endpoint answered ${response.status}`);

  // Step 5: return it; the app may keep it in memory but should not log it.
  return subscription;
}

// ---------------------------------------------------------------------------
// unregisterPush
// ---------------------------------------------------------------------------

/**
 * Drop this device's subscription at the push service and, when
 * unsubscribeUrl is given, tell the server to forget it (the server calls
 * Outbox.unregister(profileId, deviceId)).  Returns true if a subscription
 * was released.  Safe to call when nothing is registered.  Without the
 * server call, the next send is answered 410 by the push service and the
 * outbox prunes the device by itself; telling it is just tidier.
 */
export async function unregisterPush({
  unsubscribeUrl,
  profileId,
  deviceId,
  serviceWorkerUrl = SW_URL_DEFAULT,
  fetchImpl = fetch,
} = {}) {
  if (!('serviceWorker' in navigator)) return false;
  // Step 1: find our registration; none means nothing to do.
  const registration = await navigator.serviceWorker.getRegistration(serviceWorkerUrl);
  if (!registration) return false;
  // Step 2: release the subscription at the push service.
  const subscription = await registration.pushManager.getSubscription();
  let released = false;
  if (subscription) released = await subscription.unsubscribe();
  // Step 3: tell the server, best effort.
  if (unsubscribeUrl && profileId && deviceId) {
    try {
      await fetchImpl(unsubscribeUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ profileId, deviceId, transport: TRANSPORT }),
      });
    } catch (err) {
      // The server finds out on its next send (410 -> pruned).
    }
  }
  return released;
}

// ---------------------------------------------------------------------------
// newDeviceId: a stable id for this browser, generated once by the app and
// kept (localStorage is the usual place).  The outbox keys subscriptions by
// (profileId, deviceId), so a browser that re-registers with the same id
// replaces its own row instead of adding a ghost.
// ---------------------------------------------------------------------------

export function newDeviceId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}
