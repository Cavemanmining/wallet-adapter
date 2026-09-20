# Jarvis alerts: client reference files

The server half of `jarvis_alerts` (see `contracts.py`) writes every alert to
a durable outbox and pushes it through a transport. These two files are the
client half: what receives the push and shows it **while the app is closed**,
and what creates the subscription the server pushes to. They are plain
JavaScript with no bundler and no framework; the Jarvis web app adapts them.

| File | Runs where | Does what |
| --- | --- | --- |
| `sw.js` | the service worker | `push`: parse the payload, drop repeats by alert id, show a notification tagged with the id. `notificationclick`: focus an open app window or open `data.url`. |
| `register-push.js` | the page | `registerPush(...)`: register `sw.js`, ask permission (from a user gesture), subscribe with the VAPID public key, POST the subscription to the server. `unregisterPush(...)` undoes it. |

Nothing is embedded in either file. The VAPID public key and the server URLs
are parameters of `registerPush`. The VAPID **private** key lives only with
the app's sender callable (the thing the app injects into
`jarvis_alerts.transports.WebPushTransport`); it is never in this package and
never in the browser.

## Plugging the files into the app

1. **Serve `sw.js` from the app's origin at the root**, e.g. `/sw.js`. A
   service worker only controls pages at or below its own path, so putting
   it under `/static/` would leave the app uncovered. The `Content-Type` must
   be JavaScript. If the file is served under a different path, pass
   `serviceWorkerUrl` to `registerPush` and `unregisterPush`.

2. **Serve `register-push.js` as an ES module** and import it from the page
   that has the "turn alerts on" control:

   ```js
   import { registerPush, unregisterPush, newDeviceId } from '/register-push.js';

   const deviceId = localStorage.getItem('jarvis.deviceId') || newDeviceId();
   localStorage.setItem('jarvis.deviceId', deviceId);

   enableButton.addEventListener('click', async () => {
     const { vapidPublicKey } = await (await fetch('/api/push/config')).json();
     await registerPush({
       vapidPublicKey,                       // a parameter, not a constant
       subscribeUrl: '/api/push/subscribe',
       profileId: session.profileId,
       deviceId,
     });
   });

   disableButton.addEventListener('click', () =>
     unregisterPush({ unsubscribeUrl: '/api/push/unsubscribe', profileId: session.profileId, deviceId }));
   ```

   `registerPush` must run inside the click handler: browsers grant
   notification permission only from a user gesture. Do not call it at page
   load.

3. **Add the subscribe endpoint.** `registerPush` POSTs
   `{ profileId, deviceId, transport: "webpush", blob }` as JSON, where
   `blob` is the browser's `PushSubscription.toJSON()` serialised to a
   string. The endpoint takes the **profile from the authenticated
   session**, never from the body: a client that could name any profile
   could subscribe its own device to another owner's alerts, or
   unregister that owner's devices. The body's `profileId` is only there
   to be cross-checked. Likewise the transport is fixed by the endpoint,
   not copied from the body (the name is stored and printed by the worker
   as text). Then hand the blob to the service unchanged:

   ```python
   if body.get("profileId") != session.profile_id:
       return 403
   service.register_device(
       session.profile_id, body["deviceId"], "webpush", body["blob"],
       backfill_s=3600, supersede_same_blob=True,
   )
   ```

   `backfill_s` queues the last hour's alerts for the device in the same
   transaction as the registration, so an alert published in the same
   instant cannot slip between "register" and "backfill".
   `supersede_same_blob` forgets any earlier device id that holds the very
   same subscription (a browser whose site data was cleared makes a new
   `deviceId` but keeps its push subscription), so one device never gets
   every push twice.

   The blob is the owner's device data. Store it, never log it; the outbox
   and the transports already refer to it only by `(profile_id, device_id)`.
   The unsubscribe endpoint calls
   `service.unregister_device(session.profile_id, body["deviceId"])`, again
   with the session's profile; without one the push service answers the
   next send with 410 and the outbox prunes the device itself.

4. **Give the worker a real transport.** The app builds
   `WebPushTransport(sender)` where
   `sender(endpoint, payload_bytes, headers, keys)` signs the request with
   the VAPID private key, encrypts the body with `keys["p256dh"]` and
   `keys["auth"]` (the transport reads them out of the stored blob, so the
   outbox is the only copy of the subscription the app keeps), makes the
   HTTP call and returns the status. Any Web Push library does this; the
   package only shapes the payload and interprets the status. With
   `pywebpush`, for example:

   ```python
   def sender(endpoint, body, headers, keys):
       response = webpush(
           {"endpoint": endpoint, "keys": keys}, body,
           vapid_private_key=PRIVATE_KEY_PEM, vapid_claims={"sub": "mailto:owner@example"},
           headers=headers, ttl=int(headers["TTL"]),
       )
       return response.status_code
   ```

5. **Handle `pushsubscriptionchange`.** `sw.js` cannot re-subscribe (it has
   no key), so it posts `{ type: 'jarvis-alerts:resubscribe' }` to open
   windows. Listen for it on `navigator.serviceWorker` and run `registerPush`
   again the next time the owner interacts with the page.

The payload the worker parses is exactly what
`jarvis_alerts.transports.payload_for` produces: `id`, `kind`, `title`,
`body`, `data`, `priority` (`"low" | "normal" | "high"`). Put the page a
click should open in `data.url` (same origin; anything else falls back to
`/`).

## The iOS caveat

Web Push on iOS and iPadOS needs **iOS 16.4 or later** and the app **added
to the Home Screen** (Share → Add to Home Screen), which installs it as a
web app. Only that installed copy can subscribe: `PushManager` does not
exist in a Safari tab, so `registerPush` throws there with a message saying
so, and the "turn alerts on" control should be shown only when
`window.matchMedia('(display-mode: standalone)').matches`.

Two consequences to tell the owner about:

* A push **will not fire while the app is a Safari tab that has been
  closed**. It is the Home Screen copy that receives pushes when closed.
* The Home Screen copy has its own storage and its own subscription. It
  must be registered from *inside* that copy, once; registering in Safari
  does nothing for it.

Also on iOS: the permission prompt appears only from a tap inside the
installed app, and macOS Safari 16+ works without installation.

## Checklist: confirming delivery with the app fully closed

1. **Register from the installed app, in a tap.** Open the app (on iOS, the
   Home Screen copy), tap "turn alerts on", accept the permission prompt.
   Confirm the server received the row: `outbox.subscription(profile_id,
   device_id)` is not `None` and `gone` is `False`.

2. **Close the app completely.** On iOS, swipe it away in the app switcher;
   on desktop, quit the browser (Chrome keeps a background process only if
   "continue running background apps" is on; either way the *tab* must be
   gone). Lock the phone.

3. **Publish a test alert from the server**, not from the page:
   `outbox.publish(Alert(id, profile_id, "test", "Test", "Sent while closed",
   time.time()))`, with a worker running against the real transport.

4. **Watch it arrive** as a system notification within a few seconds
   without opening anything. Then check the record: the outbox row for
   `(alert id, device_id)` is `DELIVERED`, `attempts_for(row_id)` shows the
   attempts, and the push service answered 201.

5. **Tap the notification.** The app opens on `data.url` (or `/`), and if a
   window was already open it is focused instead of a second one opening.
   Publish the same alert id once more (or replay the push): no second
   notification appears, because `sw.js` remembers the id.

If step 4 shows nothing: `outbox.stats()` says where the row is. `pending`
with a future `next_due` means the push service returned a retryable
status and the worker is backing off; `dead` with a reason naming a 404 or
410 means the subscription is stale and the device must register again;
`dead` with a 5xx reason (`dead_reason=exhausted` in `cli dead`) means the
retry budget ran out during an outage: the worker puts it back by itself
fifteen minutes after it died, for as long as the alert is younger than a
day, and `python3 -m jarvis_alerts.cli requeue --row N` puts it back at
once (and is the only way back for an older alert, or for a row whose
`dead_reason` is anything else);
`pending_unreachable` means the device is not registered, was reported
gone, or is pruned after twenty consecutive transient failures (it is
retried after five minutes, or at once when it registers again). A
notification that shows Chrome's generic "This site has been updated in
the background" text means the worker ran but showed nothing of its own:
check that the payload is the JSON above.
