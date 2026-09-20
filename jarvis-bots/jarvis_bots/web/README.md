# The Bots button and index

Two drop-in files that give Jarvis an entry point for its background helpers.
No framework, no build step, no external requests.

| File | What it is |
| --- | --- |
| `bots-button.js` | `<jarvis-bots-button>`, a custom element for the app's navigation. Shows a badge when a bot wants attention. |
| `bots.html` | The page the button opens: one card per bot with its state, key figures, last event, and a pause control. |

## Mounting the button

The element does not navigate on its own. It emits `jarvis-bots:open` and the
app decides what that means, so it works with a router it has never heard of.

```html
<script type="module" src="/static/bots/bots-button.js"></script>

<jarvis-bots-button label="Bots"
                    status-url="/api/bots/status"
                    poll-ms="30000"></jarvis-bots-button>
```

```js
document.addEventListener('jarvis-bots:open', () => router.go('/bots'));
```

In a bottom tab bar, add `compact` to stack the icon over the label:

```html
<jarvis-bots-button compact label="Bots" status-url="/api/bots/status"></jarvis-bots-button>
```

If Jarvis already knows the status, skip the second poll and push it in:

```js
document.querySelector('jarvis-bots-button').status = { attention: 2, state: 'warn' };
```

Call `setCurrent(true)` when the bots route is active, so the button gets
`aria-current="page"`.

### Matching the app's look

The button reads the app's own custom properties when they exist and falls
back to its own otherwise. Set any of these on an ancestor to restyle it
without touching the component:

`--jarvis-nav-fg`, `--jarvis-nav-fg-strong`, `--jarvis-nav-bg`,
`--jarvis-nav-hover`, `--jarvis-accent`, `--jarvis-warn`, `--jarvis-error`,
`--jarvis-badge-fg`.

## Telling `bots.html` where the API is

This is the step that used to be missing from this file, and missing it is
silent: the page falls back to demo data and there is no console warning, no
error, and one of the invented bots is a plausible "Stock watcher". Three
ways to say it, and the page tries them in this order:

```html
<!-- 1. a global, set before the page's own script runs -->
<script>window.JARVIS_BOTS_API = "/api/bots/";</script>

<!-- 2. or a meta tag, if you would rather not inline a script -->
<meta name="jarvis-bots-api" content="/api/bots/">
```

3. **or nothing at all.** With neither of the above the page requests
   `/api/bots/`, so serving it from the app that owns that route just works.

The difference between 1-2 and 3 is what a failure means. If you named an
endpoint and it does not answer, that is a fault and the page shows it. If
you named none and the default does not answer, nobody wired anything, and
the page shows demo data behind a banner that says so in words and names
this file.

The generated detail pages (`jarvis_bots.cli new-bot`) work the same way,
with `window.JARVIS_BOT_API`, `<meta name="jarvis-bot-api">`, and a default
of `/api/bots/<id>`.

## The endpoints

All four are `jarvis_bots/api.py` one-liners over a `Supervisor`, and the
CLI prints each one so a page can be driven with no server at all
(`python3 -m jarvis_bots.cli state`, `status`, `detail --id <id>`):

| Route | `jarvis_bots.api` |
| --- | --- |
| `GET /api/bots/` | `launcher_state(supervisor)` |
| `GET /api/bots/status` | `badge_status(supervisor)` |
| `GET /api/bots/<id>` | `bot_detail(supervisor, bot_id)` |
| `POST /api/bots/pause` | `set_paused(supervisor, body)` |

`ApiError.status` is 404 for an unknown bot id and 400 for a body that is
not the documented shape.

`GET /api/bots/status` backs the badge. Keep it cheap; it is polled.

```json
{ "attention": 2, "state": "ok" }
```

`state` is `ok`, `warn` or `error`. `attention` is how many bots want a
decision from the owner right now.

`GET /api/bots/` backs the page.

```json
{
  "generated_at": 1758340000,
  "bots": [
    {
      "id": "poke",
      "name": "Pokémon buying assistant",
      "blurb": "Watches sealed product prices and stock.",
      "kind": "cart",
      "state": "running",
      "attention": 1,
      "href": "/bots/poke",
      "can_pause": true,
      "stats": [{ "label": "Watching", "value": "12 products" }],
      "last_event": { "at": 1758338920, "text": "Buy: Surging Sparks ETB at $47.99" }
    }
  ]
}
```

`POST /api/bots/pause` takes `{ "bot_id": "poke", "paused": true }` and
returns the badge payload, so the page that paused a bot can update the nav
from the same response instead of racing the next poll.

`GET /api/bots/<id>` backs a generated detail page: the same card, plus
`detail`, plus the bot's open questions and its recent events (newest
first, up to `jarvis_bots.EVENT_HISTORY`).

```json
{
  "generated_at": 1758340000,
  "bot": {
    "id": "poke", "name": "Pokémon buying assistant", "…": "…",
    "detail": "12 products, 4 sources",
    "attention_items": [
      { "key": "restock:sv08-etb", "text": "Buy: Surging Sparks ETB at $47.99",
        "href": "/bots/poke", "since": 1758338920 }
    ],
    "events": [
      { "at": 1758338920, "severity": "action",
        "text": "Buy: Surging Sparks ETB at $47.99", "href": "/bots/poke" }
    ]
  }
}
```

`kind` picks the icon: `cart`, `grid`, `radar`, or anything else for the
generic bot glyph. `generated_at` is what relative times are measured
against, so a phone with a wrong clock still reads correctly.

## Behaviour worth knowing

- **With nothing to talk to the page renders demo data**, says so in the
  header *and* in a banner that explains how to point it at a real API.
  That makes it explorable before the backend exists, without ever passing
  invented figures off as real. An endpoint you named and that does not
  answer is an error, never demo data.
- **The badge shows a dot when a bot is paused or failing.** Those states
  create no *attention* (a paused bot reports none, and quarantine is the
  supervisor's business, not a decision for you), so the count is zero --
  but the nav still has to show that something is not running.
- **Polling stops while the tab is hidden** and refreshes once on return.
  A badge nobody can see is not worth the battery.
- **The poll interval has a 5 second floor** regardless of what `poll-ms`
  says, and exactly one poll loop is ever live. Re-labelling the button,
  changing `poll-ms`, or removing the element from the DOM leaves no loop
  behind; the endpoint sees the rate you configured and no more.
- **State is never carried by colour alone.** Each pill spells out the word,
  and the button's accessible name includes the count and the condition.

## The boundary, stated in the page itself

These bots watch and decide. They do not buy. When something clears your
rules you get an alert linking to the seller's own page and you complete the
purchase there. There is no cart action anywhere in this UI, and that is on
purpose: automated checkout breaks retailer terms and is what gets accounts
banned and orders cancelled. The slow part of catching a restock is finding
out, and that is fully automated.
