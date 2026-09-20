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

## The two endpoints

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

`POST /api/bots/pause` takes `{ "bot_id": "poke", "paused": true }`.

`kind` picks the icon: `cart`, `grid`, `radar`, or anything else for the
generic bot glyph. `generated_at` is what relative times are measured
against, so a phone with a wrong clock still reads correctly.

## Behaviour worth knowing

- **With no API configured the page renders demo data** and says so in the
  header. That makes it explorable before the backend exists, without ever
  passing invented figures off as real.
- **Polling stops while the tab is hidden** and refreshes once on return.
  A badge nobody can see is not worth the battery.
- **The poll interval has a 5 second floor** regardless of what `poll-ms`
  says.
- **State is never carried by colour alone.** Each pill spells out the word,
  and the button's accessible name includes the count and the condition.

## The boundary, stated in the page itself

These bots watch and decide. They do not buy. When something clears your
rules you get an alert linking to the seller's own page and you complete the
purchase there. There is no cart action anywhere in this UI, and that is on
purpose: automated checkout breaks retailer terms and is what gets accounts
banned and orders cancelled. The slow part of catching a restock is finding
out, and that is fully automated.
