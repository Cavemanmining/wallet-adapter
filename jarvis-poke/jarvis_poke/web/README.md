# The Pokemon page

`index.html` is the Pokemon page inside the Jarvis app: one self-contained
file, inline CSS and JS, no build step, no framework, no CDN, no external
request of any kind. Open it from disk and it works.

## What it is, and the line it does not cross

This page **monitors and decides. It never buys.** There is no cart, no
payment field, no order call, and no button anywhere on it that completes a
purchase. When a product clears the owner's rule, the page says so, shows
the reasons the engine gave, and offers one thing: a plain anchor to the
retailer's own listing (`target="_blank" rel="noopener noreferrer"`). The
person taps it and finishes the purchase themselves, on the retailer's site,
under the retailer's terms. That boundary is stated in
`jarvis_poke/contracts.py` ("What this is not") and the page repeats it in
plain words next to every link, because a tool that watches is a different
thing from a tool that buys and the difference should never be ambiguous on
screen.

The page also makes no retailer requests itself. It talks to one thing: the
Jarvis API described below. All polling — the per-host minimum interval, the
robots.txt permission, the conditional requests, the widening backoff, the
pause after repeated errors — happens in the service, and the page's sources
strip exists so the owner can *see* that behaviour and stop any source
instantly.

## Mounting it

The page is a single file with no dependencies, so "mounting" it means
serving it and letting it know where the API is.

* **In the Jarvis app shell** — load it in the app's web view (or an
  `<iframe>`) and set `window.JARVIS_POKE_API` *before* the page's own script
  runs. The simplest form is a base URL:

  ```html
  <script>window.JARVIS_POKE_API = "/api/poke";</script>
  <!-- then the page, e.g. inlined or in an iframe -->
  ```

* **Served next to the API** — if the page is served from the same origin as
  the API, a relative base is enough: `window.JARVIS_POKE_API = "poke/"`, and
  the page requests `poke/state`, `poke/rule`, and so on.

* **With headers, or your own transport** — pass an object instead:

  ```js
  window.JARVIS_POKE_API = {
    base: "/api/poke",             // optional when getState/post are given
    headers: { "X-Jarvis-Token": "…" },
    credentials: "same-origin",    // passed to fetch
    refresh_s: 60,                 // auto-refresh interval, floor 15s
    fetch: myFetch,                // optional, defaults to window.fetch
    getState: async () => state,   // optional, replaces GET state entirely
    post: async (path, body) => {} // optional, replaces the POSTs entirely
  };
  ```

  `getState` / `post` let the app hand the page data it already holds
  (over a native bridge, say) without any HTTP at all.

* **Pushing state in** — once loaded, the page exposes
  `window.jarvisPoke.setState(stateObject)` and `window.jarvisPoke.reload()`.
  `setState` renders the object immediately, which is the easy path for a
  native shell that already has the snapshot.

The page auto-refreshes in live mode (default every 60 s, and on tab focus),
but it never refreshes while a rule edit is in progress, so it cannot
overwrite something being typed.

## The JSON contract

Money is **integer cents** in every field whose name ends in `_cents`
(`jarvis_poke/contracts.py`, "Money is integer cents"). The page formats
those integers by integer arithmetic and does no float maths on money at
all. Timestamps are unix seconds (int or float). `null` means "not known",
and the page renders it as "no price" / "no median" / a gap — never as zero.

### `GET state`

```jsonc
{
  "generated_at": 1789254840,          // when this snapshot was computed
  "budget": {
    "total_cents": 60000,
    "spent_cents": 21497,
    "remaining_cents": 38503,          // contracts.Budget.remaining
    "window_days": 30                  // contracts.Budget.window_s / 86400
  },
  "watch": [
    {
      "product": {                     // contracts.Product
        "id": "sv08-surging-sparks-booster-box",
        "name": "Pokemon TCG: Scarlet & Violet-Surging Sparks Booster Box",
        "set_code": "SV08",
        "kind": "booster_box",         // contracts.ProductKind value
        "msrp_cents": 16164            // nullable
      },
      "rule": {                        // contracts.Rule
        "max_price_cents": 14900,
        "quantity": 1,
        "min_discount_pct": 5.0,
        "enabled": true,
        "cooldown_s": 3600,
        "include_shipping": true       // optional, default true
      },
      "best": {                        // cheapest purchasable listing, or null
        "source": "cardbarn",
        "sku": "CA-SV08-…",
        "price_cents": 13499,
        "shipping_cents": 0,
        "landed_cents": 13499,         // contracts.Observation.landed
        "stock": "in_stock",           // contracts.Stock value
        "url": "https://cardbarn.example.com/p/…",
        "at": 1789254420
      },
      "offers": [ { …same shape as best… } ],   // OPTIONAL, see below
      "market": {                      // contracts.MarketRef
        "median_cents": 15250,         // nullable
        "p25_cents": 14399,            // nullable
        "low_cents": 13199,            // nullable
        "samples": 27,
        "stale": false
      },
      "verdict": {                     // contracts.Verdict
        "action": "buy",               // buy | watch | skip | no_stock
        "reasons": ["rule: up to $149.00 landed, …", "…"],
        "discount_pct": 11.48,         // nullable; POSITIVE means under market
        "quantity": 1,
        "at": 1789254750,
        "source": "cardbarn",          // optional
        "sku": "CA-SV08-…",            // optional
        "price_cents": 13499,          // optional
        "landed_cents": 13499,         // optional
        "market_cents": 15250,         // optional
        "url": "https://cardbarn.example.com/p/…"
      },
      "trend": [[1789000000, 14983], [1789021600, null], …]
    }
  ],
  "sources": [
    {
      "id": "examplemart",
      "state": "ok",                   // ok | backoff | paused | disallowed | error
      "next_due_at": 1789254983,
      "errors": 0,                     // consecutive errors
      "paused_until": 0,
      "name": "ExampleMart (placeholder)",  // optional, else id is shown
      "min_interval_s": 300.0,              // optional
      "robots_allows": true,                // optional
      "last_reason": "ok 200"               // optional, shown as a tooltip
    }
  ],
  "recent": [ { …verdict…, "product_name": "…", "set_code": "SV08" } ]
}
```

Notes on the fields the page is fussy about:

* **`trend`** is a list of `[timestamp, cents_or_null]` buckets, oldest
  first. A bucket with no reading must be `null`, not `0`: the sparkline
  draws it as a gap in the line and says how many gaps there were in its
  `aria-label`. Roughly 20–30 buckets reads well; more still works.
* **`offers`** is optional. When present, the expanded row lists every
  source's price; when absent it shows `best` alone and says so. `best`
  should be one of the `offers` entries (matched on `source` + `sku`) so it
  can be highlighted.
* **`verdict.reasons`** is rendered **verbatim**, in order, as the "why".
  The engine already writes these for a person (`DecisionEngine._finish`
  guarantees the list is never empty), so the page does not paraphrase,
  truncate or re-order them.
* **`verdict.discount_pct`** follows `engine.discount_against`: positive
  means the price is *under* the market reference. The row shows the price's
  own signed movement instead — `−11.5%` for eleven and a half per cent
  below the median — because a negative number reading "cheaper" is what a
  person expects from a price. It is exactly `-discount_pct`, and the full
  wording is in the element's tooltip and `aria-label`.
* **`sources[].state`** drives the strip's dot and wording:
  `ok` → "polling normally", `backoff` → "backing off after errors",
  `paused` → "paused", `disallowed` → "robots.txt disallows",
  `error` → "erroring". An unknown value is treated as `ok`.
  `PollScheduler.pause_state()` and `.stats()` carry everything needed to
  fill these in.
* **`rule.include_shipping`** is optional and defaults to `true`. When it
  is `false` the engine scores the shelf price, so the row labels the figure
  "price" rather than "landed". The page cannot edit this flag — `POST rule`
  carries the three fields the owner edits inline plus `enabled` — so the
  service keeps whatever it had.
* Any `url` that is not `http://…` or `https://…` is dropped rather than
  rendered, so a bad or hostile state cannot put a `javascript:` link on the
  page.

### `POST rule`

Sent when the owner saves an inline rule edit.

```json
{"product_id": "sv08-…", "max_price_cents": 14900, "quantity": 1,
 "min_discount_pct": 5.0, "enabled": true}
```

The page validates before sending, with the same limits as
`contracts.Rule.__post_init__`: `max_price_cents` is a positive integer
number of cents parsed from the typed dollars (`"$1,499.00"` → `149900`,
mirroring `contracts.to_cents`), `quantity >= 1`, and
`0 <= min_discount_pct < 100`. A rejected value stays on screen with the
reason next to the field and nothing is sent.

### `POST budget`

```json
{"total_cents": 60000}
```

### `POST pause`

```json
{"source_id": "examplemart", "paused": true}
```

The toggle updates optimistically and reverts if the POST fails.

Any non-2xx response, or a thrown error from a custom `post`, is shown to the
owner with the status text; the page never silently swallows a failed write.
After a successful write the page re-reads `state` rather than guessing what
the service did with it.

## Demo mode

If `window.JARVIS_POKE_API` is not set, the page falls back to an embedded
dataset and is fully explorable with nothing behind it: rows expand, filters
work, rules and the budget can be edited, sources can be paused.

Demo mode is marked, not hidden. A badge sits in the header, the footer says
in full that nothing was fetched and the numbers are invented, the header
reads "sample snapshot" instead of an update time, and every edit answers
with a toast saying the change happened in this page only and was sent
nowhere. Edits mutate the in-memory copy so the interaction is real; a reload
restores the dataset.

The four retailers in the dataset — ExampleMart, CardBarn, HobbyHub,
BigBoxCo — are placeholders, and every URL is under `example.com`, matching
`jarvis_poke/data/sources.json`. The product names and set codes are real,
because that is how the owner searches. The dataset was generated offline
from `lucifer_gen.seed.SeedFields(0x504F4B45DEADBEEF).stream("poke.trend:<id>")`
and pasted in as a literal, so it is reproducible and the page itself draws
no random numbers at runtime. Relative times in demo mode are measured
against the snapshot's own `generated_at`, so it still reads sensibly however
long after it was built.

## Design and accessibility

* Dark first, because this is a phone tool used at odd hours. Both themes are
  CSS custom properties on `:root`; a `prefers-color-scheme: light` block
  supplies the light set, `:root[data-theme="light"]` / `[data-theme="dark"]`
  override it explicitly, and the header's theme button cycles
  auto → light → dark (remembered in `localStorage`, wrapped in try/catch so
  a private window is fine). `body` has an explicit background.
* The status pill is the loudest thing in a row and never relies on colour:
  it carries the word (BUY / WATCH / NO STOCK / SKIP) plus a drawn glyph.
* Every sparkline is `role="img"` with an `aria-label` summarising the range,
  the latest reading and the number of empty buckets.
* Every price column is `tabular-nums`. Interactive targets are at least
  44 px. Focus is a visible 3 px ring. `prefers-reduced-motion` switches
  animation and transitions off. No emoji is used as an icon; the handful of
  icons are inline SVG.
* The budget meter is a `role="progressbar"` with an `aria-valuetext` that
  reads the amounts aloud rather than a bare percentage.
* Loading shows skeleton rows; an error with no data offers a retry and
  quotes the reason; an error with stale data keeps the data and says how old
  it is; empty watchlist, empty sources and empty log each say what would
  make them non-empty.

## Verifying it

From the package root:

```bash
python3 - <<'PY'
import html.parser, pathlib, re
src = pathlib.Path("jarvis_poke/web/index.html").read_text()
html.parser.HTMLParser(convert_charrefs=True).feed(src)        # parses
urls = set(re.findall(r"https?://[^\s\"'<>)]+", src))
assert all("example.com" in u for u in urls), urls             # no live hosts
assert "checkout" not in src.lower()                           # no such action
print(len(src.encode()), "bytes")                              # under 120 KB
PY
```

There is no test module for the page: it is a static file, and the checks
that matter are the ones above plus opening it.
