"""The one seam between deciding and telling: a BUY verdict to a phone.

Design: :mod:`jarvis_poke.contracts` -- "A watchlist, a stock and price
monitor, a deal scorer, and a decision engine that tells the owner *when to
act and why*, delivering that through ``jarvis_alerts`` so it reaches a
phone with the app closed" -- and :class:`~jarvis_poke.contracts.Verdict`,
whose ``should_alert`` is true for exactly one action.

What this module does
---------------------
It turns a :class:`~jarvis_poke.contracts.Verdict` into one call to
:class:`jarvis_alerts.api.AlertService`.  That is the whole of it.  The
alert carries a deep link to the listing so the notification the service
worker shows opens the product page the owner then buys from, by hand.

What this module must never do
------------------------------
contracts.py: "It does not check out."  Nothing here carts, pays, or
retries a checkout.  The terminal output of the engine is a verdict plus a
link a person taps, and this module is the delivery of that link and
nothing more.  It holds no retailer knowledge either: it never names a
retailer, never parses a page, and takes the url from the verdict.

Privacy: ``jarvis_alerts`` keeps each device's push subscription as the
owner's opaque data and never puts it in a message.  This module holds up
that end -- it never reads the registry, and :func:`alert_data` is checked
to be flat JSON scalars, so a blob or a credential cannot ride along in an
alert's ``data`` even by accident.

Deduplication, and why the key is the landed price
--------------------------------------------------
``dedupe_key`` is ``poke:buy|<product_id>|<source>|<landed cents>``.

* A listing that flaps -- in stock, out, in again within a minute, which is
  what a restock looks like from outside -- produces the same key every
  time, so the owner gets one notification rather than nine.
* A *genuine* price drop changes the key, and a drop is exactly the news
  worth a second buzz: the first alert said $54.99 and the owner passed;
  $44.99 is a different decision.
* The price in the key is the **landed** price (shipping included), because
  that is the number the owner compares against the rule's cap.  Keying on
  the shelf price would let a shipping change of a dollar pass silently.
* The source is in the key so two retailers hitting the cap in the same
  minute are two alerts: they are two different purchases, and the deep
  link differs.

The key is time-free, so the *window* does the rest, in two places that
deliberately agree.  ``jarvis_alerts.outbox`` collapses a repeat of a key
published less than ``DEDUPE_WINDOW_S`` ago; this bridge also refuses to
hand the service a repeat inside the same window, so the suppression holds
even for an ``AlertService`` wired to some other store.  Neither is the
*cooldown*: :class:`~jarvis_poke.contracts.Rule` has ``cooldown_s`` and the
engine enforces it, which is a statement about how often the owner wants
to be asked to spend money.  This window is only about not sending the
same sentence twice.

Determinism: the clock is injected, as everywhere else in this package.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

from jarvis_alerts.contracts import Priority
from jarvis_alerts.outbox import DEDUPE_WINDOW_S

from jarvis_poke.contracts import Action, Cents, Product, Verdict, fmt_cents
from jarvis_poke.engine import explain

__all__ = [
    "ALERT_KIND",
    "DEDUPE_WINDOW_S",
    "DEFAULT_PROFILE_ID",
    "AlertBridge",
    "BridgeError",
    "alert_data",
    "dedupe_key_for",
    "landed_cents",
]

#: The alert ``kind`` every buy notification carries.  One kind, so the
#: client can route all of them to the same tap handler.
ALERT_KIND = "poke_buy"

#: Whose phone, when the app does not say.  ``jarvis_alerts`` is
#: multi-profile; a personal watchlist has one owner.
DEFAULT_PROFILE_ID = "owner"

#: The keys :meth:`AlertBridge.publish_verdict` puts in an alert's ``data``,
#: and the only keys it puts there.  The service worker reads ``url`` to
#: deep link straight to the product page.
DATA_KEYS = (
    "product_id", "source", "sku", "url", "price_cents", "quantity", "market_cents",
)

Clock = Callable[[], float]


class BridgeError(ValueError):
    """A verdict that cannot be turned into an honest alert.

    ``ValueError`` to match the rest of the package.  Raised for a verdict
    paired with the wrong product, and for a BUY with no price -- there is
    no way to write "buy this at ..." without one, and no way to dedupe it.
    """


# --------------------------------------------------------------------------
# Pure helpers: what the alert says, and how it is keyed
# --------------------------------------------------------------------------


def landed_cents(verdict: Verdict) -> Cents:
    """The price the owner actually pays, shipping included.

    ``Verdict.landed`` when the engine set it, otherwise ``Verdict.price``.
    Raises :class:`BridgeError` when neither is set: contracts.py has BUY
    mean "meets every rule", and a rule is a ceiling on a number.
    """
    for value in (verdict.landed, verdict.price):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise BridgeError(
                    f"verdict price must be integer cents (contracts.py: money is "
                    f"never a float); got {type(value).__name__} {value!r}"
                )
            return value
    raise BridgeError(
        f"verdict for {verdict.product_id!r} is a BUY with no price; there is "
        f"nothing to put in the alert and nothing to dedupe on"
    )


def dedupe_key_for(verdict: Verdict) -> str:
    """The dedupe key for a buy alert: product, source, landed price.

    See this module's docstring for why those three and nothing else.  The
    separator is ``|`` because a product id is a slug and a source id is a
    host label; neither contains one.
    """
    source = verdict.source or "unknown"
    return f"poke:buy|{verdict.product_id}|{source}|{landed_cents(verdict)}"


def alert_data(verdict: Verdict) -> Dict[str, Any]:
    """The alert's ``data``: exactly :data:`DATA_KEYS`, all JSON scalars.

    ``url`` is the deep link the notification opens -- the listing page,
    straight from the verdict, so the owner lands on the thing being
    bought rather than on a home page.  ``market_cents`` is present even
    when it is ``None``, so the client never has to branch on a missing
    key.
    """
    data: Dict[str, Any] = {
        "product_id": verdict.product_id,
        "source": verdict.source,
        "sku": verdict.sku,
        "url": verdict.url,
        "price_cents": landed_cents(verdict),
        "quantity": int(verdict.quantity),
        "market_cents": verdict.market,
    }
    _check_payload(data)
    return data


def _check_payload(data: Dict[str, Any], keys: Any = None) -> None:
    """Refuse anything that is not a flat JSON scalar.

    A nested object in an alert's ``data`` is how a subscription blob, a
    token or a whole config ends up in a notification.  This module builds
    the payload itself, so the check can only ever fail on a future edit --
    which is the point of having it.

    ``keys`` is the allowlist the payload must match exactly, defaulting to
    :data:`DATA_KEYS`.  :mod:`jarvis_poke.snipe` passes its own superset:
    the allowlist is per-payload-shape, but the scalar rule below is not
    negotiable and is the same for every caller.
    """
    allowed = DATA_KEYS if keys is None else tuple(keys)
    if set(data) != set(allowed):
        raise BridgeError(
            f"alert data must carry exactly {sorted(allowed)}; got {sorted(data)}"
        )
    for key, value in data.items():
        if value is None or isinstance(value, str):
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise BridgeError(
                f"alert data[{key!r}] must be a string, an int or None; got "
                f"{type(value).__name__}"
            )
    json.dumps(data)  # a last, cheap proof that it is serialisable


def _title(product: Product, verdict: Verdict) -> str:
    """"Buy 2 x <name> at $44.99" -- the product and the price, which is
    all a lock screen shows before the owner decides to look."""
    price = fmt_cents(landed_cents(verdict))
    quantity = f"{verdict.quantity} x " if verdict.quantity > 1 else ""
    return f"Buy {quantity}{product.name} at {price}"


def _body(verdict: Verdict) -> str:
    """The engine's own explanation, plus the source when it is not
    already in it (``engine.explain`` names the source next to the price,
    but only when there is a price to name it next to)."""
    text = explain(verdict)
    source = verdict.source
    if source and source not in text:
        text = f"{text} Source: {source}."
    return text


# --------------------------------------------------------------------------
# The bridge
# --------------------------------------------------------------------------


class AlertBridge:
    """Publishes BUY verdicts through :class:`jarvis_alerts.api.AlertService`.

    ``alert_service``  anything with ``AlertService.publish``'s signature;
                       the tests pass a fake, the app passes the real one.
    ``clock``          returns unix seconds.  Required, and used only for
                       this bridge's own dedupe window -- the alert's
                       ``created_at`` is stamped by the service's clock, so
                       give both the same one.
    ``profile_id``     whose phone (``jarvis_alerts`` is multi-profile).
    ``dedupe_window_s``  defaults to the outbox's own window, so the two
                       layers suppress the same repeats.

    It holds one small piece of state: when each dedupe key was last
    published, pruned as it goes.  Nothing else; no retailer knowledge, no
    device registry, no credentials.
    """

    def __init__(
        self,
        alert_service: Any,
        clock: Clock,
        *,
        profile_id: str = DEFAULT_PROFILE_ID,
        kind: str = ALERT_KIND,
        dedupe_window_s: float = DEDUPE_WINDOW_S,
    ) -> None:
        if alert_service is None or not callable(getattr(alert_service, "publish", None)):
            raise BridgeError(f"alert_service has no publish(): {alert_service!r}")
        if not callable(clock):
            raise BridgeError(
                "AlertBridge needs an injected clock: a callable returning unix "
                "seconds (this package never calls time.time())"
            )
        if not isinstance(profile_id, str) or not profile_id:
            raise BridgeError("profile_id must be a non-empty string")
        if not isinstance(kind, str) or not kind:
            raise BridgeError("kind must be a non-empty string")
        if dedupe_window_s < 0:
            raise BridgeError("dedupe_window_s must not be negative")
        self.service = alert_service
        self._clock = clock
        self.profile_id = profile_id
        self.kind = kind
        self.dedupe_window_s = float(dedupe_window_s)
        #: dedupe key -> (when it was published, the alert id it got)
        self._recent: Dict[str, tuple] = {}
        self.published = 0
        self.suppressed = 0

    def __repr__(self) -> str:
        return (
            f"<AlertBridge profile={self.profile_id!r} published={self.published} "
            f"suppressed={self.suppressed}>"
        )

    def now(self) -> float:
        return float(self._clock())

    # -- the one method that matters ---------------------------------------

    def publish_verdict(self, verdict: Verdict, product: Product) -> Optional[str]:
        """Alert the owner about a BUY; return the alert id, or ``None``.

        ``None`` for WATCH, SKIP and NO_STOCK -- contracts.py gives BUY
        alone ``should_alert``, and a monitor that buzzes for "still out of
        stock" is a monitor that gets muted, after which it cannot do its
        one job.

        For a BUY: the title names the product and the landed price, the
        body is :func:`jarvis_poke.engine.explain` plus the source, the
        data is :func:`alert_data` (deep link included) and the priority is
        HIGH -- a restock at a price the owner set a rule for is the one
        thing this tool is allowed to interrupt someone about.

        A repeat of the same key inside :attr:`dedupe_window_s` is not
        published again; the id of the alert that *was* published comes
        back instead, so a caller never has to branch (the same bargain
        ``AlertService.publish`` makes for an alert the outbox collapses).
        :attr:`suppressed` counts those.
        """
        if not isinstance(verdict, Verdict):
            raise BridgeError(f"not a Verdict: {verdict!r}")
        if verdict.action is not Action.BUY:
            return None
        if not isinstance(product, Product):
            raise BridgeError(f"not a Product: {product!r}")
        if product.id != verdict.product_id:
            raise BridgeError(
                f"verdict is about {verdict.product_id!r} but the product given is "
                f"{product.id!r}; an alert naming the wrong box is worse than none"
            )

        key = dedupe_key_for(verdict)
        now = self.now()
        self._forget_old(now)
        previous = self._recent.get(key)
        if previous is not None and now - previous[0] < self.dedupe_window_s:
            self.suppressed += 1
            return previous[1]

        alert_id = self.service.publish(
            self.profile_id,
            self.kind,
            self.alert_title(product, verdict),
            self.alert_body(verdict),
            data=self.build_data(verdict),
            priority=Priority.HIGH,
            dedupe_key=key,
        )
        self._recent[key] = (now, alert_id)
        self.published += 1
        return alert_id

    # -- what the notification says (overridable) --------------------------
    #
    # Three seams, so a subclass can change the words and the payload
    # without re-implementing the dedupe bookkeeping above --
    # :class:`jarvis_poke.snipe.SnipeAlertBridge` is that subclass.  The
    # defaults are the module-level functions, which stay the reference
    # behaviour and stay tested on their own.

    def alert_title(self, product: Product, verdict: Verdict) -> str:
        return _title(product, verdict)

    def alert_body(self, verdict: Verdict) -> str:
        return _body(verdict)

    def build_data(self, verdict: Verdict) -> Dict[str, Any]:
        return alert_data(verdict)

    # -- internals ----------------------------------------------------------

    def _forget_old(self, now: float) -> None:
        """Drop keys outside the window, so a long-running monitor's memory
        is bounded by how many distinct prices it saw in five minutes."""
        cutoff = now - self.dedupe_window_s
        for key in [k for k, (at, _) in self._recent.items() if at <= cutoff]:
            del self._recent[key]


# --------------------------------------------------------------------------
# Smoke run: python3 -m jarvis_poke.alerts_bridge
# --------------------------------------------------------------------------


def _demo() -> Dict[str, Any]:
    """One BUY, the same BUY again, a cheaper BUY, and a WATCH, through a
    fake service with a hand-driven clock.  No network, no sleeping."""
    from jarvis_poke.contracts import ProductKind

    published = []

    class FakeService:
        def publish(self, profile_id, kind, title, body, data=None, priority=None,
                    dedupe_key=None):
            published.append(
                {"profile_id": profile_id, "kind": kind, "title": title, "body": body,
                 "data": data, "priority": int(priority), "dedupe_key": dedupe_key}
            )
            return f"alert{len(published)}"

    ticks = [1_700_000_000.0]
    bridge = AlertBridge(FakeService(), clock=lambda: ticks[0])
    product = Product(
        id="sv08-surging-sparks-etb",
        name="Placeholder Elite Trainer Box",
        set_code="SV08",
        kind=ProductKind.ELITE_TRAINER_BOX,
        msrp=4999,
    )

    def verdict(action: Action, landed: Optional[int]) -> Verdict:
        return Verdict(
            product_id=product.id, action=action, at=ticks[0], source="examplemart",
            sku="EM-SV08-ETB", price=landed, landed=landed, market=5200,
            discount_pct=13.5 if landed else None, quantity=1,
            url="https://examplemart.example.com/p/sv08-surging-sparks-etb",
            reasons=("in stock at examplemart", "at or under the cap", "within budget"),
        )

    ids = [bridge.publish_verdict(verdict(Action.BUY, 4499), product)]
    ticks[0] += 30.0
    ids.append(bridge.publish_verdict(verdict(Action.BUY, 4499), product))   # flapping
    ticks[0] += 30.0
    ids.append(bridge.publish_verdict(verdict(Action.BUY, 3999), product))   # a real drop
    for action in (Action.WATCH, Action.SKIP, Action.NO_STOCK):
        ids.append(bridge.publish_verdict(verdict(action, 9999), product))
    return {
        "returned": ids,
        "publishes": len(published),
        "suppressed": bridge.suppressed,
        "titles": [p["title"] for p in published],
        "body": published[0]["body"],
        "data_keys": sorted(published[0]["data"]),
        "priorities": sorted({p["priority"] for p in published}),
        "dedupe_keys": [p["dedupe_key"] for p in published],
    }


if __name__ == "__main__":  # pragma: no cover - a smoke run, not a CLI
    print(json.dumps(_demo(), indent=2, sort_keys=True))
