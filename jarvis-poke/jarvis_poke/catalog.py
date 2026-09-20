"""The product catalog: what the owner can watch, and where it is listed.

Implements the "Products" section of :mod:`jarvis_poke.contracts` --
:class:`~jarvis_poke.contracts.Product`,
:class:`~jarvis_poke.contracts.SourceSku` and
:class:`~jarvis_poke.contracts.ProductKind` -- plus the on-disk form of
both, loaded from ``jarvis_poke/data/catalog.json`` (products, canonical
and retailer-independent) and ``jarvis_poke/data/sources.json`` (per-source
SKU mappings, and the :class:`~jarvis_poke.contracts.FetchPolicy` that
:mod:`jarvis_poke.sources` reads from the same file).

Two files because the axes are independent: a product exists whether or
not anybody stocks it, and a source's URL for it changes without the
product changing.  ``Product.id`` is the join key.

What is real and what is not
----------------------------
Product names, set codes and release dates are real, because a watchlist
of invented products is useless and the names are how the owner searches.
Retailers are *not*: every shipped source is an obvious placeholder
(``examplemart``, ``cardbarn``, ``hobbyhub``, ``bigboxco``) with
``example.com`` URLs, and this package ships no retailer's page structure
at all -- parsing is an injected
:class:`~jarvis_poke.contracts.Parser`, exactly as fetching is an injected
:class:`~jarvis_poke.contracts.Fetcher`.  See contracts.py, "What this is
not": this tool monitors and decides, and a person buys.

Validation is loud
------------------
:meth:`Catalog.load` refuses a file it cannot trust and says which entry
is wrong, because a catalog that silently drops a malformed row is a
watchlist with a hole in it.  Enforced on load and on every runtime
``add_*``:

* product ids are unique, non-empty strings;
* every :class:`SourceSku` points at a product that exists, and a source
  lists a product at most once;
* every URL is absolute ``http``/``https`` with a host (it is handed to a
  fetcher and shown to a person as a deep link);
* ``msrp``, when present, is a positive integer number of cents -- money
  is ``int`` cents everywhere in this package (contracts.py, "Money is
  integer cents"), so a float MSRP is rejected rather than rounded;
* ``released``, when present, is an ISO ``YYYY-MM-DD`` date.

No network, no ``urllib``
-------------------------
Nothing here opens a socket, and the URL check is hand-rolled rather than
using ``urllib.parse`` so that a grep for ``urllib`` over this package
stays empty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from jarvis_poke.contracts import Cents, Product, ProductKind, SourceSku

__all__ = [
    "CATALOG_PATH",
    "DATA_DIR",
    "SOURCES_PATH",
    "Catalog",
    "CatalogError",
    "load_catalog",
]

#: Where the shipped data lives.  ``Catalog.load()`` with no arguments
#: reads these two files.
DATA_DIR = Path(__file__).resolve().parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
SOURCES_PATH = DATA_DIR / "sources.json"


class CatalogError(ValueError):
    """A catalog or source file (or a runtime add) that cannot be trusted.

    The message always names the offending entry: the product id, or the
    ``source/product_id`` pair, plus the file it came from when it came
    from a file.
    """


# --------------------------------------------------------------------------
# small validators, deliberately free of urllib
# --------------------------------------------------------------------------


def _is_absolute_http_url(url: Any) -> bool:
    """True for an absolute ``http``/``https`` URL with a non-empty host.

    Hand-rolled: contracts.py forbids this package from touching the
    network, and the simplest way to keep that auditable is for
    ``urllib`` never to appear in it at all.
    """
    if not isinstance(url, str):
        return False
    lowered = url.strip().lower()
    for scheme in ("http://", "https://"):
        if lowered.startswith(scheme):
            rest = url.strip()[len(scheme):]
            host = rest.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
            # A bare host, optionally user@host:port. Must have something
            # that is not punctuation and no whitespace anywhere.
            if not host or any(c.isspace() for c in url.strip()):
                return False
            hostname = host.rsplit("@", 1)[-1].split(":", 1)[0]
            return bool(hostname) and hostname.strip(".") != ""
    return False


def _is_iso_date(text: Any) -> bool:
    if not isinstance(text, str) or len(text) != 10:
        return False
    if text[4] != "-" or text[7] != "-":
        return False
    y, m, d = text[:4], text[5:7], text[8:10]
    if not (y.isdigit() and m.isdigit() and d.isdigit()):
        return False
    return 1 <= int(m) <= 12 and 1 <= int(d) <= 31


def _require_str(value: Any, what: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{where}: {what} must be a non-empty string, got {value!r}")
    return value.strip()


def _msrp(value: Any, where: str) -> Optional[Cents]:
    """MSRP is optional; when present it is a positive ``int`` of cents."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CatalogError(
            f"{where}: msrp must be an integer number of cents "
            f"(money is never a float in this package), got {value!r}"
        )
    if value <= 0:
        raise CatalogError(f"{where}: msrp must be positive, got {value!r}")
    return value


# --------------------------------------------------------------------------
# the catalog
# --------------------------------------------------------------------------


class Catalog:
    """Products, and how each source refers to them.

    Construct from parsed objects (:meth:`from_obj`), from files
    (:meth:`load`), or directly from
    :class:`~jarvis_poke.contracts.Product` /
    :class:`~jarvis_poke.contracts.SourceSku` values.  The same validation
    runs in every path, including the runtime ``add_*`` methods, so an
    in-memory catalog is never weaker than a loaded one.

    Ordering is deterministic everywhere: products sort by ``id``, SKUs by
    ``(source, product_id)``.  Nothing here reads a clock or draws a
    random number.
    """

    def __init__(
        self,
        products: Iterable[Product] = (),
        skus: Iterable[SourceSku] = (),
        *,
        source_labels: Optional[Mapping[str, str]] = None,
        origin: str = "",
    ) -> None:
        self.origin = origin
        self._products: Dict[str, Product] = {}
        self._skus: Dict[Tuple[str, str], SourceSku] = {}
        self._source_labels: Dict[str, str] = dict(source_labels or {})
        for product in products:
            self.add_product(product)
        for sku in skus:
            self.add_sku(sku)

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(
        cls,
        catalog_path: Optional[Path] = None,
        sources_path: Optional[Path] = None,
    ) -> "Catalog":
        """Read the two JSON files.  Defaults to the shipped data."""
        catalog_path = Path(catalog_path) if catalog_path is not None else CATALOG_PATH
        sources_path = Path(sources_path) if sources_path is not None else SOURCES_PATH
        catalog_obj = _read_json(catalog_path)
        sources_obj = _read_json(sources_path)
        return cls.from_obj(
            catalog_obj,
            sources_obj,
            origin=f"{catalog_path.name}+{sources_path.name}",
        )

    @classmethod
    def from_obj(
        cls,
        catalog_obj: Any,
        sources_obj: Any = None,
        *,
        origin: str = "",
    ) -> "Catalog":
        """Build from already-parsed JSON objects.

        ``catalog_obj`` is ``{"products": [...]}`` (a bare list is also
        accepted).  ``sources_obj`` is ``{"sources": [{"id": ...,
        "skus": [...]}, ...]}``; its ``policy`` blocks are ignored here
        and read by :func:`jarvis_poke.sources.load_policies`.
        """
        products = [
            _product_from_obj(entry, origin, index)
            for index, entry in enumerate(_entries(catalog_obj, "products", origin))
        ]
        skus: List[SourceSku] = []
        labels: Dict[str, str] = {}
        for index, entry in enumerate(_entries(sources_obj, "sources", origin)):
            if not isinstance(entry, dict):
                raise CatalogError(f"{origin or 'sources'}: source #{index} is not an object")
            source = _require_str(entry.get("id"), "source id", f"{origin or 'sources'} #{index}")
            label = entry.get("name")
            if isinstance(label, str) and label.strip():
                labels[source] = label.strip()
            raw_skus = entry.get("skus", [])
            if not isinstance(raw_skus, list):
                raise CatalogError(f"source {source!r}: 'skus' must be a list")
            for sku_index, sku_entry in enumerate(raw_skus):
                skus.append(_sku_from_obj(sku_entry, source, sku_index))
        return cls(products, skus, source_labels=labels, origin=origin)

    # -- reading -----------------------------------------------------------

    def products(self) -> List[Product]:
        """Every product, sorted by id."""
        return [self._products[pid] for pid in sorted(self._products)]

    def product(self, product_id: str) -> Product:
        """The product with this id.

        Raises :class:`CatalogError` when it is unknown: a caller that
        named a specific id has a typo or a stale rule, and returning
        ``None`` only moves the crash somewhere less informative.  Use
        :meth:`find` when absence is an ordinary answer.
        """
        try:
            return self._products[product_id]
        except KeyError:
            raise CatalogError(f"unknown product id {product_id!r}") from None

    def find(self, product_id: str) -> Optional[Product]:
        """The product with this id, or ``None``."""
        return self._products.get(product_id)

    def skus_for(self, product_id: str) -> List[SourceSku]:
        """Every source's SKU for one product, sorted by source."""
        self.product(product_id)  # raises for an unknown id
        return [
            self._skus[key]
            for key in sorted(self._skus)
            if key[1] == product_id
        ]

    def sku(self, source: str, product_id: str) -> Optional[SourceSku]:
        """One source's SKU for one product, or ``None`` if it has none."""
        return self._skus.get((source, product_id))

    def skus(self) -> List[SourceSku]:
        """Every SKU, sorted by ``(source, product_id)``.

        This is the scheduler's input: one pollable listing per entry.
        """
        return [self._skus[key] for key in sorted(self._skus)]

    def skus_from(self, source: str) -> List[SourceSku]:
        """Every SKU belonging to one source, sorted by product id."""
        return [self._skus[key] for key in sorted(self._skus) if key[0] == source]

    def sources(self) -> List[str]:
        """Source ids that have at least one SKU, sorted."""
        return sorted({source for source, _ in self._skus})

    def source_label(self, source: str) -> str:
        """The display name for a source, falling back to its id."""
        return self._source_labels.get(source, source)

    def search(self, text: str) -> List[Product]:
        """Case-insensitive search over name, set code and kind.

        The query is split on whitespace and *every* token must appear in
        at least one of those three fields, so "surging etb" finds the
        Surging Sparks Elite Trainer Box while "surging tin" finds
        nothing.  An empty query returns everything, which is what a
        search box with nothing typed in it should show.  Results keep
        the :meth:`products` ordering.
        """
        tokens = [t for t in str(text).lower().split() if t]
        if not tokens:
            return self.products()
        hits: List[Product] = []
        for product in self.products():
            fields = (
                product.name.lower(),
                product.set_code.lower(),
                product.kind.value.lower(),
            )
            if all(any(token in field for field in fields) for token in tokens):
                hits.append(product)
        return hits

    # -- runtime mutation --------------------------------------------------

    def add_product(self, product: Product) -> Product:
        """Add a product.  Raises :class:`CatalogError` on a duplicate id."""
        if not isinstance(product, Product):
            raise CatalogError(f"not a Product: {product!r}")
        pid = _require_str(product.id, "product id", "product")
        if pid in self._products:
            raise CatalogError(f"duplicate product id {pid!r}")
        _require_str(product.name, "name", f"product {pid!r}")
        _require_str(product.set_code, "set_code", f"product {pid!r}")
        if not isinstance(product.kind, ProductKind):
            raise CatalogError(f"product {pid!r}: kind must be a ProductKind, got {product.kind!r}")
        _msrp(product.msrp, f"product {pid!r}")
        if product.released is not None and not _is_iso_date(product.released):
            raise CatalogError(
                f"product {pid!r}: released must be an ISO YYYY-MM-DD date, "
                f"got {product.released!r}"
            )
        self._products[pid] = product
        return product

    def remove_product(self, product_id: str) -> Product:
        """Remove a product *and every SKU pointing at it*.

        Cascading is the only option that keeps the invariant "every SKU
        points at a known product" true at all times.
        """
        product = self.product(product_id)
        del self._products[product_id]
        for key in [k for k in self._skus if k[1] == product_id]:
            del self._skus[key]
        return product

    def add_sku(self, sku: SourceSku) -> SourceSku:
        """Add one source's SKU.  The product must already exist."""
        if not isinstance(sku, SourceSku):
            raise CatalogError(f"not a SourceSku: {sku!r}")
        source = _require_str(sku.source, "source", "sku")
        product_id = _require_str(sku.product_id, "product_id", f"sku on {source!r}")
        where = f"sku {source}/{product_id}"
        _require_str(sku.sku, "sku code", where)
        if product_id not in self._products:
            raise CatalogError(
                f"{where}: points at unknown product {product_id!r}"
            )
        if not _is_absolute_http_url(sku.url):
            raise CatalogError(
                f"{where}: url must be an absolute http(s) url, got {sku.url!r}"
            )
        key = (source, product_id)
        if key in self._skus:
            raise CatalogError(
                f"{where}: source {source!r} already lists product {product_id!r}"
            )
        self._skus[key] = sku
        return sku

    def remove_sku(self, source: str, product_id: str) -> SourceSku:
        """Remove one source's SKU for one product."""
        try:
            return self._skus.pop((source, product_id))
        except KeyError:
            raise CatalogError(
                f"source {source!r} does not list product {product_id!r}"
            ) from None

    # -- conveniences ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._products)

    def __iter__(self) -> Iterator[Product]:
        return iter(self.products())

    def __contains__(self, product_id: object) -> bool:
        return product_id in self._products

    def __repr__(self) -> str:
        return (
            f"<Catalog {len(self._products)} products, {len(self._skus)} skus, "
            f"{len(self.sources())} sources>"
        )

    def to_obj(self) -> Dict[str, Any]:
        """A JSON-ready view, for the page and for round-tripping."""
        return {
            "products": [
                {
                    "id": p.id,
                    "name": p.name,
                    "set_code": p.set_code,
                    "kind": p.kind.value,
                    "msrp": p.msrp,
                    "released": p.released,
                    "upc": p.upc,
                }
                for p in self.products()
            ],
            "skus": [
                {
                    "source": s.source,
                    "product_id": s.product_id,
                    "sku": s.sku,
                    "url": s.url,
                }
                for s in self.skus()
            ],
        }


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise CatalogError(f"no such catalog file: {path}") from None
    except json.JSONDecodeError as exc:
        raise CatalogError(f"{path.name}: not valid JSON ({exc})") from None


def _entries(obj: Any, key: str, origin: str) -> Sequence[Any]:
    if obj is None:
        return ()
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        entries = obj.get(key, [])
        if not isinstance(entries, list):
            raise CatalogError(f"{origin or key}: {key!r} must be a list")
        return entries
    raise CatalogError(f"{origin or key}: expected an object with {key!r}, got {type(obj).__name__}")


def _product_from_obj(entry: Any, origin: str, index: int) -> Product:
    where = f"{origin or 'catalog'} product #{index}"
    if not isinstance(entry, dict):
        raise CatalogError(f"{where}: not an object")
    pid = _require_str(entry.get("id"), "id", where)
    where = f"product {pid!r}"
    kind_value = entry.get("kind")
    try:
        kind = ProductKind(kind_value)
    except ValueError:
        known = ", ".join(k.value for k in ProductKind)
        raise CatalogError(
            f"{where}: unknown kind {kind_value!r} (known kinds: {known})"
        ) from None
    upc = entry.get("upc")
    if upc is not None and not isinstance(upc, str):
        raise CatalogError(f"{where}: upc must be a string or null, got {upc!r}")
    return Product(
        id=pid,
        name=_require_str(entry.get("name"), "name", where),
        set_code=_require_str(entry.get("set_code"), "set_code", where),
        kind=kind,
        msrp=_msrp(entry.get("msrp"), where),
        released=entry.get("released"),
        upc=upc,
    )


def _sku_from_obj(entry: Any, source: str, index: int) -> SourceSku:
    where = f"source {source!r} sku #{index}"
    if not isinstance(entry, dict):
        raise CatalogError(f"{where}: not an object")
    product_id = _require_str(entry.get("product_id"), "product_id", where)
    return SourceSku(
        source=source,
        product_id=product_id,
        sku=_require_str(entry.get("sku"), "sku", f"sku {source}/{product_id}"),
        url=entry.get("url") if isinstance(entry.get("url"), str) else "",
    )


def load_catalog(
    catalog_path: Optional[Path] = None,
    sources_path: Optional[Path] = None,
) -> Catalog:
    """Module-level shorthand for :meth:`Catalog.load`."""
    return Catalog.load(catalog_path, sources_path)


if __name__ == "__main__":  # pragma: no cover - a smoke run, not a CLI
    cat = Catalog.load()
    print(repr(cat))
    for prod in cat.products():
        listed = ", ".join(s.source for s in cat.skus_for(prod.id))
        print(f"  {prod.id:<44} {prod.kind.value:<19} {listed}")
