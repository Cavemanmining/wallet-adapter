"""Generate a new bot: a module, a page and a test, from one command.

This is the file that makes :mod:`jarvis_bots` a platform rather than a
program.  ``jarvis_bots/contracts.py`` opens by promising that "adding the
second bot should be a class and a page, not a refactor"; this module is
where that promise is cashed.  :func:`new_bot` writes three files that
already work together:

===========================  ==============================================
``<module>.py``              a complete bot: identity, tick, events,
                             attention, status, snapshot/restore
``<bot id>.html``            its detail page, in the idiom of
                             ``jarvis_bots/web/bots.html``
``test_<module>.py``         pytest covering tick, attention and persistence
===========================  ==============================================

Nothing is a stub.  The generated test passes against the generated bot with
no edits, which is the only way a scaffold stays honest: if the template
rots, the generated test fails on the next run.

The page is not a stub either, and that took a fix rather than a promise:
it reads a detail payload that :func:`jarvis_bots.api.bot_detail` produces
and ``python3 -m jarvis_bots.cli detail --id <id>`` prints.  For a while
nothing in the package could produce it -- the supervisor kept one event
per bot and the page renders a feed -- so the generated page's "Recent
events" section was unfillable by construction.

Why ``str.replace`` and not ``str.format``
------------------------------------------
The templates are real Python, CSS and JavaScript, and all three are full of
braces.  ``format`` would have to have every one of them doubled, which makes
the templates unreadable and un-runnable on their own.  So substitution is
explicit: each marker is ``@@NAME@@`` -- a spelling that occurs in no Python
dunder, no CSS rule and no JS idiom -- and :func:`render` replaces exactly the
markers it was given and then checks that no ``@@...@@`` survived.

What the generator refuses
--------------------------
* An id that is not a simple lowercase slug.  ``BotInfo.__post_init__``
  accepts more than that (see :func:`validate_slug` for the two cases where
  it is too generous), but the id also has to become a Python module name and
  a filename, so this module holds the tighter line.
* Overwriting a file, unless ``force=True``.  A half-generated bot is worse
  than none, so the check runs over *all* the destinations before anything is
  written: either every file is created or none is.

Determinism and the outside world
---------------------------------
Rendering is pure text substitution: same arguments, same bytes, no clock and
no randomness.  Nothing here opens a socket, and neither does anything it
writes.
"""

from __future__ import annotations

import html
import json
import keyword
import re
from pathlib import Path
from typing import Dict, List, Mapping

__all__ = [
    "ScaffoldError",
    "TEMPLATE_DIR",
    "SLUG_RE",
    "KNOWN_KINDS",
    "validate_slug",
    "module_name_for",
    "class_name_for",
    "planned_paths",
    "render",
    "new_bot",
]


class ScaffoldError(Exception):
    """A refusal the caller can print as one line.

    The CLI turns this into a single line on stderr and exit status 1; no
    traceback, because a bad ``--id`` is a user error, not a crash.
    """


#: Where ``bot.py.tmpl``, ``page.html.tmpl`` and ``test.py.tmpl`` live.
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

#: A bot id: lowercase ASCII, starting with a letter, words joined by single
#: ``-`` or ``_``.  Deliberately narrower than ``BotInfo``; see
#: :func:`validate_slug`.
SLUG_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")

#: The kinds ``bots.html`` has an icon for.  Anything else is accepted and
#: renders as the generic bot glyph -- ``BotInfo.kind`` says so explicitly --
#: so this list steers the CLI's ``--kind`` choices rather than gating them.
KNOWN_KINDS = ("cart", "grid", "radar", "bot")

#: Markers the generator knows how to substitute, per template.
_MARKER_RE = re.compile(r"@@[A-Z0-9_]+@@")

_MAX_ID_LEN = 40
_MAX_TEXT_LEN = 200


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------


def validate_slug(bot_id: str) -> str:
    """Return ``bot_id`` if it is usable as an id, a module name and a file
    name; raise :class:`ScaffoldError` otherwise.

    ``BotInfo.__post_init__`` only asks that the id be non-empty and
    ``id.replace("_", "").replace("-", "").isalnum()``.  That is too generous
    twice over for a generator:

    * ``str.isalnum`` is true for non-ASCII digits and letters, so ``"1²"``
      or ``"café"`` pass it.  Those make poor file names and worse JSON
      keys.
    * ``"9"`` passes it and is not a legal Python module name.

    Everything this function accepts is also accepted by ``BotInfo``, so a
    generated bot never trips the contract's own check.
    """
    if not isinstance(bot_id, str):
        raise ScaffoldError(f"bot id must be a string, got {type(bot_id).__name__}")
    bot_id = bot_id.strip()
    if not bot_id:
        raise ScaffoldError("bot id is required")
    if len(bot_id) > _MAX_ID_LEN:
        raise ScaffoldError(f"bot id is too long (max {_MAX_ID_LEN}): {bot_id!r}")
    if not SLUG_RE.match(bot_id):
        raise ScaffoldError(
            f"bot id must be a lowercase slug like 'stock-watcher': {bot_id!r}"
        )
    module = bot_id.replace("-", "_")
    if keyword.iskeyword(module) or keyword.issoftkeyword(module):
        raise ScaffoldError(f"bot id would shadow a Python keyword: {bot_id!r}")
    if not module.isidentifier():  # pragma: no cover - SLUG_RE already ensures it
        raise ScaffoldError(f"bot id is not a usable module name: {bot_id!r}")
    return bot_id


def module_name_for(bot_id: str) -> str:
    """``stock-watcher`` -> ``stock_watcher``.  Hyphens read well in an id and
    in a URL; Python will not import them."""
    return validate_slug(bot_id).replace("-", "_")


def class_name_for(bot_id: str) -> str:
    """``stock-watcher`` -> ``StockWatcherBot``.

    An id that already ends in ``bot`` keeps its word, so ``spend-bot`` is
    ``SpendBot`` and not ``SpendBotBot``.
    """
    parts = [p for p in re.split(r"[-_]", validate_slug(bot_id)) if p]
    name = "".join(p[:1].upper() + p[1:] for p in parts)
    if not parts[-1].lower() == "bot":
        name += "Bot"
    return name


def _clean_text(value: str, field: str, fallback: str) -> str:
    """One line of display text: no control characters, bounded length."""
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ScaffoldError(f"{field} must be a string, got {type(value).__name__}")
    text = " ".join(value.split())
    if not text:
        text = fallback
    if len(text) > _MAX_TEXT_LEN:
        raise ScaffoldError(f"{field} is too long (max {_MAX_TEXT_LEN} characters)")
    return text


def _docstring_safe(text: str) -> str:
    """Text that can sit inside a triple-quoted docstring unescaped.

    Backslashes and double quotes are the only two ways a caller's ``--name``
    could end a docstring early or smuggle an escape into the generated
    module, so both are neutralised here.  Every *other* use of the name in
    generated Python goes through :func:`_py_literal`.
    """
    return text.replace("\\", "").replace('"', "'")


def _py_literal(text: str) -> str:
    """A Python string literal for ``text``.

    ``json.dumps`` with the default ``ensure_ascii`` emits only escapes that
    Python reads the same way, so its output is a valid Python literal and
    the caller's quotes, backslashes and non-ASCII survive intact.
    """
    return json.dumps(text)


def _js_literal(text: str) -> str:
    """A JavaScript string literal safe to embed in an inline ``<script>``.

    ``</script>`` inside a string would end the element, so the three
    characters that could start markup are escaped to their ``\\uXXXX`` forms.
    """
    out = json.dumps(text)
    return out.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render(template: str, values: Mapping[str, str]) -> str:
    """Substitute ``@@NAME@@`` markers by explicit replacement.

    Not ``str.format``: the templates are working Python, CSS and JavaScript,
    all of which use braces, and doubling every brace would stop them being
    readable or runnable.

    Raises :class:`ScaffoldError` if a marker survives, which is the check
    that catches a template using a name the generator does not supply.
    """
    out = template
    for name, value in values.items():
        out = out.replace(f"@@{name}@@", value)
    leftover = sorted(set(_MARKER_RE.findall(out)))
    if leftover:
        raise ScaffoldError(
            "template has markers the generator does not fill: " + ", ".join(leftover)
        )
    return out


def _substitutions(bot_id: str, name: str, blurb: str, kind: str) -> Dict[str, str]:
    """Every marker the three templates may use, in one mapping.

    Each value is escaped for where it lands: ``*_LITERAL`` for Python,
    ``*_JS`` for an inline script, ``*_HTML`` for element content, and
    ``*_TEXT`` for a docstring.  Escaping at the substitution site, rather
    than in the templates, is what keeps a bot named ``Smith & Co "deals"``
    from producing broken markup.
    """
    module = module_name_for(bot_id)
    cls = class_name_for(bot_id)
    return {
        "BOT_ID": bot_id,
        "BOT_MODULE": module,
        "BOT_CLASS": cls,
        "BOT_PAGE": f"{bot_id}.html",
        "BOT_NAME_TEXT": _docstring_safe(name),
        "BOT_BLURB_TEXT": _docstring_safe(blurb),
        "BOT_ID_LITERAL": _py_literal(bot_id),
        "BOT_NAME_LITERAL": _py_literal(name),
        "BOT_BLURB_LITERAL": _py_literal(blurb),
        "BOT_KIND_LITERAL": _py_literal(kind),
        "BOT_ID_JS": _js_literal(bot_id),
        "BOT_NAME_JS": _js_literal(name),
        "BOT_BLURB_JS": _js_literal(blurb),
        "BOT_KIND_JS": _js_literal(kind),
        "BOT_NAME_HTML": html.escape(name),
        "BOT_BLURB_HTML": html.escape(blurb),
    }


def planned_paths(bot_id: str, dest_dir) -> List[Path]:
    """The three files :func:`new_bot` would write, in the order it writes
    them: module, page, test.  Exposed so a caller can show a dry run."""
    module = module_name_for(bot_id)
    dest = Path(dest_dir)
    return [
        dest / f"{module}.py",
        dest / f"{bot_id}.html",
        dest / f"test_{module}.py",
    ]


def _read_template(filename: str) -> str:
    path = TEMPLATE_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScaffoldError(f"cannot read template {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------


def new_bot(
    bot_id: str,
    name: str,
    blurb: str = "",
    kind: str = "bot",
    dest_dir=".",
    force: bool = False,
) -> List[Path]:
    """Write a working bot, its page and its test.  Returns the paths written.

    The three files are flat siblings in ``dest_dir`` on purpose: the
    generated test imports the generated module by name, and pytest puts a
    test file's own directory on ``sys.path``, so the set works wherever it is
    dropped as long as the repository root is importable.

    ``force`` overwrites.  Without it, an existing destination is refused
    *before* anything is written, so a refusal never leaves one new file and
    two old ones.
    """
    bot_id = validate_slug(bot_id)
    name = _clean_text(name, "name", fallback=bot_id)
    blurb = _clean_text(
        blurb, "blurb", fallback="Watches something and says when to act."
    )
    kind = _clean_text(kind, "kind", fallback="bot")

    dest = Path(dest_dir)
    if dest.exists() and not dest.is_dir():
        raise ScaffoldError(f"destination is not a directory: {dest}")

    targets = planned_paths(bot_id, dest)
    if not force:
        clashes = [str(p) for p in targets if p.exists()]
        if clashes:
            raise ScaffoldError(
                "refusing to overwrite (use force): " + ", ".join(clashes)
            )

    values = _substitutions(bot_id, name, blurb, kind)
    rendered = [
        render(_read_template("bot.py.tmpl"), values),
        render(_read_template("page.html.tmpl"), values),
        render(_read_template("test.py.tmpl"), values),
    ]

    try:
        dest.mkdir(parents=True, exist_ok=True)
        for path, text in zip(targets, rendered):
            # newline="\n" so a generated file is byte-identical on every
            # platform: the scaffold is deterministic or it is not testable.
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
    except OSError as exc:
        raise ScaffoldError(f"cannot write into {dest}: {exc}") from exc

    return list(targets)
