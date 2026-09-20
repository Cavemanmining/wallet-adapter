"""The generator has to produce working files, not plausible ones.

:mod:`jarvis_bots.scaffold` is the file that makes ``jarvis_bots`` a platform
("adding the second bot should be a class and a page, not a refactor"), so
the bar for it is higher than "the strings came out right":

* the generated module imports, and registers with the real registry;
* the generated test passes, run by a real pytest against the temp
  directory -- if the template rots, this suite goes red;
* the generated page asks nothing of the network and defines every theme
  token it uses in the bare ``:root``, so it renders in a browser with no
  system preference and in an app that blocks third parties;
* refusals are refusals: a bad slug, and an existing file without ``force``.

The generated files go in ``tmp_path`` and the module names they introduce
are cleaned out of ``sys.modules`` afterwards, so one test's bot cannot be
imported by another's.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from jarvis_bots import cli, scaffold
from jarvis_bots.contracts import Severity
from jarvis_bots.scaffold import ScaffoldError, new_bot

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolate_imports():
    """Generated modules live in a temp directory that is about to vanish.

    Leaving them in ``sys.modules`` would let a later test import a file that
    no longer exists, which is the kind of failure that only shows up when
    the tests run in a different order.
    """
    before_modules = set(sys.modules)
    before_path = list(sys.path)
    yield
    for name in set(sys.modules) - before_modules:
        sys.modules.pop(name, None)
    sys.path[:] = before_path


def generate(dest: Path, bot_id: str = "inbox-watch", **kwargs) -> list:
    return new_bot(
        bot_id=bot_id,
        name=kwargs.pop("name", "Inbox watcher"),
        blurb=kwargs.pop("blurb", "Counts what is waiting in the drop folder."),
        kind=kwargs.pop("kind", "radar"),
        dest_dir=dest,
        **kwargs,
    )


def import_generated(path: Path):
    """Import a generated module by its own name, as its test does."""
    sys.path.insert(0, str(path.parent))
    import importlib

    return importlib.import_module(path.stem)


# ---------------------------------------------------------------------------
# what it writes
# ---------------------------------------------------------------------------


def test_new_bot_writes_the_three_files_it_returns(tmp_path):
    paths = generate(tmp_path, "inbox-watch")

    assert [p.name for p in paths] == [
        "inbox_watch.py",
        "inbox-watch.html",
        "test_inbox_watch.py",
    ]
    assert all(p.exists() for p in paths)
    assert paths == scaffold.planned_paths("inbox-watch", tmp_path)
    # Nothing else appears in the destination.
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(p.name for p in paths)


def test_rendering_is_deterministic(tmp_path):
    first = generate(tmp_path / "a", "inbox-watch")
    second = generate(tmp_path / "b", "inbox-watch")

    for left, right in zip(first, second):
        assert left.read_bytes() == right.read_bytes()


def test_no_marker_survives_in_any_generated_file(tmp_path):
    for path in generate(tmp_path, "inbox-watch"):
        assert not re.search(r"@@[A-Z0-9_]+@@", path.read_text(encoding="utf-8"))


def test_the_generated_module_imports_and_is_a_usable_bot(tmp_path):
    module_path, _page, _test = generate(tmp_path, "inbox-watch")

    module = import_generated(module_path)

    assert module.BOT_ID == "inbox-watch"
    assert module.INFO.id == "inbox-watch"
    assert module.INFO.kind == "radar"
    assert module.ATTENTION_KEY == "inbox-watch:over-threshold"

    clock = lambda: 1_700_000_000.0  # noqa: E731 - injected, never time.time
    bot = module.build(clock=clock, watch_dir=str(tmp_path), threshold=1)
    events = list(bot.tick(clock()))

    assert len(events) == 1
    assert events[0].severity is Severity.ACTION
    assert events[0].wants_attention
    assert events[0].attention_key == module.ATTENTION_KEY


def test_the_generated_bot_satisfies_the_registry(tmp_path):
    """check_bot() is the gate every bot passes before it can be run."""
    registry = pytest.importorskip("jarvis_bots.registry")
    module_path, _page, _test = generate(tmp_path, "inbox-watch")
    module = import_generated(module_path)

    bot = module.build(clock=lambda: 0.0, watch_dir=str(tmp_path))
    info = registry.check_bot(bot)

    assert info.id == "inbox-watch"
    assert registry.BotRegistry([bot]).ids() == ["inbox-watch"]


# ---------------------------------------------------------------------------
# the generated test really passes
# ---------------------------------------------------------------------------


def test_the_generated_test_passes_unmodified(tmp_path):
    """Run pytest for real against the generated files.

    In-process, because the point is to prove *these* templates work against
    *this* checkout of the framework; a subprocess would be testing whatever
    happens to be installed.
    """
    _module, _page, test_path = generate(tmp_path, "gen-check")

    code = pytest.main(
        ["-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path), str(test_path)]
    )

    assert int(code) == 0, "the generated test must pass with no edits"


def test_the_generated_test_is_not_vacuous(tmp_path):
    """A test file that collects nothing would also "pass"."""
    _module, _page, test_path = generate(tmp_path, "gen-count")

    body = test_path.read_text(encoding="utf-8")
    names = re.findall(r"(?m)^def (test_\w+)", body)

    assert len(names) >= 6
    assert any("attention" in n for n in names)
    assert any("snapshot" in n for n in names)


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------


def page_of(tmp_path, bot_id="inbox-watch") -> str:
    _module, page, _test = generate(tmp_path, bot_id)
    return page.read_text(encoding="utf-8")


def test_the_page_makes_no_external_request(tmp_path):
    page = page_of(tmp_path)

    assert "http://" not in page
    assert "https://" not in page
    assert "@import" not in page
    assert not re.search(r"<(?:script|img|iframe|source|video)[^>]*\ssrc\s*=", page, re.I)
    assert not re.search(r"<link\b", page, re.I)
    # No protocol-relative escape hatch either.
    assert not re.search(r"""(?:src|href)\s*=\s*["']//""", page)
    # The only url() a self-contained page could need is a data: one, and it
    # needs none at all.
    assert re.findall(r"url\(", page) == []


def test_every_theme_token_is_defined_in_the_bare_root(tmp_path):
    page = page_of(tmp_path)

    bare = re.search(r"(?m)^:root\{\n(.*?)^\}", page, re.S)
    assert bare, "the page must open with a bare :root palette"
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", bare.group(1)))
    used = set(re.findall(r"var\(\s*(--[a-z0-9-]+)", page))

    assert used, "a themed page uses tokens"
    missing = sorted(used - defined)
    assert not missing, f"tokens used but not defined in the bare :root: {missing}"


def test_the_dark_blocks_are_guarded_and_complete(tmp_path):
    page = page_of(tmp_path)

    media = re.search(
        r"@media \(prefers-color-scheme: dark\)\{\n"
        r"  :root:not\(\[data-theme=\"light\"\]\)\{\n(.*?)^  \}",
        page,
        re.S | re.M,
    )
    attr = re.search(r"(?m)^:root\[data-theme=\"dark\"\]\{\n(.*?)^\}", page, re.S)
    assert media, 'the media block must be guarded with :not([data-theme="light"])'
    assert attr, 'the page must honour :root[data-theme="dark"]'

    bare = re.search(r"(?m)^:root\{\n(.*?)^\}", page, re.S)
    bare_tokens = set(re.findall(r"(--[a-z0-9-]+)\s*:", bare.group(1)))
    for block, label in ((media.group(1), "media"), (attr.group(1), "attribute")):
        tokens = set(re.findall(r"(--[a-z0-9-]+)\s*:", block))
        assert tokens, f"the {label} dark block defines nothing"
        assert tokens <= bare_tokens, (
            f"the {label} dark block defines tokens the bare :root does not: "
            f"{sorted(tokens - bare_tokens)}"
        )


def test_the_page_is_readable_on_a_phone(tmp_path):
    page = page_of(tmp_path)

    assert 'name="viewport"' in page
    assert re.search(r"body\{[^}]*background:var\(--bg\)", page, re.S), (
        "body needs an explicit background, or the page is white on white "
        "inside a dark app shell"
    )
    assert "min-height:44px" in page
    assert "font-variant-numeric:tabular-nums" in page


def test_the_page_says_when_it_is_showing_demo_data(tmp_path):
    page = page_of(tmp_path)

    assert "JARVIS_BOT_API" in page
    assert "DEMO_STATE" in page
    assert re.search(r">Demo data<", page)
    # And it states the boundary the framework is built on.
    assert "never buys" in page


def test_page_text_is_escaped_not_injected(tmp_path):
    _module, page, _test = new_bot(
        bot_id="quoted",
        name='Smith & Co "deals"',
        blurb="</script><b>not markup</b>",
        kind="cart",
        dest_dir=tmp_path,
    )
    text = page.read_text(encoding="utf-8")

    assert "Smith &amp; Co &quot;deals&quot;" in text
    assert "</script><b>" not in text
    assert "\\u003c/script\\u003e" in text  # the JS copy is escaped too


def test_python_text_is_a_literal_not_an_injection(tmp_path):
    module_path, _page, _test = new_bot(
        bot_id="quoted",
        name='Smith & Co "deals"',
        blurb="line one\nline two",
        dest_dir=tmp_path,
    )

    module = import_generated(module_path)

    assert module.INFO.name == 'Smith & Co "deals"'
    assert module.INFO.blurb == "line one line two"  # collapsed to one line


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "9lives", "Inbox", "caf\u00e9", "inbox watch", "in--box", "-inbox",
     "class", "inbox/../etc", "x" * 60],
)
def test_an_invalid_slug_is_refused(tmp_path, bad):
    with pytest.raises(ScaffoldError):
        generate(tmp_path, bad)

    assert list(tmp_path.iterdir()) == []


def test_an_existing_file_is_refused_and_nothing_is_written(tmp_path):
    generate(tmp_path, "inbox-watch")
    module_path, page_path, test_path = scaffold.planned_paths("inbox-watch", tmp_path)
    page_path.unlink()  # only one of the three is in the way now
    before = module_path.read_bytes()

    with pytest.raises(ScaffoldError) as caught:
        generate(tmp_path, "inbox-watch")

    assert "refusing to overwrite" in str(caught.value)
    assert not page_path.exists(), "a refusal must not half-generate"
    assert module_path.read_bytes() == before


def test_force_overwrites(tmp_path):
    module_path, page_path, _test = generate(tmp_path, "inbox-watch")
    module_path.write_text("# hand-edited\n", encoding="utf-8")
    page_path.unlink()

    generate(tmp_path, "inbox-watch", force=True)

    assert "# hand-edited" not in module_path.read_text(encoding="utf-8")
    assert page_path.exists()


def test_a_missing_marker_is_caught_rather_than_shipped():
    with pytest.raises(ScaffoldError) as caught:
        scaffold.render("id=@@BOT_ID@@ name=@@BOT_NAME_LITERAL@@", {"BOT_ID": "x"})

    assert "@@BOT_NAME_LITERAL@@" in str(caught.value)


# ---------------------------------------------------------------------------
# two bots
# ---------------------------------------------------------------------------


def test_two_bots_in_one_directory_do_not_collide(tmp_path):
    first = generate(tmp_path, "inbox-watch", name="Inbox watcher")
    second = generate(tmp_path, "spend-bot", name="Spend watcher", kind="cart")

    assert not set(first) & set(second)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "inbox-watch.html",
        "inbox_watch.py",
        "spend-bot.html",
        "spend_bot.py",
        "test_inbox_watch.py",
        "test_spend_bot.py",
    ]

    one = import_generated(first[0])
    two = import_generated(second[0])

    assert one.INFO.id != two.INFO.id
    assert one.ATTENTION_KEY != two.ATTENTION_KEY
    assert one.INFO.href != two.INFO.href
    assert hasattr(one, "InboxWatchBot") and hasattr(two, "SpendBot")

    # And both generated tests pass side by side, which is the case that
    # catches a template with a module-level name two bots would share.
    code = pytest.main(
        ["-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path),
         str(first[2]), str(second[2])]
    )
    assert int(code) == 0


def test_class_names_do_not_stutter():
    assert scaffold.class_name_for("inbox-watch") == "InboxWatchBot"
    assert scaffold.class_name_for("spend-bot") == "SpendBot"
    assert scaffold.class_name_for("radar") == "RadarBot"


# ---------------------------------------------------------------------------
# the CLI wrapper
# ---------------------------------------------------------------------------


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(argv, out=out, err=err, clock=lambda: 1_700_000_000.0)
    return code, out.getvalue(), err.getvalue()


def test_cli_new_bot_prints_what_it_wrote(tmp_path):
    code, out, err = run_cli(
        ["new-bot", "--id", "inbox-watch", "--name", "Inbox watcher",
         "--blurb", "Counts the drop folder.", "--kind", "radar",
         "--dest", str(tmp_path)]
    )

    assert code == 0, err
    written = [line for line in out.splitlines() if line.startswith(str(tmp_path))]
    assert len(written) == 3
    assert all(Path(line).exists() for line in written)


def test_cli_errors_are_one_line_and_never_a_traceback(tmp_path):
    code, out, err = run_cli(
        ["new-bot", "--id", "9lives", "--name", "No", "--dest", str(tmp_path)]
    )

    assert code == 1
    assert out == ""
    assert len(err.strip().splitlines()) == 1
    assert err.startswith("error: ")
    assert "Traceback" not in err


def test_cli_refuses_an_overwrite_then_accepts_force(tmp_path):
    argv = ["new-bot", "--id", "inbox-watch", "--name", "Inbox watcher",
            "--dest", str(tmp_path)]

    assert run_cli(argv)[0] == 0
    code, _out, err = run_cli(argv)
    assert code == 1 and "refusing to overwrite" in err
    assert run_cli(argv + ["--force"])[0] == 0


def test_cli_usage_error_exits_two():
    code, _out, _err = run_cli(["new-bot", "--name", "no id given"])

    assert code == cli.EXIT_USAGE == 2


def test_cli_module_entry_point_runs(tmp_path):
    """``python3 -m jarvis_bots.cli`` is the documented way in, so run it."""
    result = subprocess.run(
        [sys.executable, "-m", "jarvis_bots.cli", "new-bot", "--id", "sub-proc",
         "--name", "Subprocess bot", "--dest", str(tmp_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "sub_proc.py").exists()


# ---------------------------------------------------------------------------
# the generated bot inside the real framework
# ---------------------------------------------------------------------------


def test_a_generated_bot_runs_in_a_real_supervisor(tmp_path):
    supervisor_mod = pytest.importorskip("jarvis_bots.supervisor")
    registry_mod = pytest.importorskip("jarvis_bots.registry")
    module_path, _page, _test = generate(tmp_path / "gen", "inbox-watch")
    module = import_generated(module_path)

    watched = tmp_path / "watched"
    watched.mkdir()
    times = [1_700_000_000.0]
    clock = lambda: times[0]  # noqa: E731

    bot = module.build(clock=clock, watch_dir=str(watched), threshold=2)
    supervisor = supervisor_mod.Supervisor(registry_mod.BotRegistry([bot]), clock)

    report = supervisor.run_round()
    assert report.ticked == 1 and report.failed == 0
    assert supervisor.badge_status() == {"attention": 0, "state": "ok"}

    (watched / "a").write_text("x", encoding="utf-8")
    (watched / "b").write_text("x", encoding="utf-8")
    times[0] += 3600.0
    report = supervisor.run_round()

    assert report.events == 1
    assert supervisor.badge_status()["attention"] == 1
    items = supervisor.attention_items("inbox-watch")
    assert [i.key for i in items] == [module.ATTENTION_KEY]

    # The launcher payload is exactly what web/README.md documents.
    card = supervisor.launcher_state()["bots"][0]
    assert card["id"] == "inbox-watch"
    assert card["state"] == "running"
    assert card["attention"] == 1
    assert [s["label"] for s in card["stats"]] == ["Watching", "Peak"]
    assert json.loads(json.dumps(card)) == card


def test_a_generated_bot_can_clear_its_own_attention(tmp_path):
    """No longer xfail: the cross-lane gap this named is closed.

    ``Supervisor._apply_events`` used to open an attention key and never
    close one, so the generated bot's resolving event (the same key, below
    ACTION, ``data['resolved'] is True``) left the badge item standing for
    ever.  The supervisor now reads that flag -- see
    ``jarvis_bots.supervisor.RESOLVED_FLAG`` -- so a generated bot clears
    its own attention with no app-side help, which is what makes the
    template's convention worth following.
    """
    supervisor_mod = pytest.importorskip("jarvis_bots.supervisor")
    registry_mod = pytest.importorskip("jarvis_bots.registry")
    module_path, _page, _test = generate(tmp_path / "gen", "inbox-watch")
    module = import_generated(module_path)

    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "a").write_text("x", encoding="utf-8")
    (watched / "b").write_text("x", encoding="utf-8")
    times = [1_700_000_000.0]
    clock = lambda: times[0]  # noqa: E731

    bot = module.build(clock=clock, watch_dir=str(watched), threshold=2)
    supervisor = supervisor_mod.Supervisor(registry_mod.BotRegistry([bot]), clock)
    supervisor.run_round()
    assert supervisor.badge_status()["attention"] == 1

    (watched / "a").unlink()
    (watched / "b").unlink()
    times[0] += 3600.0
    supervisor.run_round()

    assert supervisor.badge_status()["attention"] == 0


def test_a_generated_bot_survives_a_supervisor_restart(tmp_path):
    supervisor_mod = pytest.importorskip("jarvis_bots.supervisor")
    registry_mod = pytest.importorskip("jarvis_bots.registry")
    module_path, _page, _test = generate(tmp_path / "gen", "inbox-watch")
    module = import_generated(module_path)

    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "a").write_text("x", encoding="utf-8")
    times = [1_700_000_000.0]
    clock = lambda: times[0]  # noqa: E731
    store_path = str(tmp_path / "state.json")

    def fresh():
        bot = module.build(clock=clock, watch_dir=str(watched), threshold=1)
        store = supervisor_mod.JsonFileStore(store_path)
        return bot, supervisor_mod.Supervisor(registry_mod.BotRegistry([bot]), clock, store)

    bot, supervisor = fresh()
    supervisor.run_round()
    supervisor.save_state()
    assert supervisor.badge_status()["attention"] == 1

    bot2, supervisor2 = fresh()
    assert supervisor2.load_state() is True
    assert bot2.snapshot()["peak"] == 1
    # Already raised before the restart, so the next round must not re-raise.
    times[0] += 3600.0
    assert supervisor2.run_round().events == 0
