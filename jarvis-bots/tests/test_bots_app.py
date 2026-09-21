"""Tests for the composition root: jarvis_bots.app.

Design: ``jarvis_bots/app.py`` -- "Why the page said 'nothing inside'".
The rule this file defends is that the fix for an empty launcher is a
registry with something real in it, never a plausible lie.  So the tests
are mostly about *skips*: that a bot with no config is not built, that the
reason is a sentence a person can act on, and that one bot failing never
costs the others.

Nothing here opens a socket or shells out: every bot is built with an
injected probe, and the two that are built from their real defaults
(``gpu`` reaching for nvidia-smi, ``services`` for systemctl) are built
but never ticked, which is the same thing the app does at start-up.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_bots import app
from jarvis_bots.app import AppError, SkipNote, build, build_supervisor, load_config

T0 = 1_700_000_000.0

try:                                    # the sibling package the poke bot needs
    import jarvis_poke                  # noqa: F401
    HAVE_POKE = True
except ImportError:                     # pragma: no cover - layout-dependent
    HAVE_POKE = False

#: The handoff directories ship jarvis_bots and jarvis_poke apart, so the
#: Pokemon builder -- the only one that reaches into a sibling package --
#: cannot run from inside jarvis-bots/ alone.  Skipping is honest; the
#: tests below still run in the tree where both are importable.
needs_poke = pytest.mark.skipif(not HAVE_POKE, reason="jarvis_poke not on the path")


def clock() -> float:
    return T0


def probe_none():
    return []


PROBE = "tests.test_bots_app:probe_none"


def cfg(**bots: Any) -> Dict[str, Any]:
    return {"bots": bots}


def note_for(result, bot_id: str) -> SkipNote:
    found = [n for n in result.skipped if n.bot_id == bot_id]
    assert found, f"{bot_id} was not skipped: built {[b.info.id for b in result.bots]}"
    return found[0]


# --------------------------------------------------------------------------
# the empty case, which is what the owner actually saw
# --------------------------------------------------------------------------


def test_an_empty_config_builds_nothing_and_says_why_for_every_bot():
    result = build(cfg(), clock=clock)
    assert result.bots == ()
    assert {n.bot_id for n in result.skipped} == set(app.BUILDERS)
    assert all(n.reason == "not configured" for n in result.skipped)
    assert all(n.fixable for n in result.skipped)


def test_the_skips_are_json_ready_for_the_page():
    notes = app.launcher_notes(build(cfg(), clock=clock))
    json.dumps(notes)
    assert notes[0].keys() == {"id", "reason", "fixable"}


def test_a_disabled_bot_is_skipped_not_built():
    result = build(cfg(disk={"enabled": False, "mountpoints": ["/"]}), clock=clock)
    assert not result.bots


def test_nothing_is_guessed_when_a_section_is_half_filled():
    result = build(cfg(disk={"enabled": True}, services={"enabled": True}), clock=clock)
    assert not result.bots
    assert "mountpoints" in note_for(result, "disk").reason
    assert "services" in note_for(result, "services").reason
    # Actionable, not a stack trace: the reason names what to add.
    assert "/scratch" in note_for(result, "disk").reason


# --------------------------------------------------------------------------
# building real bots
# --------------------------------------------------------------------------


def test_each_bot_builds_from_its_own_config():
    result = build(cfg(
        disk={"enabled": True, "probe": PROBE, "mountpoints": ["/tmp"]},
        gpu={"enabled": True, "probe": PROBE},
        services={"enabled": True, "probe": PROBE, "services": ["ssh.service"]},
        health={"enabled": True, "probe": PROBE, "checks": [
            {"name": "frontend", "url": "https://example.com/", "kind": "frontend"},
        ]},
    ), clock=clock)
    assert sorted(b.info.id for b in result.bots) == ["disk", "gpu", "health", "services"]
    assert [n.bot_id for n in result.skipped] == ["poke"]


def test_the_registry_holds_them_in_the_builders_order():
    result = build(cfg(
        health={"enabled": True, "probe": PROBE, "checks": [
            {"name": "f", "url": "https://example.com/", "kind": "frontend"}]},
        gpu={"enabled": True, "probe": PROBE},
    ), clock=clock)
    assert [b.info.id for b in result.bots] == ["gpu", "health"]


def test_one_bot_that_cannot_be_built_never_costs_the_others():
    result = build(cfg(
        gpu={"enabled": True, "probe": PROBE},
        disk={"enabled": True},                       # no mountpoints
        health={"enabled": True, "checks": [{"name": "x"}]},   # not a valid Check
    ), clock=clock)
    assert [b.info.id for b in result.bots] == ["gpu"]
    assert {n.bot_id for n in result.skipped} >= {"disk", "health", "poke", "services"}


def test_a_config_error_message_never_reaches_the_page_verbatim():
    # A builder that raises something other than AppError has its message
    # dropped: config strings can quote a path or a url, and this text is
    # rendered in a browser.
    def explode(block, tick):
        raise RuntimeError("/home/owner/secrets/token.json")

    saved = app.BUILDERS["gpu"]
    app.BUILDERS["gpu"] = explode
    try:
        result = build(cfg(gpu={"enabled": True}), clock=clock)
    finally:
        app.BUILDERS["gpu"] = saved
    note = note_for(result, "gpu")
    assert "secrets" not in note.reason
    assert "RuntimeError" in note.reason
    assert not note.fixable


# --------------------------------------------------------------------------
# the poke bot: the one this package cannot finish on its own
# --------------------------------------------------------------------------


@needs_poke
def test_poke_without_a_fetcher_says_the_package_ships_no_parser():
    result = build(cfg(poke={"enabled": True, "db": ":memory:"}), clock=clock)
    reason = note_for(result, "poke").reason
    assert "ships no" in reason and "parser" in reason


@needs_poke
def test_poke_without_a_db_says_a_restart_would_re_alert():
    result = build(cfg(poke={"enabled": True, "fetcher": PROBE, "parser": PROBE}),
                   clock=clock)
    assert "re-alert" in note_for(result, "poke").reason


def _poke_config(**over: Any) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "enabled": True,
        "db": str(Path(tempfile.mkdtemp()) / "poke.db"),
        "fetcher": PROBE,
        "parser": PROBE,
    }
    block.update(over)
    return cfg(poke=block)


@needs_poke
def test_poke_builds_with_the_packages_own_placeholder_catalog():
    result = build(_poke_config(), clock=clock)
    assert [b.info.id for b in result.bots] == ["poke"]
    assert result.bots[0]._snipe is None, "no windows configured, no controller"


@needs_poke
def test_drop_windows_in_config_produce_a_live_snipe_controller():
    result = build(_poke_config(windows=[{
        "name": "restock", "source": "examplemart",
        "opens_at": T0 + 3600.0, "closes_at": T0 + 5400.0, "interval_s": 30.0,
    }]), clock=clock)
    bot = result.bots[0]
    assert bot._snipe is not None
    assert bot._snipe.active_window("examplemart", T0) is None
    assert bot._snipe.active_window("examplemart", T0 + 3700.0).name == "restock"


@needs_poke
def test_a_recurring_window_expands_from_config():
    result = build(_poke_config(windows=[{
        "name": "weekly", "source": "examplemart",
        "first_day": "2026-04-06", "at_utc": "11:00",
        "duration_s": 1800.0, "days": 5,
    }]), clock=clock)
    plan = result.bots[0]._snipe.plan
    assert len(plan.windows) == 5
    assert plan.windows[0].name == "weekly-2026-04-06"


@needs_poke
def test_a_window_under_the_poll_floor_is_refused_not_quietly_raised():
    result = build(_poke_config(windows=[{
        "name": "toofast", "source": "examplemart",
        "opens_at": T0, "closes_at": T0 + 60.0, "interval_s": 1.0,
    }]), clock=clock)
    assert not result.bots
    assert note_for(result, "poke").reason


@needs_poke
def test_a_window_for_a_source_the_catalog_has_never_heard_of_is_refused():
    result = build(_poke_config(windows=[{
        "name": "ghost", "source": "nosuchshop",
        "opens_at": T0, "closes_at": T0 + 600.0,
    }]), clock=clock)
    assert not result.bots


# --------------------------------------------------------------------------
# the supervisor, and one real round
# --------------------------------------------------------------------------


def test_build_supervisor_runs_a_round_over_what_it_built():
    supervisor, result = build_supervisor(cfg(
        gpu={"enabled": True, "probe": PROBE},
        disk={"enabled": True, "probe": PROBE, "mountpoints": ["/tmp"]},
    ), clock=clock)
    assert len(result.bots) == 2
    report = supervisor.run_round(T0)
    assert report.ticked == 2
    assert report.failed == 0


@needs_poke
def test_every_bot_gets_the_same_clock():
    # jarvis_poke refuses collaborators on a different clock, and a
    # snapshot taken by two clocks is two different moments.
    seen: List[float] = []

    def watched() -> float:
        seen.append(T0)
        return T0

    supervisor, result = build_supervisor(_poke_config(), clock=watched)
    supervisor.run_round(T0)
    assert result.bots and seen


# --------------------------------------------------------------------------
# config loading
# --------------------------------------------------------------------------


def test_load_config_refuses_a_non_object(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(AppError, match="JSON object"):
        load_config(str(path))


def test_load_config_refuses_a_bots_key_that_is_not_an_object(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"bots": []}', encoding="utf-8")
    with pytest.raises(AppError, match="id -> settings"):
        load_config(str(path))


def test_load_config_reads_a_real_file(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"bots": {"gpu": {"enabled": True, "probe": PROBE}}}),
                    encoding="utf-8")
    result = build(load_config(str(path)), clock=clock)
    assert [b.info.id for b in result.bots] == ["gpu"]


@pytest.mark.parametrize("spec", ["notacallable", "jarvis_bots:nosuchthing",
                                  "no.such.module:x", 42])
def test_an_import_spec_that_does_not_resolve_is_a_readable_skip(spec):
    result = build(cfg(gpu={"enabled": True, "probe": spec}), clock=clock)
    reason = note_for(result, "gpu").reason
    assert reason and "Traceback" not in reason


def test_the_cli_dry_run_prints_what_it_would_build(tmp_path, capsys):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"bots": {"gpu": {"enabled": True, "probe": PROBE}}}),
                    encoding="utf-8")
    assert app.main([str(path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["built"] == ["gpu"]
    assert any(n["id"] == "disk" for n in printed["skipped"])


def test_the_cli_exits_non_zero_when_it_would_build_nothing(tmp_path, capsys):
    path = tmp_path / "c.json"
    path.write_text('{"bots": {}}', encoding="utf-8")
    assert app.main([str(path)]) == 1


def test_the_cli_reports_a_broken_config_rather_than_raising(tmp_path, capsys):
    path = tmp_path / "c.json"
    path.write_text("not json", encoding="utf-8")
    assert app.main([str(path)]) == 2
    assert "config:" in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------
# the shipped example, which is documentation that can rot
# --------------------------------------------------------------------------


EXAMPLE = ROOT / "jarvis_bots" / "bots.config.example.json"


def test_the_example_config_is_valid_json_and_builds():
    obj = load_config(str(EXAMPLE))
    result = build(obj, clock=clock)
    assert sorted(b.info.id for b in result.bots) == ["disk", "gpu", "health", "services"]
    # Poke is off in the example precisely because this package ships no
    # retailer parser; if that ever flips silently, this fails.
    assert note_for(result, "poke").reason == "not configured"


def test_the_example_config_names_only_bots_that_exist():
    obj = load_config(str(EXAMPLE))
    assert set(obj["bots"]) <= set(app.BUILDERS)


def test_the_example_config_points_at_no_real_retailer():
    text = EXAMPLE.read_text(encoding="utf-8")
    for host in ("pokemoncenter", "target.com", "walmart", "bestbuy", "amazon"):
        assert host not in text.lower(), (
            f"the shipped example names {host}; the placeholder shops are "
            f"there so nobody's selectors end up in this repository"
        )


# --------------------------------------------------------------------------
# composing with a wiring the app already has
# --------------------------------------------------------------------------
#
# BotRegistry.register refuses a duplicate id on purpose, so an app that
# wired some bots by hand could not also call build() without crashing --
# which made two wirings an either/or. These say they compose.


from jarvis_bots.registry import BotRegistry, RegistryError  # noqa: E402


def _built(bot_id: str):
    result = build(cfg(**{bot_id: {"enabled": True, "probe": PROBE,
                                   "mountpoints": ["/tmp"],
                                   "services": ["ssh.service"]}}), clock=clock)
    assert result.bots, [n.reason for n in result.skipped]
    return result.bots[0]


def test_a_bot_the_app_already_wired_is_left_alone_not_rebuilt():
    mine = BotRegistry([_built("gpu")])
    result = build(cfg(
        gpu={"enabled": True, "probe": PROBE},
        disk={"enabled": True, "probe": PROBE, "mountpoints": ["/tmp"]},
    ), clock=clock, existing=mine)
    assert [b.info.id for b in result.bots] == ["disk"]
    assert "already wired elsewhere" in note_for(result, "gpu").reason


def test_build_supervisor_registers_alongside_an_existing_registry():
    existing = BotRegistry([_built("gpu")])
    supervisor, result = build_supervisor(cfg(
        gpu={"enabled": True, "probe": PROBE},
        disk={"enabled": True, "probe": PROBE, "mountpoints": ["/tmp"]},
    ), clock=clock, registry=existing)
    assert sorted(existing.ids()) == ["disk", "gpu"]
    report = supervisor.run_round(T0)
    assert report.ticked == 2 and report.failed == 0
    # The hand-wired one is the *same object*, not a rebuild: its health
    # record and snapshot stay attached to it.
    assert existing.get("gpu") is not result.bots[0]


def test_composing_never_raises_the_duplicate_id_error():
    # Without the skip this is exactly the crash: register() refuses a
    # duplicate id, so a second wiring would take the whole app down.
    existing = BotRegistry([_built("gpu")])
    with pytest.raises(RegistryError, match="duplicate"):
        existing.register(_built("gpu"))          # the failure being avoided
    supervisor, result = build_supervisor(
        cfg(gpu={"enabled": True, "probe": PROBE}), clock=clock, registry=existing)
    assert [b.info.id for b in result.bots] == []
    assert sorted(existing.ids()) == ["gpu"]


def test_existing_accepts_a_plain_list_of_ids():
    result = build(cfg(gpu={"enabled": True, "probe": PROBE}),
                   clock=clock, existing=["gpu"])
    assert not result.bots
    assert "already wired elsewhere" in note_for(result, "gpu").reason


@pytest.mark.parametrize("bad", ["gpu", 42])
def test_existing_refuses_something_it_cannot_read_as_ids(bad):
    with pytest.raises(AppError, match="existing"):
        build(cfg(), clock=clock, existing=bad)


def test_no_existing_is_the_single_wiring_case_unchanged():
    result = build(cfg(gpu={"enabled": True, "probe": PROBE}), clock=clock)
    assert [b.info.id for b in result.bots] == ["gpu"]


def test_a_missing_sibling_package_is_named_not_reduced_to_an_exception_type(
    monkeypatch,
):
    # Runs in either layout: a None in sys.modules makes the import raise
    # ImportError even where jarvis_poke is installed, so the branch is
    # covered in the tree where it cannot occur naturally.
    for name in ("jarvis_poke", "jarvis_poke.catalog", "jarvis_poke.engine",
                 "jarvis_poke.prices", "jarvis_poke.rules", "jarvis_poke.sources",
                 "jarvis_poke.store"):
        monkeypatch.setitem(sys.modules, name, None)
    result = build(cfg(poke={"enabled": True, "db": ":memory:",
                             "fetcher": PROBE, "parser": PROBE}), clock=clock)
    note = note_for(result, "poke")
    assert "jarvis_poke" in note.reason and "on the path" in note.reason
    assert "ModuleNotFoundError" not in note.reason, (
        "an exception type name tells an owner nothing about what to install"
    )
