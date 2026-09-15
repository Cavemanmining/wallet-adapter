"""Tests for the Descent table view (``lucifer_descent.render``) and the
command line tool (``lucifer_descent.cli``).

Spec: docs/WORLD_BIBLE.md section 03 -- "States and colours: locked grey,
reachable blue, active amber, cleared green, failed red. Colour is derived
from state, never stored."

The render tests decode the PNG with :func:`lucifer_gen.png.decode_png` and
look for colours in the pixels: a state's RGB must be there exactly when a
node in that state exists.  The CLI tests drive ``main`` in-process on a
temporary sqlite file with a stub map probe injected through the module flag
``lucifer_descent.cli.MAP_PROBE``, and once through a real subprocess with
the ``LUCIFER_DESCENT_PROBE`` environment variable, so both documented hooks
are exercised.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from lucifer_descent import cli, render
from lucifer_descent.contracts import (
    Mechanic,
    NodeState,
    Pinnacle,
    ProfileState,
    Web,
    WebEdge,
    WebNode,
)
from lucifer_descent.engine import DescentEngine, MapProbe
from lucifer_descent.store import SqliteStore
from lucifer_descent.web import generate_web
from lucifer_gen.png import decode_png, iter_chunks

RGB = Tuple[int, int, int]
SEED = 0x5EED
SIGIL_RE = re.compile(r"sg-[0-9a-f]{8}")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def small_state(profile_id: str = "alice", rings: int = 4) -> ProfileState:
    """A fresh profile on a real generated web, small enough to render fast."""
    return DescentEngine.new_profile(profile_id, generate_web(SEED, rings))


def plain_web() -> Web:
    """A hand-built web with no mechanics, arenas or glyphs at all."""
    nodes = (
        WebNode(id=0, tier=0, ring_index=0, template="crypt", x=0.0, y=0.0),
        WebNode(id=1, tier=1, ring_index=0, template="crypt", x=10.0, y=0.0),
        WebNode(id=2, tier=1, ring_index=1, template="crypt", x=-10.0, y=0.0),
        WebNode(id=3, tier=2, ring_index=0, template="crypt", x=0.0, y=20.0),
    )
    edges = (WebEdge(0, 1), WebEdge(0, 2), WebEdge(1, 3), WebEdge(2, 3))
    return Web(profile_seed=7, origin_id=0, nodes=nodes, edges=edges)


def colours_in(data: bytes) -> Set[RGB]:
    _, _, rows = decode_png(data)
    seen: Set[RGB] = set()
    for row in rows:
        seen.update(zip(row[0::3], row[1::3], row[2::3]))
    return seen


def state_colours(seen: Set[RGB]) -> Set[NodeState]:
    return {s for s in NodeState if render.state_rgb(s) in seen}


class StubProbe:
    def __init__(self, has_boss: bool, elite_total: int) -> None:
        self.answer = MapProbe(has_boss=has_boss, elite_total=elite_total)
        self.calls: List[Tuple[str, int, int]] = []

    def __call__(self, template: str, map_seed: int, sigil_tier: int) -> MapProbe:
        self.calls.append((template, map_seed, sigil_tier))
        return self.answer


# --------------------------------------------------------------------------
# render.py
# --------------------------------------------------------------------------


def test_png_is_structurally_valid_and_square(tmp_path: Path) -> None:
    out = tmp_path / "web.png"
    written = render.render_web(small_state(), out, size=300)
    data = out.read_bytes()
    assert written == len(data)
    tags = [tag for tag, _ in iter_chunks(data)]  # checks signature and CRCs
    assert tags == [b"IHDR", b"IDAT", b"IEND"]
    width, height, rows = decode_png(data)
    assert (width, height) == (300, 300)
    assert len(rows) == 300 and all(len(r) == 300 * 3 for r in rows)


def test_default_size_is_900(tmp_path: Path) -> None:
    out = tmp_path / "web.png"
    render.render_web(small_state(rings=2), out)
    width, height, _ = decode_png(out.read_bytes())
    assert (width, height) == (render.DEFAULT_SIZE, render.DEFAULT_SIZE) == (900, 900)


def test_render_is_byte_deterministic(tmp_path: Path) -> None:
    state = small_state()
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    render.render_web(state, a, size=320)
    render.render_web(state, b, size=320)
    assert a.read_bytes() == b.read_bytes()
    # The in-memory writer and the file writer draw through the same function.
    assert render.render_web_bytes(state, size=320) == a.read_bytes()
    # A fresh, independently generated state renders identically too.
    assert render.render_web_bytes(small_state(), size=320) == a.read_bytes()


def test_all_colours_are_distinct() -> None:
    every = list(render.STATE_RGB.values()) + list(render.PALETTE.values())
    assert len(every) == len(set(every))
    for state in NodeState:
        assert render.state_rgb(state) in render.STATE_RGB.values()


@pytest.mark.parametrize("only", list(NodeState))
def test_state_colour_appears_exactly_when_a_node_has_that_state(only: NodeState) -> None:
    state = small_state()
    for node_id in state.states:
        state.states[node_id] = only
    seen = state_colours(colours_in(render.render_web_bytes(state, size=300)))
    assert seen == {only}


def test_fresh_profile_shows_grey_blue_green_and_no_amber_or_red() -> None:
    state = small_state()
    seen = state_colours(colours_in(render.render_web_bytes(state, size=300)))
    assert seen == {NodeState.LOCKED, NodeState.REACHABLE, NodeState.CLEARED}

    reachable = state.reachable_ids()
    state.states[reachable[0]] = NodeState.ACTIVE
    state.states[reachable[1]] = NodeState.FAILED
    seen = state_colours(colours_in(render.render_web_bytes(state, size=300)))
    assert seen == set(NodeState)


def test_marks_outlines_and_stamp_are_drawn() -> None:
    state = small_state()
    web = state.web
    assert any(n.mechanic is not None for n in web.nodes)
    assert any(n.pinnacle is Pinnacle.ARBITER for n in web.nodes)
    assert any(n.glyph is Pinnacle.MONOLITH for n in web.nodes)
    seen = colours_in(render.render_web_bytes(state, size=400))
    for name in ("mark", "text", "edge", "ring_guide", "arbiter", "monolith", "background"):
        assert render.PALETTE[name] in seen, name


def test_plain_web_has_no_marks_or_pinnacle_outlines() -> None:
    state = DescentEngine.new_profile("bob", plain_web())
    seen = colours_in(render.render_web_bytes(state, size=200))
    assert render.PALETTE["mark"] not in seen
    assert render.PALETTE["arbiter"] not in seen
    assert render.PALETTE["monolith"] not in seen
    assert render.PALETTE["edge"] in seen
    assert render.PALETTE["text"] in seen
    assert state_colours(seen) == {NodeState.CLEARED, NodeState.REACHABLE, NodeState.LOCKED}


def test_each_mechanic_and_both_outlines_render_on_a_hand_built_web() -> None:
    nodes = [WebNode(id=0, tier=0, ring_index=0, template="crypt")]
    for i, mechanic in enumerate(Mechanic, start=1):
        nodes.append(
            WebNode(id=i, tier=1, ring_index=i, template="crypt", mechanic=mechanic,
                    x=30.0 * i - 75.0, y=-30.0)
        )
    nodes.append(WebNode(id=9, tier=1, ring_index=9, template="crypt", pinnacle=Pinnacle.MONOLITH, x=-30.0, y=30.0))
    nodes.append(WebNode(id=10, tier=1, ring_index=10, template="crypt", glyph=Pinnacle.ARBITER, x=30.0, y=30.0))
    edges = tuple(WebEdge(0, n.id) for n in nodes[1:])
    web = Web(profile_seed=3, origin_id=0, nodes=tuple(nodes), edges=edges)
    state = DescentEngine.new_profile("carol", web)
    seen = colours_in(render.render_web_bytes(state, size=240))
    assert render.PALETTE["mark"] in seen
    assert render.PALETTE["monolith"] in seen
    assert render.PALETTE["arbiter"] in seen


def test_origin_is_larger_and_centred() -> None:
    state = small_state()
    geo = render.geometry_for(state, 300)
    origin = state.web.node(state.web.origin_id)
    assert geo.origin_radius > geo.node_radius
    assert geo.pixel(origin) == (150, 150)
    assert sorted(geo.ring_radius) == [1, 2, 3, 4]
    assert geo.ring_radius[1] < geo.ring_radius[2] < geo.ring_radius[3] < geo.ring_radius[4]
    assert geo.ring_radius[4] + geo.node_radius < 150


def test_every_node_lands_inside_the_image() -> None:
    state = small_state()
    for size in (64, 300, 900):
        geo = render.geometry_for(state, size)
        for node in state.web.nodes:
            x, y = geo.pixel(node)
            r = geo.radius_of(node, state.web.origin_id) + geo.outline_gap + geo.outline_thickness + 2
            assert r <= x < size - r and r <= y < size - r, (size, node.id)


def test_too_small_a_size_is_refused() -> None:
    with pytest.raises(ValueError):
        render.render_web_bytes(small_state(), size=32)


def test_missing_node_state_is_a_clear_error() -> None:
    state = small_state()
    del state.states[3]
    with pytest.raises(ValueError, match="node 3"):
        render.render_web_bytes(state, size=200)


# --------------------------------------------------------------------------
# cli.py: the probe hooks
# --------------------------------------------------------------------------


def test_parse_probe_spec() -> None:
    assert cli.parse_probe_spec("") is cli.default_map_probe
    assert cli.parse_probe_spec("real") is cli.default_map_probe
    assert cli.parse_probe_spec("boss")("crypt", 1, 1) == MapProbe(True, 0)
    assert cli.parse_probe_spec("BOSS:3")("crypt", 1, 1) == MapProbe(True, 3)
    assert cli.parse_probe_spec("noboss:5")("crypt", 1, 1) == MapProbe(False, 5)
    for bad in ("bogus", "boss:x", "noboss:-1"):
        with pytest.raises(cli.CliError):
            cli.parse_probe_spec(bad)


def test_resolve_probe_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = StubProbe(True, 1)
    monkeypatch.setenv(cli.PROBE_ENV, "noboss:2")
    monkeypatch.setattr(cli, "MAP_PROBE", stub)
    assert cli.resolve_probe() is stub
    monkeypatch.setattr(cli, "MAP_PROBE", None)
    assert cli.resolve_probe()("crypt", 1, 1) == MapProbe(False, 2)
    monkeypatch.delenv(cli.PROBE_ENV)
    assert cli.resolve_probe() is cli.default_map_probe


# --------------------------------------------------------------------------
# cli.py: new / grant / open / event / show / render, in process
# --------------------------------------------------------------------------


class Shell:
    """Runs ``cli.main`` against one temp db and captures its output."""

    def __init__(self, db: Path, capsys: pytest.CaptureFixture) -> None:
        self.db = str(db)
        self.capsys = capsys

    def __call__(self, *argv: str) -> Tuple[int, str, str]:
        rc = cli.main(["--db", self.db, *argv])
        captured = self.capsys.readouterr()
        return rc, captured.out, captured.err

    def ok(self, *argv: str) -> str:
        rc, out, err = self(*argv)
        assert rc == 0, err
        assert err == ""
        return out

    def fails(self, *argv: str) -> str:
        rc, out, err = self(*argv)
        assert rc == 1
        assert out == ""
        assert err.startswith("error: ")
        assert err.count("\n") == 1
        assert "Traceback" not in err
        return err


@pytest.fixture
def shell(tmp_path: Path, capsys: pytest.CaptureFixture) -> Shell:
    return Shell(tmp_path / "descent.sqlite3", capsys)


def test_cli_end_to_end(shell: Shell, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = StubProbe(has_boss=False, elite_total=5)
    monkeypatch.setattr(cli, "MAP_PROBE", probe)

    out = shell.ok("new", "--profile", "alice", "--seed", "0x5EED", "--rings", "3")
    assert out.startswith("new: profile alice seed 0x0000000000005EED nodes 31")
    assert "reachable 1 2 3 4 5 6 7 8" in out
    assert out.count("\n") == 1

    err = shell.fails("new", "--profile", "alice", "--seed", "0x5EED")
    assert "already exists" in err

    out = shell.ok("grant", "--profile", "alice", "--tier", "3", "--count", "2")
    first, second = SIGIL_RE.findall(out)
    assert first != second
    assert out.startswith("grant: 2 tier-3 sigil(s)")

    out = shell.ok("show", "--profile", "alice")
    assert "stash: 2 sigil(s)  tier 3 x2" in out
    assert first in out and second in out
    assert "instance: none" in out
    assert "states: locked 22  reachable 8  active 0  cleared 1  failed 0" in out

    # Nothing is live yet, so every report is refused in one line.
    shell.fails("event", "--profile", "alice", "--kind", "boss")

    out = shell.ok("open", "--profile", "alice", "--node", "1", "--sigil", first)
    assert out.startswith("open: node 1 (crypt tier 1) with sigil " + first)
    assert "boss no; elite packs 5" in out
    assert probe.calls == [("crypt", _seed_of(shell, first, "alice"), 3)]

    # One instance at a time; a locked node; an unknown Sigil.
    shell.fails("open", "--profile", "alice", "--node", "2", "--sigil", second)
    for i in range(3):
        out = shell.ok("event", "--profile", "alice", "--kind", "elite")
        assert f"still active, elites {i + 1}/5" in out
    out = shell.ok("event", "--profile", "alice", "--kind", "elite")  # 4/5 = 80 percent
    assert "node 1; cleared" in out and f"sigil {first} consumed" in out
    assert "newly reachable" in out

    out = shell.ok("show", "--profile", "alice")
    assert "cleared 2" in out and "instance: none" in out and "stash: 1 sigil(s)" in out

    shell.fails("open", "--profile", "alice", "--node", "20", "--sigil", second)  # locked
    shell.fails("open", "--profile", "alice", "--node", "2", "--sigil", "sg-00000000")

    out = shell.ok("open", "--profile", "alice", "--node", "2", "--sigil", second)
    out = shell.ok("event", "--profile", "alice", "--kind", "died")
    assert "node 2; failed" in out and f"sigil {second} consumed" in out
    out = shell.ok("show", "--profile", "alice")
    assert "failed: 2" in out and "stash: 0 sigil(s)" in out

    # A consumed id is never minted again; the retry reopens the failed node.
    out = shell.ok("grant", "--profile", "alice", "--tier", "3")
    (third,) = SIGIL_RE.findall(out)
    assert third not in (first, second)
    shell.ok("open", "--profile", "alice", "--node", "2", "--sigil", third)
    out = shell.ok("event", "--profile", "alice", "--kind", "timeout")
    assert "node 2; failed" in out

    # The picture reflects the saved states.
    png = tmp_path / "web.png"
    out = shell.ok("render", "--profile", "alice", "--out", str(png), "--size", "300")
    assert out.startswith(f"render: wrote {png} (300x300, ")
    seen = state_colours(colours_in(png.read_bytes()))
    assert seen == {NodeState.LOCKED, NodeState.REACHABLE, NodeState.CLEARED, NodeState.FAILED}

    # And the store holds what the commands said.
    with SqliteStore(shell.db) as store:
        state = store.load("alice")
    assert state is not None
    assert state.states[1] is NodeState.CLEARED
    assert state.states[2] is NodeState.FAILED
    assert state.instance is None and state.stash == {}
    assert [e.sigil_id for e in state.history if e.sigil_id] == [first, first, second, second, third, third]


def _seed_of(shell: Shell, sigil: str, profile: str) -> int:
    """Read a stashed Sigil's seed back off ``show``, or the ledger if it is spent."""
    with SqliteStore(shell.db) as store:
        state = store.load(profile)
    assert state is not None
    if sigil in state.stash:
        return state.stash[sigil].seed
    assert state.instance is not None and state.instance.sigil.id == sigil
    return state.instance.sigil.seed


def test_cli_boss_kill_clears_and_grants_rewards(shell: Shell, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "MAP_PROBE", StubProbe(has_boss=True, elite_total=4))
    shell.ok("new", "--profile", "bob", "--seed", "0x5EED", "--rings", "2")
    with SqliteStore(shell.db) as store:
        state = store.load("bob")
    assert state is not None
    mechanic = next(n.id for n in state.web.nodes if n.tier == 1 and n.mechanic is not None)

    out = shell.ok("grant", "--profile", "bob", "--tier", "1")
    (sigil,) = SIGIL_RE.findall(out)
    shell.ok("open", "--profile", "bob", "--node", str(mechanic), "--sigil", sigil)
    # Elites never clear a map that has a boss.
    for _ in range(4):
        assert "still active" in shell.ok("event", "--profile", "bob", "--kind", "elite")
    out = shell.ok("event", "--profile", "bob", "--kind", "boss")
    assert f"node {mechanic}; cleared" in out
    assert "+1 passive point(s), total 1" in out
    assert "points: 1" in shell.ok("show", "--profile", "bob")


def test_cli_abandon_behaves_like_death(shell: Shell, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "MAP_PROBE", StubProbe(has_boss=True, elite_total=0))
    shell.ok("new", "--profile", "dan", "--seed", "0x10", "--rings", "2")
    (sigil,) = SIGIL_RE.findall(shell.ok("grant", "--profile", "dan", "--tier", "2"))
    shell.ok("open", "--profile", "dan", "--node", "3", "--sigil", sigil)
    assert "node 3; failed" in shell.ok("event", "--profile", "dan", "--kind", "abandon")
    assert "failed: 3" in shell.ok("show", "--profile", "dan")


def test_cli_tick_option_and_default(shell: Shell) -> None:
    shell.ok("new", "--profile", "eve", "--seed", "1", "--rings", "2", "--tick", "42")
    with SqliteStore(shell.db) as store:
        state = store.load("eve")
    assert state is not None and state.history
    assert {e.tick for e in state.history} == {42}
    shell.ok("grant", "--profile", "eve", "--tier", "5", "--count", "3")
    with SqliteStore(shell.db) as store:
        state = store.load("eve")
    assert state is not None and len(state.stash) == 3


def test_cli_grant_and_new_validation(shell: Shell) -> None:
    shell.fails("show", "--profile", "nobody")
    shell.ok("new", "--profile", "fay", "--seed", "0xFF", "--rings", "2")
    assert "--tier" in shell.fails("grant", "--profile", "fay", "--tier", "16")
    assert "--count" in shell.fails("grant", "--profile", "fay", "--tier", "1", "--count", "0")
    shell.fails("new", "--profile", "gus", "--seed", "1", "--rings", "99")  # web.py refuses
    out = shell.ok("new", "--profile", "fay", "--seed", "0xFE", "--rings", "2", "--force")
    assert "seed 0x00000000000000FE" in out


def test_cli_usage_error_exits_2(shell: Shell) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--db", shell.db, "event", "--profile", "x", "--kind", "explode"])
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------
# cli.py: the gate, against a stub validate module
# --------------------------------------------------------------------------


def _stub_validate(monkeypatch: pytest.MonkeyPatch, result: object) -> List[Tuple[int, int]]:
    calls: List[Tuple[int, int]] = []
    module = types.ModuleType("lucifer_descent.validate")

    def run_suite(profiles: int, start_seed: int):
        calls.append((profiles, start_seed))
        return result

    module.run_suite = run_suite  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lucifer_descent.validate", module)
    return calls


@pytest.mark.parametrize(
    "result, expected",
    [
        (True, 0),
        (False, 1),
        ([], 0),
        (["tier mismatch on seed 3"], 1),
        (types.SimpleNamespace(ok=True, summary="all good"), 0),
        (types.SimpleNamespace(ok=False, failures=["x"]), 1),
        ({"passed": True}, 0),
        ((True, "5 profiles"), 0),
        (0, 0),
        (2, 1),
        (None, 0),
        (object(), 1),
    ],
)
def test_cli_gate_exit_status(shell: Shell, monkeypatch: pytest.MonkeyPatch, result: object, expected: int) -> None:
    calls = _stub_validate(monkeypatch, result)
    rc, out, err = shell("gate", "--profiles", "3", "--start-seed", "0x10")
    assert rc == expected
    assert calls == [(3, 0x10)]
    assert out.startswith("gate: PASS" if expected == 0 else "gate: FAIL")
    assert err == ""


def test_cli_gate_without_validate_module_is_one_line(shell: Shell, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "lucifer_descent.validate", None)  # forces ImportError
    err = shell.fails("gate", "--profiles", "1", "--start-seed", "1")
    assert "gate unavailable" in err


# --------------------------------------------------------------------------
# cli.py: audit and refusal of a tampered file
# --------------------------------------------------------------------------


def test_cli_check_audits_a_stored_profile_and_play_refuses_a_tampered_web(
    shell: Shell, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3
    monkeypatch.setattr(cli, "MAP_PROBE", StubProbe(has_boss=True, elite_total=3))
    shell.ok("new", "--profile", "ivy", "--seed", "0x5EED")   # full size: the gate holds arenas to tier 15
    (sigil,) = SIGIL_RE.findall(shell.ok("grant", "--profile", "ivy", "--tier", "1"))
    shell.ok("open", "--profile", "ivy", "--node", "1", "--sigil", sigil)
    out = shell.ok("check", "--profile", "ivy")
    assert out.strip().endswith("check: ivy OK; ledger 9 entries")
    shell.fails("check", "--profile", "nobody")

    # Edit the live instance under the engine: the next report is refused,
    # and check names the disagreement.
    raw = sqlite3.connect(shell.db)
    raw.execute("UPDATE instance SET has_boss = 0, elite_total = 0")
    raw.commit()
    raw.close()
    err = shell.fails("event", "--profile", "ivy", "--kind", "elite")
    assert "ledger's open" in err
    rc, out, _ = shell("check", "--profile", "ivy")
    assert rc == 1 and "instance_disagrees_with_ledger" in out and "problem(s)" in out

    # Edit the web: every playing command refuses, check reports provenance.
    raw = sqlite3.connect(shell.db)
    raw.execute("UPDATE instance SET has_boss = 1, elite_total = 3")
    raw.execute("UPDATE web_nodes SET mechanic = 'dig'")
    raw.commit()
    raw.close()
    for argv in (("show", "--profile", "ivy"), ("event", "--profile", "ivy", "--kind", "boss"), ("grant", "--profile", "ivy", "--tier", "1")):
        assert "not the web seed" in shell.fails(*argv)
    rc, out, _ = shell("check", "--profile", "ivy")
    assert rc == 1 and "web_not_from_seed" in out


def test_cli_open_reports_a_map_that_cleared_on_open(shell: Shell, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "MAP_PROBE", StubProbe(has_boss=False, elite_total=0))
    shell.ok("new", "--profile", "kim", "--seed", "0x5EED")
    (sigil,) = SIGIL_RE.findall(shell.ok("grant", "--profile", "kim", "--tier", "1"))
    out = shell.ok("open", "--profile", "kim", "--node", "1", "--sigil", sigil)
    assert "cleared on open" in out
    assert "no instance is active" in shell.fails("event", "--profile", "kim", "--kind", "elite")
    assert "cleared 2" in shell.ok("show", "--profile", "kim")
    assert "OK; ledger" in shell.ok("check", "--profile", "kim")


def test_cli_gate_does_not_open_the_profile_database(shell: Shell, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3
    raw = sqlite3.connect(shell.db)
    raw.execute("CREATE TABLE schema_version (id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)")
    raw.execute("INSERT INTO schema_version (id, version) VALUES (1, 1)")
    raw.commit()
    raw.close()
    _stub_validate(monkeypatch, True)
    rc, out, err = shell("gate", "--profiles", "1", "--start-seed", "1")
    assert rc == 0 and out.startswith("gate: PASS") and err == ""
    assert "schema version 1" in shell.fails("show", "--profile", "x")


# --------------------------------------------------------------------------
# python3 -m lucifer_descent.cli, for real, with the environment hook
# --------------------------------------------------------------------------


def test_module_entry_point_with_env_probe(tmp_path: Path) -> None:
    db = str(tmp_path / "descent.sqlite3")
    env = dict(os.environ, LUCIFER_DESCENT_PROBE="boss:2")

    def run(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "lucifer_descent.cli", "--db", db, *argv],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
        )

    done = run("new", "--profile", "zed", "--seed", "0x5EED", "--rings", "2")
    assert done.returncode == 0, done.stderr
    done = run("grant", "--profile", "zed", "--tier", "2")
    assert done.returncode == 0, done.stderr
    (sigil,) = SIGIL_RE.findall(done.stdout)
    done = run("open", "--profile", "zed", "--node", "1", "--sigil", sigil)
    assert done.returncode == 0, done.stderr
    assert "boss yes; elite packs 2" in done.stdout
    done = run("event", "--profile", "zed", "--kind", "boss")
    assert done.returncode == 0 and "node 1; cleared" in done.stdout
    done = run("event", "--profile", "zed", "--kind", "boss")
    assert done.returncode == 1
    assert done.stderr == "error: no instance is active\n"
    png = tmp_path / "zed.png"
    done = run("render", "--profile", "zed", "--out", str(png), "--size", "200")
    assert done.returncode == 0, done.stderr
    assert NodeState.CLEARED in state_colours(colours_in(png.read_bytes()))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
