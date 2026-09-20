"""Tests for the app-facing API and the command line tool.

Design: jarvis_alerts/contracts.py.  The CLI is the composition root, so
these tests drive it the way an operator would -- ``register`` from a blob
file, ``publish``, ``worker --once``, ``stats``, ``dead`` -- on a temp
sqlite file, mostly in-process through ``cli.main(argv, out, err)`` and
once through the real ``python3 -m jarvis_alerts.cli`` entry point.

The scripted failures come through ``cli.TRANSPORT_FACTORIES``, the seam
the CLI leaves for swapping the fake transport; the senders the real
transports need go through ``api.set_sender``.  Both are reset per test.

Privacy: every blob written here carries a canary and every byte of
stdout and stderr the CLI produces, on happy paths and error paths alike,
is checked not to contain it.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Runnable as `pytest tests/test_alerts_cli.py` or
# `python3 tests/test_alerts_cli.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_alerts import api, cli
from jarvis_alerts.contracts import (
    EXHAUSTED_RETRY_COOLDOWN_S,
    MAX_ATTEMPTS,
    Alert,
    DeadReason,
    Priority,
    RowState,
    SendResult,
    Subscription,
    backoff_seconds,
)
from jarvis_alerts.outbox import DEDUPE_WINDOW_S, Outbox
from jarvis_alerts.transports import FakeTransport
from jarvis_alerts.worker import SimClock, Worker

CANARY = "BLOB-CANARY-51c3e7"
BLOB = (
    '{"endpoint":"https://push.example/' + CANARY + '",'
    '"keys":{"p256dh":"' + CANARY + '-p256dh","auth":"' + CANARY + '-auth"}}'
)
HEX32 = re.compile(r"^[0-9a-f]{32}$")
T0 = 1_700_000_000.0


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_wiring(monkeypatch):
    """Every test starts with no senders and the stock transport factories."""
    api.clear_senders()
    monkeypatch.setattr(cli, "TRANSPORT_FACTORIES", dict(cli.TRANSPORT_FACTORIES))
    yield
    api.clear_senders()


@pytest.fixture
def db(tmp_path) -> str:
    return str(tmp_path / "alerts.sqlite3")


@pytest.fixture
def blob_file(tmp_path) -> str:
    path = tmp_path / "blob.json"
    path.write_text(BLOB + "\n", encoding="utf-8")   # trailing newline, as an editor leaves it
    return str(path)


class Run:
    """Drive ``cli.main`` in-process and keep every byte it wrote."""

    def __init__(self, db: str) -> None:
        self.db = db
        self.all_output: List[str] = []

    def __call__(self, *argv: str, db: bool = True) -> Tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        args = (["--db", self.db] if db else []) + list(argv)
        code = cli.main(args, out, err)
        self.all_output.append(out.getvalue())
        self.all_output.append(err.getvalue())
        return code, out.getvalue(), err.getvalue()

    def stats(self) -> Dict[str, Any]:
        code, out, _ = self("stats")
        assert code == 0
        return json.loads(out)


@pytest.fixture
def run(db) -> Run:
    return Run(db)


def register(run: Run, blob_file: str, device: str = "phone", transport: str = "fake") -> None:
    code, out, err = run("register", "--profile", "owner", "--device", device,
                         "--transport", transport, "--blob-file", blob_file)
    assert code == 0, err
    assert out == f"registered (owner, {device}) via {transport}\n"


def publish(run: Run, **extra: str) -> str:
    argv = ["publish", "--profile", "owner", "--kind", "render_done",
            "--title", "Render finished", "--body", "crypt.png is ready"]
    for key, value in extra.items():
        argv += [f"--{key}", value]
    code, out, err = run(*argv)
    assert code == 0, err
    alert_id = out.strip()
    assert HEX32.match(alert_id), alert_id
    return alert_id


def scripted_fake(script) -> None:
    """Make the CLI's "fake" transport follow ``script(sub, alert, index)``."""
    cli.TRANSPORT_FACTORIES["fake"] = lambda sender: FakeTransport(script)


def permanent(sub: Subscription, alert: Alert, index: int) -> SendResult:
    return SendResult(ok=False, retryable=False, reason="payload too large")


# --------------------------------------------------------------------------
# The happy path: register -> publish -> worker --once -> stats
# --------------------------------------------------------------------------


def test_register_publish_worker_stats_delivered(run, blob_file):
    register(run, blob_file)
    alert_id = publish(run)

    code, out, err = run("worker", "--once")
    assert code == 0, err
    assert out == "pass: leased=1 delivered=1 retried=0 dead=0 pruned=0 errors=0 parked=0\n"

    stats = run.stats()
    assert stats["delivered"] == 1
    assert stats["pending"] == stats["leased"] == stats["dead"] == 0
    assert stats["alerts"] == 1 and stats["subscriptions_live"] == 1
    assert "dead_letters" not in stats          # ``dead`` lists them

    code, out, _ = run("dead")
    assert code == 0 and out == ""

    with Outbox(run.db, clock=time.time) as outbox:
        row, = outbox.rows_for(alert_id)
        assert row.state is RowState.DELIVERED and row.attempts == 1
        assert outbox.subscription("owner", "phone").blob == BLOB   # stored intact, newline stripped


def test_register_reads_blob_from_file_never_argv(run, blob_file):
    register(run, blob_file)
    # There is no way to hand the blob itself to the parser, and the two
    # likely mistakes -- ``--blob TEXT`` (an abbreviation of ``--blob-file``
    # under argparse's defaults) and a bare positional -- are refused
    # without echoing the text.
    for extra in (["--blob", BLOB], [BLOB], ["--blob=" + BLOB]):
        code, out, err = run("register", "--profile", "o", "--device", "d", "--transport", "fake",
                             "--blob-file", blob_file, *extra)
        assert code == 2, (extra, err)
        assert out == "" and err.startswith("error: unrecognized arguments") and err.count("\n") == 1
        assert CANARY not in err
    # And the blob given *as* the path is refused before a "no such file"
    # error could quote it.
    code, out, err = run("register", "--profile", "o", "--device", "d", "--transport", "fake",
                         "--blob-file", BLOB)
    assert code == 1 and "looks like the blob itself" in err and err.count("\n") == 1
    assert CANARY not in err
    assert run.stats()["subscriptions_live"] == 1          # only the file-based one


def test_publish_priority_dedupe_and_data(run, blob_file):
    register(run, blob_file)
    first = publish(run, priority="high", dedupe="render:crypt", data='{"file":"crypt.png"}')
    again = publish(run, dedupe="render:crypt")             # inside the window: collapsed
    assert run.stats()["alerts"] == 1
    with Outbox(run.db, clock=time.time) as outbox:
        alert = outbox.alert(first)
        assert alert.priority is Priority.HIGH
        assert alert.dedupe_key == "render:crypt"
        assert alert.data == {"file": "crypt.png"}
        if again != first:                                   # ids only coincide in one clock instant
            assert outbox.alert(again) is None
    other = publish(run, kind="gpu_missing")
    assert other != first and run.stats()["alerts"] == 2


# --------------------------------------------------------------------------
# Dead letters
# --------------------------------------------------------------------------


def test_dead_lists_row_after_scripted_permanent_failure(run, blob_file):
    scripted_fake(permanent)
    register(run, blob_file)
    alert_id = publish(run)

    code, out, err = run("worker", "--once")
    assert code == 0, err
    assert "dead=1" in out and "delivered=0" in out

    stats = run.stats()
    assert stats["dead"] == 1 and stats["delivered"] == 0 and stats["pending"] == 0

    code, out, _ = run("dead")
    assert code == 0
    # A permanent death names itself: unlike an ``exhausted`` one, nothing
    # brings this row back but ``requeue``.
    assert out == (
        f"row=1 alert={alert_id} profile=owner device=phone attempts=1 "
        f"reason='payload too large' dead_reason=permanent\n"
    )
    code, out, _ = run("dead", "--limit", "0")
    assert code == 0 and out == ""


def test_dead_after_retries_exhausted_and_gone_device(run, blob_file):
    """Retryable failures dead-letter only at MAX_ATTEMPTS; a gone result at
    once, and the device is pruned.  Backoff means the retries are not due
    within one pass, so this drives the worker directly with a SimClock
    (the CLI's own loop uses the wall clock)."""
    register(run, blob_file)
    register(run, blob_file, device="tablet")
    publish(run)
    clock = SimClock(time.time() + 1.0)

    def script(sub: Subscription, alert: Alert, index: int) -> SendResult:
        if sub.device_id == "tablet":
            return SendResult(ok=False, gone=True, reason="410")
        return SendResult(ok=False, retryable=True, reason="503")

    with Outbox(run.db, clock=clock) as outbox:
        worker = Worker(outbox, {"fake": FakeTransport(script)}, clock, jitter=lambda: 0.0)
        for _ in range(MAX_ATTEMPTS):
            worker.run_once()
            clock.advance(400.0)                              # past any backoff and lease
    code, out, _ = run("dead")
    assert code == 0
    lines = out.splitlines()
    assert len(lines) == 2                                    # newest first
    assert lines[0].startswith("row=2 ")
    assert "device=tablet attempts=1 reason='410' dead_reason=gone" in lines[0]
    assert lines[1].startswith("row=1 ")
    assert f"device=phone attempts={MAX_ATTEMPTS} reason='503' dead_reason=exhausted" in lines[1]
    stats = run.stats()
    assert stats["dead"] == 2 and stats["subscriptions_gone"] == 1
    # Only the exhausted row is one the outbox will bring back by itself;
    # the gone device's row waits for a re-registration and a requeue.
    assert stats["dead_exhausted"] == 1 and stats["dead_permanent"] == 1
    assert stats["dead_revivable"] == 1 and stats["revived"] == 0


def test_worker_revives_an_exhausted_dead_letter_once_the_cooldown_has_passed(run, blob_file):
    """A transient outage that outlasts the retry budget used to be the end
    of an alert.  Now the next worker pass more than
    EXHAUSTED_RETRY_COOLDOWN_S after the death puts the row back by itself
    and says ``revived=`` on its line.  The death is staged with a clock of
    its own, far enough in the past that the CLI's wall-clock pass is past
    the cooldown (the CLI's loop has no clock to hand)."""
    register(run, blob_file)
    started = time.time() - EXHAUSTED_RETRY_COOLDOWN_S - 200.0
    clock = SimClock(started)

    def outage(sub: Subscription, alert: Alert, index: int) -> SendResult:
        return SendResult(ok=False, retryable=True, reason="503")

    with Outbox(run.db, clock=clock) as outbox:
        outbox.publish(Alert("a-outage", "owner", "render_done", "Render finished",
                             "crypt.png is ready", clock()))
        worker = Worker(outbox, {"fake": FakeTransport(outage)}, clock, jitter=lambda: 0.0)
        for n in range(1, MAX_ATTEMPTS + 1):
            worker.run_once()
            clock.advance(backoff_seconds(n, 0.0))               # exactly the scheduled wait
        (row,) = outbox.rows_for("a-outage")
        assert row.state is RowState.DEAD and row.dead_reason is DeadReason.EXHAUSTED
        assert row.dead_at + EXHAUSTED_RETRY_COOLDOWN_S < time.time()

    stats = run.stats()
    assert stats["dead"] == 1 and stats["dead_exhausted"] == 1 and stats["dead_revivable"] == 1
    assert stats["revived"] == 0 and stats["delivered"] == 0
    code, out, _ = run("dead")
    assert code == 0 and "dead_reason=exhausted" in out

    # The outage is over, so the stock fake delivers it on the revived pass.
    code, out, err = run("worker", "--once")
    assert code == 0, err
    assert "revived=1" in out and "delivered=1" in out and "dead=0" in out

    stats = run.stats()
    assert stats["dead"] == 0 and stats["delivered"] == 1 and stats["revived"] == 1
    assert stats["dead_exhausted"] == 0 and stats["dead_revivable"] == 0
    code, out, _ = run("dead")
    assert code == 0 and out == ""


# --------------------------------------------------------------------------
# Privacy: the blob never reaches stdout or stderr
# --------------------------------------------------------------------------


def test_blob_never_appears_in_any_output(run, blob_file, tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("not json " + CANARY, encoding="utf-8")
    empty = tmp_path / "empty.json"
    empty.write_text("  \n", encoding="utf-8")

    register(run, blob_file)
    register(run, blob_file, device="laptop", transport="webpush")      # unwired: parked
    assert run("register", "--profile", "owner", "--device", "tv", "--transport", "fake",
               "--blob-file", str(bad))[0] == 1
    assert run("register", "--profile", "owner", "--device", "tv", "--transport", "fake",
               "--blob-file", str(empty))[0] == 1
    assert run("register", "--profile", "owner", "--device", "tv", "--transport", "fake",
               "--blob-file", str(tmp_path / "missing.json"))[0] == 1
    publish(run)
    scripted_fake(lambda sub, alert, index: (_ for _ in ()).throw(ConnectionError(sub.blob)))
    assert run("worker", "--once")[0] == 0                 # a transport raising with the blob in it
    scripted_fake(permanent)
    run("worker", "--once")
    run("stats")
    run("dead")
    run("unregister", "--profile", "owner", "--device", "phone")
    assert run("unregister", "--profile", "owner", "--device", "phone")[0] == 1

    everything = "".join(run.all_output)
    assert CANARY not in everything
    assert "Traceback" not in everything
    # The error lines name the device, the path or the problem, one line each.
    for chunk in run.all_output:
        for line in chunk.splitlines():
            if line.startswith("error: "):
                assert "\n" not in line.strip()


def test_api_register_rejects_bad_blob_without_echoing_it():
    outbox = Outbox(":memory:", clock=lambda: T0)
    service = api.AlertService(outbox, clock=lambda: T0)
    for blob in ("", "   ", "not json " + CANARY, 42):
        with pytest.raises(ValueError) as info:
            service.register_device("owner", "phone", "fake", blob)  # type: ignore[arg-type]
        assert CANARY not in str(info.value)
        assert "(owner, phone)" in str(info.value)


# --------------------------------------------------------------------------
# Transport wiring
# --------------------------------------------------------------------------


def test_unwired_transport_rows_are_left_pending_not_dead(run, blob_file):
    register(run, blob_file, device="laptop", transport="webpush")
    alert_id = publish(run)

    code, out, err = run("worker", "--once", "--lease", "0.001")
    assert code == 0
    notices = [line for line in err.splitlines() if "'webpush' is not wired in this process" in line]
    assert len(notices) == 1, err
    assert "parked=1" in out and "dead=0" in out

    stats = run.stats()
    assert stats["dead"] == 0 and stats["delivered"] == 0
    assert stats["pending"] + stats["leased"] == 1              # LEASED until the lease expires
    with Outbox(run.db, clock=time.time) as outbox:
        row, = outbox.rows_for(alert_id)
        assert row.state is not RowState.DEAD
        assert row.attempts == 0 and outbox.attempts_for(row.row_id) == []

    # Now wire webpush: the app installs a sender; the CLI builds the real
    # WebPushTransport around it and the parked row (lease long expired)
    # is delivered.  The sender sees the endpoint, as the transport contract
    # says, and nothing of it is printed.
    seen: List[Tuple[str, bytes, Dict[str, str]]] = []

    def sender(endpoint: str, body: bytes, headers: Dict[str, str]) -> int:
        seen.append((endpoint, body, headers))
        return 201

    api.set_sender("webpush", sender)
    time.sleep(0.01)
    code, out, err = run("worker", "--once")
    assert code == 0, err
    assert "delivered=1" in out and "parked=0" in out
    assert "'webpush' is not wired" not in err
    assert len(seen) == 1 and seen[0][0] == "https://push.example/" + CANARY
    assert json.loads(seen[0][1])["id"] == alert_id
    assert run.stats()["delivered"] == 1
    assert CANARY not in "".join(run.all_output)


def test_unknown_transport_name_is_parked_with_one_notice(run, blob_file):
    """A name no factory knows (registered through the API, which stores
    any name) is parked too, with one line, and the fake rows still flow."""
    with Outbox(run.db, clock=time.time) as outbox:
        api.AlertService(outbox, clock=time.time).register_device("owner", "watch", "apns", BLOB)
    register(run, blob_file)
    publish(run)
    publish(run, kind="gpu_missing")
    code, out, err = run("worker", "--once")
    assert code == 0
    assert out == "pass: leased=2 delivered=2 retried=0 dead=0 pruned=0 errors=0 parked=2\n"
    assert err.count("'apns' is not wired in this process") == 1
    stats = run.stats()
    # The apns rows were never leased by a process that cannot send them:
    # they stay PENDING, due, for a process that can.
    assert stats["delivered"] == 2 and stats["dead"] == 0 and stats["leased"] == 0 and stats["pending"] == 2
    assert stats["pending_by_transport"] == {"apns": 2}


def test_parked_rows_do_not_starve_wired_rows(run, blob_file):
    """Fifty unwired rows ahead of one fake row in a batch of 5: the parker
    keeps leasing past them and the fake row is delivered in the pass."""
    with Outbox(run.db, clock=time.time) as outbox:
        service = api.AlertService(outbox, clock=time.time)
        for i in range(50):
            service.register_device("owner", f"w{i:02d}", "webpush", BLOB)
    register(run, blob_file, device="zz-phone")
    publish(run)
    code, out, err = run("worker", "--once", "--batch", "5")
    assert code == 0, err
    assert "delivered=1" in out and "parked=50" in out
    assert run.stats()["delivered"] == 1


def test_build_transports_wires_only_what_has_a_sender(monkeypatch):
    transports, unwired = cli.build_transports()
    assert set(transports) == {"fake"} and set(unwired) == {"webpush", "fcm"}
    assert transports["fake"].name == "fake"
    for line in unwired.values():
        assert "is not wired in this process" in line and "set_sender" in line

    api.set_sender("fcm", lambda token, message: (200, ""))
    transports, unwired = cli.build_transports()
    assert set(transports) == {"fake", "fcm"} and set(unwired) == {"webpush"}
    assert transports["fcm"].name == "fcm"

    # A sender is installed but the transport class cannot be imported.
    def broken(sender):
        raise ImportError("no such module")
    monkeypatch.setitem(cli.TRANSPORT_FACTORIES, "fcm", broken)
    transports, unwired = cli.build_transports()
    assert set(transports) == {"fake"} and "fcm" in unwired
    assert "ImportError" in unwired["fcm"] and "is not wired in this process" in unwired["fcm"]


def test_set_sender_validates():
    with pytest.raises(TypeError):
        api.set_sender("webpush", "not callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        api.set_sender("", print)
    api.set_sender("webpush", print)
    assert api.installed_senders() == ["webpush"] and api.get_sender("webpush") is print
    api.clear_senders()
    assert api.installed_senders() == [] and api.get_sender("webpush") is None


def test_unwired_rows_are_taken_at_once_by_a_process_that_has_the_transport(run, blob_file):
    """Two worker processes on one file, one per transport: the one without
    fcm never leases the fcm row, so the one with it delivers immediately
    instead of after a lease expires."""
    fcm_blob = Path(blob_file).with_name("fcm.json")
    fcm_blob.write_text('{"token": "' + CANARY + '-token"}\n', encoding="utf-8")
    register(run, str(fcm_blob), device="phone", transport="fcm")
    alert_id = publish(run)
    code, out, err = run("worker", "--once", "--lease", "300")       # fake-only process, long lease
    assert code == 0 and "parked=1" in out and "leased=0" in out
    stats = run.stats()
    assert stats["leased"] == 0 and stats["pending"] == 1 and stats["pending_by_transport"] == {"fcm": 1}
    api.set_sender("fcm", lambda token, message: (200, ""))
    code, out, err = run("worker", "--once")                          # a process that has fcm: no wait
    assert code == 0 and "delivered=1" in out and "parked=0" in out
    assert run.stats()["delivered"] == 1
    assert CANARY not in "".join(run.all_output)


def test_requeue_puts_dead_rows_back_in_flight(run, blob_file):
    scripted_fake(permanent)
    register(run, blob_file)
    register(run, blob_file, device="tablet")
    publish(run)
    assert run("worker", "--once")[1].count("dead=2") == 1
    assert run.stats()["dead"] == 2

    code, out, err = run("requeue", "--row", "1")
    assert code == 0 and out == "requeued 1 row(s)\n", err
    stats = run.stats()
    assert stats["dead"] == 1 and stats["pending"] == 1
    assert run("requeue", "--row", "1")[1] == "requeued 0 row(s)\n"     # already pending
    code, out, err = run("requeue", "--all", "--profile", "owner", "--device", "tablet")
    assert code == 0 and out == "requeued 1 row(s)\n"
    assert run.stats()["dead"] == 0
    cli.TRANSPORT_FACTORIES["fake"] = lambda sender: FakeTransport()    # the service recovered
    assert "delivered=2" in run("worker", "--once")[1]
    assert run.stats()["delivered"] == 2
    with Outbox(run.db, clock=time.time) as outbox:
        assert [a.ok for a in outbox.attempts_for(1)] == [False, True]  # the history is kept

    for argv in (["requeue"], ["requeue", "--row", "1", "--all"], ["requeue", "--all", "--device", "x"],
                 ["requeue", "--row", "1", "--profile", "owner"], ["requeue", "--row", "999"]):
        code, out, err = run(*argv)
        assert code == 1 and out == "" and err.startswith("error: ") and err.count("\n") == 1, argv
    code, out, err = run("requeue", "--row", BLOB)
    assert code == 2 and CANARY not in err and "not shown" in err


def test_blob_like_values_are_never_echoed_by_any_usage_error(run, blob_file, tmp_path):
    """Every path on which argparse or the CLI could quote a value: the
    blob-file value that is not a path, an invalid choice, an unparsable
    number, a stray token, a blob typed as an id."""
    endpoint = "https://fcm.googleapis.com/fcm/send/" + CANARY
    token = "dGhpcyBpcyBh:APA91b" + CANARY
    cases = [
        (1, ["register", "--profile", "o", "--device", "d", "--transport", "webpush", "--blob-file", endpoint]),
        (1, ["register", "--profile", "o", "--device", "d", "--transport", "fcm", "--blob-file", token]),
        (1, ["register", "--profile", "o", "--device", "d", "--transport", "fcm", "--blob-file", "'" + BLOB + "'"]),
        (1, ["register", "--profile", "o", "--device", "d", "--transport", "fcm", "--blob-file", "@" + BLOB]),
        (1, ["register", "--profile", "o", "--device", "d", "--transport", "fcm", "--blob-file", "\ufeff" + BLOB]),
        (2, ["register", "--profile", "o", "--device", "d", "--transport", "fcm", "--blob-file", "-" + BLOB]),
        (2, ["register", "--profile", "o", "--device", "d", "--transport", BLOB, "--blob-file", blob_file]),
        (2, [BLOB]),
        (2, ["stats", "-" + token]),
        (2, ["stats", "--" + CANARY]),
        (2, ["stats", "--blob-file", BLOB]),                              # a real option, wrong command
        (2, ["worker", "--once", "--batch", BLOB]),
        (2, ["publish", "--profile", "o", "--kind", "k", "--title", "t", "--body", "b", "--priority", BLOB]),
        (1, ["unregister", "--profile", BLOB, "--device", "d"]),
        (1, ["register", "--profile", "o", "--device", "d", "--transport", "fake",
             "--blob-file", str(tmp_path / ("missing-" + CANARY + ".json"))]),
    ]
    for expected, argv in cases:
        code, out, err = run(*argv)
        assert code == expected, (argv, err)
        assert out == "" and CANARY not in err and "Traceback" not in err, (argv, err)
    code, out, err = run("stats", "--blob-file", BLOB)
    assert "--blob-file" in err and "1 value not shown" in err            # the option is named, the value is not
    # An existing path is still named, because then it is a path.
    unreadable = tmp_path / "dir.json"
    unreadable.mkdir()
    code, out, err = run("register", "--profile", "o", "--device", "d", "--transport", "fake",
                         "--blob-file", str(unreadable))
    assert code == 1 and str(unreadable) in err


def test_register_device_can_backfill_and_supersede():
    clock = SimClock(T0)
    service = make_service(clock)
    early = service.publish("owner", "render_done", "old", "b")
    clock.advance(100.0)
    assert service.register_device("owner", "phone", "webpush", BLOB, backfill_s=3600) == 1
    assert [r.device_id for r in service.outbox.rows_for(early)] == ["phone"]
    # The same subscription under a new device id (site data cleared) replaces the old one.
    assert service.register_device("owner", "phone-2", "webpush", BLOB, backfill_s=3600, supersede_same_blob=True) == 1
    assert service.outbox.subscription("owner", "phone") is None
    assert [r.device_id for r in service.outbox.rows_for(early)] == ["phone", "phone-2"]
    assert service.stats()["pending_unreachable"] == 1                    # the old device's row is parked
    with pytest.raises(ValueError):
        service.register_device("owner", "phone-2", "webpush", BLOB, backfill_s=-1)
    # A lone surrogate that json.loads lets through is refused by name.
    with pytest.raises(ValueError) as info:
        service.register_device("owner", "tv", "webpush", '{"endpoint": "' + CANARY + '", "auth": "\ud800"}')
    assert CANARY not in repr(info.value) + repr(info.value.args) and "(owner, tv)" in str(info.value)
    assert service.requeue_dead("owner") == 0


# --------------------------------------------------------------------------
# The worker loop
# --------------------------------------------------------------------------


def test_worker_loop_runs_until_stopped(db, blob_file, run):
    """``run_worker`` without ``--once``: idles on the stop event (no sleep),
    prints only passes that did something, and returns once stopped."""
    register(run, blob_file)
    publish(run)
    publish(run, kind="gpu_missing")
    stop = threading.Event()

    def script(sub: Subscription, alert: Alert, index: int) -> SendResult:
        if index == 1:
            stop.set()                                   # after the second send
        return SendResult(ok=True)

    out = io.StringIO()
    with Outbox(db, clock=time.time) as outbox:
        port = cli.ParkUnwired(outbox, {"fake"}, notify=lambda line: None)
        worker = Worker(port, {"fake": FakeTransport(script)}, time.time, jitter=lambda: 0.0)
        started = time.monotonic()
        cli.run_worker(worker, port, stop, idle_s=60.0, once=False, out=out)
    assert time.monotonic() - started < 5.0             # never waited the 60 s idle
    assert out.getvalue() == "pass: leased=2 delivered=2 retried=0 dead=0 pruned=0 errors=0 parked=0\n"
    assert run.stats()["delivered"] == 2


def test_worker_once_prints_an_empty_pass(run):
    code, out, err = run("worker", "--once")
    assert code == 0
    assert out == "pass: leased=0 delivered=0 retried=0 dead=0 pruned=0 errors=0 parked=0\n"


def test_worker_seed_is_a_lucifer_stream(run, blob_file):
    """Retry backoff comes from the seeded stream: two runs with the same
    seed schedule the same next_due, a different seed a different one."""
    def schedule(seed: str) -> float:
        r = Run(run.db + "." + seed)
        register(r, blob_file)
        publish(r)
        scripted_fake(lambda sub, alert, index: SendResult(ok=False, retryable=True, reason="503"))
        before = time.time()
        assert r("worker", "--once", "--seed", seed)[0] == 0
        with Outbox(r.db, clock=time.time) as outbox:
            row = outbox.row(1)
            assert row.state is RowState.PENDING and row.attempts == 1
            return row.next_due - before                       # delay + a little wall time

    a, b, c = schedule("0x1234"), schedule("0x1234"), schedule("0x5678")
    assert abs(a - b) < 0.5
    assert abs(a - c) > 0.5 or abs(a - b) < abs(a - c)


# --------------------------------------------------------------------------
# Errors: one line, exit 1
# --------------------------------------------------------------------------


def test_errors_are_one_line_and_exit_1(run, blob_file, tmp_path):
    cases = [
        ["register", "--profile", "owner", "--device", "phone", "--transport", "fake",
         "--blob-file", str(tmp_path / "missing.json")],
        ["unregister", "--profile", "owner", "--device", "nobody"],
        ["publish", "--profile", "owner", "--kind", "k", "--title", "t", "--body", "b",
         "--data", "[1, 2]"],
        ["publish", "--profile", "owner", "--kind", "k", "--title", "t", "--body", "b",
         "--data", "{not json"],
        ["dead", "--limit", "-1"],
        ["worker", "--once", "--batch", "0"],
        ["worker", "--once", "--lease", "0"],
        ["gate", "--profiles", "0", "--alerts", "1", "--seed", "1"],
    ]
    for argv in cases:
        code, out, err = run(*argv)
        assert code == 1, argv
        assert out == "", argv
        assert err.startswith("error: ") and err.count("\n") == 1, (argv, err)
        assert "Traceback" not in err


def test_usage_errors_exit_2(run):
    code, out, err = run("bogus")
    assert code == 2 and "invalid choice" in err
    code, out, err = run("publish", "--profile", "owner")
    assert code == 2 and "required" in err


def test_db_flag_accepted_before_or_after_command(tmp_path, blob_file, monkeypatch):
    monkeypatch.chdir(tmp_path)                           # so a stray default db would show up here
    db = str(tmp_path / "after.sqlite3")

    def main(argv: List[str]) -> str:
        out, err = io.StringIO(), io.StringIO()
        assert cli.main(argv, out, err) == 0, err.getvalue()
        return out.getvalue()

    main(["register", "--profile", "o", "--device", "d", "--transport", "fake",
          "--blob-file", blob_file, "--db", db])
    assert json.loads(main(["--db", db, "stats"]))["subscriptions_live"] == 1
    assert json.loads(main(["stats", "--db", db]))["subscriptions_live"] == 1
    assert json.loads(main(["--db", db, "stats", "--db", db]))["subscriptions_live"] == 1
    assert not (tmp_path / cli.DEFAULT_DB).exists()


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def _stub_validate(monkeypatch, run_gate) -> None:
    module = types.ModuleType("jarvis_alerts.validate")
    module.run_gate = run_gate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jarvis_alerts.validate", module)


def test_gate_runs_validate_and_exits_on_verdict(run, monkeypatch):
    calls: List[Dict[str, Any]] = []

    class Report:
        def __init__(self, ok: bool, problems: List[str]) -> None:
            self.ok, self.problems = ok, problems

    def run_gate(**kwargs: Any) -> Report:
        calls.append(kwargs)
        return Report(kwargs["seed"] == 0x1234, [] if kwargs["seed"] == 0x1234 else ["row 3 lost"])

    _stub_validate(monkeypatch, run_gate)
    code, out, err = run("gate", "--profiles", "3", "--alerts", "7", "--seed", "0x1234", db=False)
    assert code == 0, err
    assert out == "gate passed: profiles=3 alerts=7 seed=0x0000000000001234\n"
    # The keyword names are validate.run_gate's own; a stub taking **kwargs
    # pins them so a rename on either side is caught here.
    assert calls[-1] == {"n_profiles": 3, "n_alerts": 7, "seed": 0x1234}

    code, out, err = run("gate", "--profiles", "3", "--alerts", "7", "--seed", "99", db=False)
    assert code == 1
    assert out == "row 3 lost\ngate FAILED: profiles=3 alerts=7 seed=0x0000000000000063\n"
    assert calls[-1]["seed"] == 99


def test_gate_runs_the_real_validate_module(run):
    """No stub: the CLI's call must match ``jarvis_alerts.validate.run_gate``
    as written, and the real report's summary (counts) is printed."""
    import importlib
    validate = importlib.import_module("jarvis_alerts.validate")
    assert validate.run_gate.__module__ == "jarvis_alerts.validate"
    code, out, err = run("gate", "--profiles", "3", "--alerts", "3", "--seed", "7", db=False)
    assert code == 0, err
    assert err == ""
    lines = out.rstrip("\n").split("\n")
    assert lines[0].startswith("gate: profiles=3 alerts=3 seed=0x0000000000000007")
    assert any(line.strip().startswith("counts:") and "rows_delivered=" in line for line in lines)
    assert "  result: OK" in lines
    assert lines[-1] == "gate passed: profiles=3 alerts=3 seed=0x0000000000000007"
    # The same run again is byte-identical: the gate is a pure function of its arguments.
    assert run("gate", "--profiles", "3", "--alerts", "3", "--seed", "7", db=False) == (code, out, err)
    # A failing gate (an injected defect is not reachable from the CLI, so
    # use the module directly) exits 1 through the same path.
    report = validate.run_gate(3, 3, 7, inject_defect="lose_on_crash")
    assert not report.ok
    passed, problem_lines = cli.gate_verdict(report)
    assert passed is False and problem_lines and all(isinstance(l, str) for l in problem_lines)


def test_gate_verdict_shapes():
    assert cli.gate_verdict(True) == (True, [])
    assert cli.gate_verdict(False) == (False, [])
    assert cli.gate_verdict({"ok": False, "failures": ["a", "b"]}) == (False, ["a", "b"])
    assert cli.gate_verdict(types.SimpleNamespace(passed=lambda: True)) == (True, [])
    with pytest.raises(cli.CliError):
        cli.gate_verdict(object())


def test_gate_without_validate_is_one_line_error(run, monkeypatch):
    monkeypatch.setitem(sys.modules, "jarvis_alerts.validate", None)   # import raises ImportError
    code, out, err = run("gate", "--profiles", "1", "--alerts", "1", "--seed", "1", db=False)
    assert code == 1 and out == ""
    assert err.startswith("error: gate unavailable") and err.count("\n") == 1


def test_gate_with_validate_missing_run_gate(run, monkeypatch):
    monkeypatch.setitem(sys.modules, "jarvis_alerts.validate", types.ModuleType("jarvis_alerts.validate"))
    code, out, err = run("gate", "--profiles", "1", "--alerts", "1", "--seed", "1", db=False)
    assert code == 1 and "no run_gate" in err


# --------------------------------------------------------------------------
# AlertService on its own
# --------------------------------------------------------------------------


def make_service(clock: SimClock, ids: Optional[List[str]] = None) -> api.AlertService:
    outbox = Outbox(":memory:", clock=clock)
    source = iter(ids or [f"id{i}" for i in range(1, 100)])
    return api.AlertService(outbox, clock=clock, id_source=lambda: next(source))


def test_alert_id_is_deterministic_from_injected_sources():
    a = make_service(SimClock(T0)).publish("owner", "render_done", "t", "b")
    b = make_service(SimClock(T0)).publish("owner", "render_done", "t", "b")
    assert a == b == api.alert_id_for("owner", "render_done", "id1", T0)
    assert HEX32.match(a)
    # Every input moves the id.
    assert make_service(SimClock(T0 + 1)).publish("owner", "render_done", "t", "b") != a
    assert make_service(SimClock(T0)).publish("other", "render_done", "t", "b") != a
    assert make_service(SimClock(T0)).publish("owner", "gpu_missing", "t", "b") != a
    assert make_service(SimClock(T0), ["zzz"]).publish("owner", "render_done", "t", "b") != a
    # With a dedupe key the id source is not consulted at all.
    keyed = make_service(SimClock(T0), ["never"]).publish("owner", "render_done", "t", "b", dedupe_key="k")
    assert keyed == api.alert_id_for("owner", "render_done", "k", T0)
    # The title and body are payload, not identity.
    assert make_service(SimClock(T0)).publish("owner", "render_done", "other", "text") == a


def test_publish_stores_what_was_said_and_fans_out():
    clock = SimClock(T0)
    service = make_service(clock)
    service.register_device("owner", "phone", "fake", BLOB)
    service.register_device("owner", "laptop", "webpush", BLOB)
    service.register_device("stranger", "phone", "fake", BLOB)
    alert_id = service.publish("owner", "render_done", "Render finished", "crypt.png is ready",
                               data={"file": "crypt.png"}, priority="HIGH", dedupe_key="render:crypt")
    alert = service.outbox.alert(alert_id)
    assert alert == Alert(alert_id, "owner", "render_done", "Render finished", "crypt.png is ready",
                          T0, Priority.HIGH, "render:crypt", {"file": "crypt.png"})
    rows = service.outbox.rows_for(alert_id)
    assert [(r.profile_id, r.device_id, r.state) for r in rows] == [
        ("owner", "laptop", RowState.PENDING), ("owner", "phone", RowState.PENDING)]
    stats = service.stats()
    assert stats["pending"] == 2 and stats["alerts"] == 1 and stats["subscriptions_live"] == 3

    # Dedupe: the outbox collapses a repeat inside the window; publish still returns an id.
    clock.advance(DEDUPE_WINDOW_S / 2)
    repeat = service.publish("owner", "render_done", "t", "b", dedupe_key="render:crypt")
    assert repeat != alert_id and service.outbox.alert(repeat) is None
    assert service.stats()["alerts"] == 1
    clock.advance(DEDUPE_WINDOW_S)
    later = service.publish("owner", "render_done", "t", "b", dedupe_key="render:crypt")
    assert service.outbox.alert(later) is not None and service.stats()["alerts"] == 2


def test_publish_validates_inputs():
    service = make_service(SimClock(T0))
    with pytest.raises(ValueError):
        service.publish("", "k", "t", "b")
    with pytest.raises(ValueError):
        service.publish("owner", "", "t", "b")
    with pytest.raises(ValueError):
        service.publish("owner", "k", None, "b")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        service.publish("owner", "k", "t", "b", priority="urgent")
    with pytest.raises(ValueError):
        service.publish("owner", "k", "t", "b", data=["not", "a", "dict"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        service.publish("owner", "k", "t", "b", dedupe_key="")
    with pytest.raises(TypeError):
        service.publish("owner", "k", "t", "b", data={"when": object()})   # not JSON-encodable
    assert service.stats()["alerts"] == 0                    # nothing half-written


def test_parse_priority():
    assert api.parse_priority("high") is Priority.HIGH
    assert api.parse_priority(" Low ") is Priority.LOW
    assert api.parse_priority(1) is Priority.NORMAL
    assert api.parse_priority(Priority.HIGH) is Priority.HIGH
    for bad in ("urgent", 7, True, None, 1.5):
        with pytest.raises(ValueError):
            api.parse_priority(bad)  # type: ignore[arg-type]


def test_register_unregister_and_backfill_device():
    clock = SimClock(T0)
    service = make_service(clock)
    early = service.publish("owner", "render_done", "old", "b")     # before any device
    assert service.outbox.rows_for(early) == []
    clock.advance(3000.0)
    recent = service.publish("owner", "render_done", "new", "b")
    clock.advance(1000.0)                                          # early is now 4000 s old

    with pytest.raises(LookupError):
        service.backfill_device("owner", "phone")                   # not registered
    service.register_device("owner", "phone", "fake", BLOB)
    sub = service.outbox.subscription("owner", "phone")
    assert sub.transport == "fake" and sub.created_at == T0 + 4000.0 and not sub.gone

    assert service.backfill_device("owner", "phone") == 1           # default hour: only ``recent``
    assert [r.alert_id for r in service.outbox.rows_for(recent)] == [recent]
    assert service.outbox.rows_for(early) == []
    assert service.backfill_device("owner", "phone") == 0           # nothing new
    assert service.backfill_device("owner", "phone", since_s=5000) == 1   # now ``early`` too
    assert service.stats()["pending"] == 2
    with pytest.raises(ValueError):
        service.backfill_device("owner", "phone", since_s=-1)

    assert service.unregister_device("owner", "phone") is True
    assert service.unregister_device("owner", "phone") is False
    assert service.stats()["pending_unreachable"] == 2              # parked, not deleted
    with pytest.raises(LookupError):
        service.backfill_device("owner", "phone")


def test_register_device_validates_names():
    service = make_service(SimClock(T0))
    for args in (("", "d", "fake", BLOB), ("o", "", "fake", BLOB), ("o", "d", "", BLOB)):
        with pytest.raises(ValueError):
            service.register_device(*args)


# --------------------------------------------------------------------------
# The real entry point
# --------------------------------------------------------------------------


def test_python_dash_m_entry_point(tmp_path, blob_file):
    db = str(tmp_path / "sub.sqlite3")
    base = [sys.executable, "-m", "jarvis_alerts.cli", "--db", db]
    outputs: List[str] = []

    def sh(*argv: str, expect: int = 0) -> str:
        proc = subprocess.run(base + list(argv), cwd=str(ROOT), capture_output=True, text=True,
                              timeout=60)
        outputs.append(proc.stdout + proc.stderr)
        assert proc.returncode == expect, (argv, proc.stdout, proc.stderr)
        return proc.stdout

    assert sh("register", "--profile", "owner", "--device", "phone", "--transport", "fake",
              "--blob-file", blob_file) == "registered (owner, phone) via fake\n"
    alert_id = sh("publish", "--profile", "owner", "--kind", "render_done",
                  "--title", "Render finished", "--body", "crypt.png is ready").strip()
    assert HEX32.match(alert_id)
    assert "delivered=1" in sh("worker", "--once")
    assert json.loads(sh("stats"))["delivered"] == 1
    assert sh("dead") == ""
    sh("unregister", "--profile", "owner", "--device", "ghost", expect=1)
    assert CANARY not in "".join(outputs)
    assert "Traceback" not in "".join(outputs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
