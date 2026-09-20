"""Tests for jarvis_alerts.transports.

Design: jarvis_alerts/contracts.py, module docstring point 2 (the transport
is injected and never holds authentication material; the blob is the app's
own data and is never logged) and point 3 (the client dedupes by alert id,
so the payload must carry it).

Everything runs against stub senders that record what they were handed and
return a scripted status; nothing here touches a network, a clock or
``random``. The one Stream-driven test uses ``lucifer_gen.seed``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Runnable as `pytest tests/test_alerts_transports.py` or
# `python3 tests/test_alerts_transports.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_alerts.contracts import Alert, Priority, SendResult, Subscription
from jarvis_alerts.transports import (
    FCM_GONE_MARKERS,
    FCMTransport,
    FakeTransport,
    WebPushTransport,
    encode_json,
    fcm_message,
    map_fcm_response,
    map_webpush_status,
    payload_bytes,
    payload_for,
    webpush_headers,
)
from lucifer_gen.seed import SeedFields

T0 = 1_700_000_000.0
PID = "owner"
ENDPOINT = "https://push.example/send/SECRET-ENDPOINT-TOKEN"
TOKEN = "fcm-registration-SECRET-TOKEN"
WEBPUSH_BLOB = json.dumps({"endpoint": ENDPOINT, "keys": {"p256dh": "P256SECRET", "auth": "AUTHSECRET"}})
FCM_BLOB = json.dumps({"token": TOKEN})
#: Every distinct secret-looking fragment of the blobs; none may leak.
SECRETS = (ENDPOINT, TOKEN, "P256SECRET", "AUTHSECRET", "SECRET")

MALFORMED_BLOBS = [
    "",
    "not json",
    "null",
    "42",
    "[]",
    '["https://push.example/x"]',
    "{}",
    '{"endpoint": 5}',
    '{"endpoint": ""}',
    '{"endpoint": null}',
    '{"token": ""}',
    '{"token": ["t"]}',
    '{"keys": {"auth": "x"}}',
    "{'endpoint': 'single quotes'}",
    '{"endpoint": "https://push.example/x"',  # truncated
]


def alert(alert_id: str = "a1", priority: Priority = Priority.NORMAL, **data: Any) -> Alert:
    return Alert(alert_id, PID, "render_done", "Render finished", "crypt.png is ready", T0,
                 priority=priority, data=data)


def webpush_sub(device: str = "laptop", blob: str = WEBPUSH_BLOB) -> Subscription:
    return Subscription(PID, device, "webpush", blob=blob, created_at=T0)


def fcm_sub(device: str = "phone", blob: str = FCM_BLOB) -> Subscription:
    return Subscription(PID, device, "fcm", blob=blob, created_at=T0)


def assert_no_secret(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text, f"leaked {secret!r} into {text!r}"


def flags(result: SendResult) -> Tuple[bool, bool, bool]:
    return (result.ok, result.retryable, result.gone)


OK, RETRY, GONE, PERMANENT = (True, False, False), (False, True, False), (False, False, True), (False, False, False)


class WebPushStub:
    """Records every (endpoint, body, headers) and returns scripted statuses."""

    def __init__(self, *statuses: Any) -> None:
        self.statuses = list(statuses)
        self.calls: List[Tuple[str, bytes, Dict[str, str]]] = []

    def __call__(self, endpoint: str, body: bytes, headers: Dict[str, str]) -> int:
        self.calls.append((endpoint, body, dict(headers)))
        return self.statuses.pop(0) if self.statuses else 201


class FCMStub:
    """Records every (token, message) and returns scripted (status, body) pairs."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def __call__(self, token: str, message: Dict[str, Any]) -> Tuple[int, str]:
        self.calls.append((token, json.loads(json.dumps(message))))
        return self.responses.pop(0) if self.responses else (200, '{"name":"projects/x/messages/1"}')


# ---------------------------------------------------------------------------
# payload_for
# ---------------------------------------------------------------------------


def test_payload_has_exactly_the_documented_keys_and_the_id() -> None:
    a = alert(file="crypt.png", seed=7)
    p = payload_for(a)
    assert set(p) == {"id", "kind", "title", "body", "data", "priority"}
    assert p["id"] == "a1"
    assert p["kind"] == "render_done"
    assert p["title"] == "Render finished"
    assert p["body"] == "crypt.png is ready"
    assert p["data"] == {"file": "crypt.png", "seed": 7}
    assert p["priority"] == "normal"


@pytest.mark.parametrize("priority, name", [(Priority.LOW, "low"), (Priority.NORMAL, "normal"), (Priority.HIGH, "high")])
def test_payload_priority_is_the_lowercase_name(priority: Priority, name: str) -> None:
    assert payload_for(alert(priority=priority))["priority"] == name


def test_payload_accepts_a_plain_int_priority() -> None:
    a = Alert("a1", PID, "k", "t", "b", T0, priority=2)  # type: ignore[arg-type]
    assert payload_for(a)["priority"] == "high"


def test_payload_rejects_an_unknown_priority() -> None:
    a = Alert("a1", PID, "k", "t", "b", T0, priority=9)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        payload_for(a)


def test_payload_is_deterministic_whatever_the_data_insertion_order() -> None:
    a = Alert("a1", PID, "k", "t", "b", T0, data={"x": 1, "y": {"b": 2, "a": 1}})
    b = Alert("a1", PID, "k", "t", "b", T0, data={"y": {"a": 1, "b": 2}, "x": 1})
    assert payload_for(a) == payload_for(b) == payload_for(a)
    assert payload_bytes(a) == payload_bytes(b) == payload_bytes(a)
    assert json.loads(payload_bytes(a)) == payload_for(a)


def test_payload_bytes_are_canonical_and_utf8() -> None:
    a = Alert("a1", PID, "k", "Fini é", "b", T0, data={"z": 1, "a": 2})
    raw = payload_bytes(a)
    assert raw == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert "é".encode("utf-8") in raw
    assert b" " not in raw.replace("Fini é".encode("utf-8"), b"")


def test_payload_data_is_a_copy() -> None:
    a = alert(file="crypt.png")
    p = payload_for(a)
    p["data"]["file"] = "changed"
    p["data"]["extra"] = 1
    assert a.data == {"file": "crypt.png"}


def test_encode_json_rejects_non_json_data() -> None:
    with pytest.raises(TypeError):
        encode_json({"o": object()})


# ---------------------------------------------------------------------------
# Web Push headers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("priority, urgency, ttl", [
    (Priority.HIGH, "high", "3600"),
    (Priority.NORMAL, "normal", "86400"),
    (Priority.LOW, "low", "259200"),
])
def test_webpush_headers_by_priority(priority: Priority, urgency: str, ttl: str) -> None:
    assert webpush_headers(priority) == {"TTL": ttl, "Urgency": urgency}
    stub = WebPushStub(201)
    WebPushTransport(stub).send(webpush_sub(), alert(priority=priority))
    assert stub.calls[0][2] == {"TTL": ttl, "Urgency": urgency}


def test_webpush_header_values_are_strings() -> None:
    for priority in Priority:
        assert all(isinstance(v, str) for v in webpush_headers(priority).values())


def test_webpush_headers_reject_unknown_priority() -> None:
    with pytest.raises(ValueError):
        webpush_headers(7)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Web Push status mapping
# ---------------------------------------------------------------------------


WEBPUSH_TABLE = [
    (200, OK), (201, OK), (202, OK),
    (404, GONE), (410, GONE),
    (429, RETRY), (500, RETRY), (502, RETRY), (503, RETRY), (504, RETRY), (599, RETRY),
    (400, PERMANENT), (401, PERMANENT), (403, PERMANENT),
    # unnamed by the brief; a retry with the same bytes cannot help
    (405, PERMANENT), (413, PERMANENT), (418, PERMANENT), (499, PERMANENT),
    (203, PERMANENT), (204, PERMANENT), (301, PERMANENT), (302, PERMANENT), (100, PERMANENT),
]


@pytest.mark.parametrize("status, expected", WEBPUSH_TABLE)
def test_webpush_status_table(status: int, expected: Tuple[bool, bool, bool]) -> None:
    assert flags(map_webpush_status(status)) == expected
    stub = WebPushStub(status)
    result = WebPushTransport(stub).send(webpush_sub(), alert())
    assert flags(result) == expected
    assert len(stub.calls) == 1
    if expected is OK:
        assert result.reason == ""
    else:
        assert str(status) in result.reason
        assert_no_secret(result.reason)


def oracle_webpush(status: int) -> Tuple[bool, bool, bool]:
    if status in (200, 201, 202):
        return OK
    if status in (404, 410):
        return GONE
    if status == 429 or 500 <= status <= 599:
        return RETRY
    return PERMANENT


def test_webpush_status_table_is_exhaustive_over_every_http_status() -> None:
    for status in range(100, 600):
        result = map_webpush_status(status)
        assert flags(result) == oracle_webpush(status), status
        assert result.ok == (result.reason == "")


def test_webpush_sender_gets_the_endpoint_exactly_once_per_send() -> None:
    stub = WebPushStub(201, 201)
    transport = WebPushTransport(stub)
    a = alert(file="crypt.png")
    assert transport.send(webpush_sub(), a).ok
    assert [c[0] for c in stub.calls] == [ENDPOINT]
    assert transport.send(webpush_sub("desk"), a).ok
    assert [c[0] for c in stub.calls] == [ENDPOINT, ENDPOINT]


def test_webpush_sender_gets_the_canonical_payload_and_headers() -> None:
    stub = WebPushStub(201)
    a = alert(priority=Priority.HIGH, file="crypt.png")
    WebPushTransport(stub).send(webpush_sub(), a)
    endpoint, body, headers = stub.calls[0]
    assert endpoint == ENDPOINT
    assert body == payload_bytes(a)
    assert json.loads(body) == payload_for(a)
    assert json.loads(body)["id"] == a.id
    assert headers == {"TTL": "3600", "Urgency": "high"}


def test_webpush_never_calls_the_sender_with_the_blob_or_keys() -> None:
    stub = WebPushStub(201)
    WebPushTransport(stub).send(webpush_sub(), alert())
    endpoint, body, headers = stub.calls[0]
    for secret in ("P256SECRET", "AUTHSECRET"):
        assert secret not in endpoint
        assert secret.encode() not in body
        assert secret not in json.dumps(headers)


def test_webpush_sender_with_a_keys_parameter_receives_the_subscription_keys() -> None:
    """Payload encryption needs the subscription's own p256dh/auth; a sender
    that takes a fourth argument gets them from the stored blob, so the
    app needs no second copy of the subscription."""
    seen: List[Tuple[str, bytes, Dict[str, str], Dict[str, str]]] = []

    def sender(endpoint: str, body: bytes, headers: Dict[str, str], keys: Dict[str, str]) -> int:
        seen.append((endpoint, body, dict(headers), dict(keys)))
        return 201

    assert WebPushTransport(sender).send(webpush_sub(), alert()).ok
    assert seen[0][0] == ENDPOINT and seen[0][3] == {"p256dh": "P256SECRET", "auth": "AUTHSECRET"}
    # No keys object, or one with non-string values: an empty dict, never an exception.
    seen.clear()
    WebPushTransport(sender).send(webpush_sub(blob=json.dumps({"endpoint": ENDPOINT})), alert())
    WebPushTransport(sender).send(webpush_sub(blob=json.dumps({"endpoint": ENDPOINT, "keys": {"auth": 5}})), alert())
    WebPushTransport(sender).send(webpush_sub(blob=json.dumps({"endpoint": ENDPOINT, "keys": "x"})), alert())
    assert [s[3] for s in seen] == [{}, {}, {}]
    # A callable object and a defaulted fourth parameter count as four; a
    # three-parameter sender (WebPushStub) keeps getting three.
    class Sender:
        def __init__(self) -> None:
            self.keys: List[Dict[str, str]] = []
        def __call__(self, endpoint: str, body: bytes, headers: Dict[str, str], keys: Dict[str, str] = {}) -> int:
            self.keys.append(keys)
            return 201
    obj = Sender()
    WebPushTransport(obj).send(webpush_sub(), alert())
    assert obj.keys == [{"p256dh": "P256SECRET", "auth": "AUTHSECRET"}]
    stub = WebPushStub(201)
    WebPushTransport(stub).send(webpush_sub(), alert())
    assert len(stub.calls[0]) == 3


@pytest.mark.parametrize("blob", MALFORMED_BLOBS)
def test_webpush_malformed_blob_is_gone_not_an_exception(blob: str) -> None:
    stub = WebPushStub()
    result = WebPushTransport(stub).send(webpush_sub(blob=blob), alert())
    assert flags(result) == GONE
    assert stub.calls == []
    assert blob == "" or blob not in result.reason
    assert "endpoint" in result.reason


def test_webpush_non_string_blob_is_gone_not_an_exception() -> None:
    sub = Subscription(PID, "laptop", "webpush", blob={"endpoint": ENDPOINT}, created_at=T0)  # type: ignore[arg-type]
    stub = WebPushStub()
    assert flags(WebPushTransport(stub).send(sub, alert())) == GONE
    assert stub.calls == []


@pytest.mark.parametrize("status", [s for s, _ in WEBPUSH_TABLE])
def test_webpush_reason_never_contains_the_blob(status: int) -> None:
    result = WebPushTransport(WebPushStub(status)).send(webpush_sub(), alert())
    assert_no_secret(result.reason)
    assert WEBPUSH_BLOB not in result.reason


def test_webpush_non_json_alert_data_is_permanent_and_quiet() -> None:
    stub = WebPushStub()
    a = Alert("a1", PID, "k", "t", "b", T0, data={"o": object()})
    result = WebPushTransport(stub).send(webpush_sub(), a)
    assert flags(result) == PERMANENT
    assert "object" not in result.reason
    assert stub.calls == []


@pytest.mark.parametrize("make_transport", [
    lambda: WebPushTransport(WebPushStub()),
    lambda: FCMTransport(FCMStub()),
])
def test_invalid_priority_is_permanent_and_quiet_for_both_transports(make_transport: Any) -> None:
    transport = make_transport()
    sub = webpush_sub() if transport.name == "webpush" else fcm_sub()
    a = Alert("a1", PID, "k", "t", "b", T0, priority=9)  # type: ignore[arg-type]
    result = transport.send(sub, a)
    assert flags(result) == PERMANENT
    assert "priority" in result.reason
    assert "9" not in result.reason
    assert transport._sender.calls == []


def test_webpush_sender_exceptions_propagate_to_the_worker() -> None:
    def boom(endpoint: str, body: bytes, headers: Dict[str, str]) -> int:
        raise ConnectionError("push service unreachable")

    with pytest.raises(ConnectionError):
        WebPushTransport(boom).send(webpush_sub(), alert())


@pytest.mark.parametrize("bad", ["201", None, 2.0, (201,)])
def test_webpush_sender_contract_violation_is_a_typeerror_without_the_blob(bad: Any) -> None:
    with pytest.raises(TypeError) as info:
        WebPushTransport(lambda e, b, h: bad).send(webpush_sub(), alert())
    assert_no_secret(str(info.value))


# ---------------------------------------------------------------------------
# FCM message
# ---------------------------------------------------------------------------


def test_fcm_message_shape() -> None:
    a = alert(priority=Priority.HIGH, file="crypt.png", seed=7)
    m = fcm_message(a)
    assert m["notification"] == {"title": "Render finished", "body": "crypt.png is ready"}
    assert set(m["data"]) == {"id", "kind", "title", "body", "data", "priority"}
    assert all(isinstance(v, str) for v in m["data"].values()), m["data"]
    assert m["data"]["id"] == "a1"
    assert m["data"]["priority"] == "high"
    assert json.loads(m["data"]["data"]) == {"file": "crypt.png", "seed": 7}
    assert m["android"] == {"priority": "high", "ttl": "3600s"}
    assert "token" not in m


@pytest.mark.parametrize("priority, android, ttl", [
    (Priority.HIGH, "high", "3600s"), (Priority.NORMAL, "normal", "86400s"), (Priority.LOW, "normal", "259200s"),
])
def test_fcm_android_priority_and_ttl(priority: Priority, android: str, ttl: str) -> None:
    assert fcm_message(alert(priority=priority))["android"] == {"priority": android, "ttl": ttl}


def test_fcm_data_keeps_alert_id_distinct_from_a_data_key_called_id() -> None:
    a = alert(id="not-the-alert-id")
    m = fcm_message(a)
    assert m["data"]["id"] == "a1"
    assert json.loads(m["data"]["data"])["id"] == "not-the-alert-id"


def test_fcm_message_is_deterministic() -> None:
    a = Alert("a1", PID, "k", "t", "b", T0, data={"x": 1, "y": {"b": 2, "a": 1}})
    b = Alert("a1", PID, "k", "t", "b", T0, data={"y": {"a": 1, "b": 2}, "x": 1})
    assert fcm_message(a) == fcm_message(b)
    assert fcm_message(a)["data"]["data"] == '{"x":1,"y":{"a":1,"b":2}}'


def test_fcm_sender_gets_the_token_exactly_once_per_send() -> None:
    stub = FCMStub()
    transport = FCMTransport(stub)
    a = alert()
    assert transport.send(fcm_sub(), a).ok
    assert [c[0] for c in stub.calls] == [TOKEN]
    assert stub.calls[0][1] == fcm_message(a)
    assert transport.send(fcm_sub("tablet"), a).ok
    assert [c[0] for c in stub.calls] == [TOKEN, TOKEN]


# ---------------------------------------------------------------------------
# FCM response mapping
# ---------------------------------------------------------------------------


UNREG = '{"error":{"code":404,"status":"NOT_FOUND","details":[{"errorCode":"UNREGISTERED"}]}}'
FCM_TABLE = [
    ((200, ""), OK),
    ((200, '{"name":"projects/x/messages/1"}'), OK),
    ((200, "UNREGISTERED"), OK),                       # 200 wins: the brief checks it first
    ((404, UNREG), GONE),
    ((404, "NOT_FOUND"), GONE),
    ((400, "UNREGISTERED"), GONE),                     # marker wins over the status
    ((503, "NOT_FOUND"), GONE),
    ((429, '{"error":{"status":"RESOURCE_EXHAUSTED"}}'), RETRY),
    ((500, '{"error":{"status":"INTERNAL"}}'), RETRY),
    ((502, ""), RETRY),
    ((503, '{"error":{"status":"UNAVAILABLE"}}'), RETRY),
    ((599, ""), RETRY),
    ((400, '{"error":{"status":"INVALID_ARGUMENT"}}'), PERMANENT),
    ((401, '{"error":{"status":"UNAUTHENTICATED"}}'), PERMANENT),
    ((403, '{"error":{"status":"SENDER_ID_MISMATCH"}}'), PERMANENT),
    ((404, ""), PERMANENT),                            # 404 without a marker is not in the gone rule
    ((404, "not_found"), PERMANENT),                   # markers are exact, like the v1 error codes
    ((413, ""), PERMANENT),
    ((201, ""), PERMANENT),                            # only 200 is ok for FCM
    ((302, ""), PERMANENT),
]


@pytest.mark.parametrize("response, expected", FCM_TABLE)
def test_fcm_response_table(response: Tuple[int, str], expected: Tuple[bool, bool, bool]) -> None:
    status, body = response
    assert flags(map_fcm_response(status, body)) == expected
    stub = FCMStub(response)
    result = FCMTransport(stub).send(fcm_sub(), alert())
    assert flags(result) == expected
    assert len(stub.calls) == 1
    if expected is OK:
        assert result.reason == ""
    else:
        assert str(status) in result.reason
        # A bare marker word as the whole body is the one case where the
        # reason legitimately contains the body; the token-echo test below
        # covers the real invariant that body text is never copied.
        if body and body not in FCM_GONE_MARKERS:
            assert body not in result.reason


def oracle_fcm(status: int, body: str) -> Tuple[bool, bool, bool]:
    if status == 200:
        return OK
    if any(m in body for m in FCM_GONE_MARKERS):
        return GONE
    if status == 429 or 500 <= status <= 599:
        return RETRY
    return PERMANENT


def test_fcm_response_table_is_exhaustive_over_every_http_status() -> None:
    for status in range(100, 600):
        for body in ("", "UNREGISTERED", "NOT_FOUND", "INTERNAL"):
            result = map_fcm_response(status, body)
            assert flags(result) == oracle_fcm(status, body), (status, body)
            assert result.ok == (result.reason == "")


def test_fcm_gone_reason_names_the_marker_not_the_body() -> None:
    body = '{"error":{"status":"NOT_FOUND","message":"Requested entity ' + TOKEN + ' was not found."}}'
    result = map_fcm_response(404, body)
    assert flags(result) == GONE
    assert result.reason == "fcm status 404 NOT_FOUND"
    assert_no_secret(result.reason)


def test_fcm_reason_never_contains_the_token_or_blob_even_when_the_body_echoes_it() -> None:
    for status in (400, 401, 403, 404, 429, 500, 503):
        body = f'{{"error":"{TOKEN} rejected","blob":{json.dumps(FCM_BLOB)}}}'
        result = FCMTransport(FCMStub((status, body))).send(fcm_sub(), alert())
        assert_no_secret(result.reason)
        assert FCM_BLOB not in result.reason


@pytest.mark.parametrize("blob", MALFORMED_BLOBS + [WEBPUSH_BLOB])
def test_fcm_malformed_blob_is_gone_not_an_exception(blob: str) -> None:
    stub = FCMStub()
    result = FCMTransport(stub).send(fcm_sub(blob=blob), alert())
    assert flags(result) == GONE
    assert stub.calls == []
    assert blob == "" or blob not in result.reason
    assert "token" in result.reason
    assert_no_secret(result.reason)


def test_fcm_body_may_be_bytes_or_none() -> None:
    assert flags(FCMTransport(FCMStub((404, b"UNREGISTERED"))).send(fcm_sub(), alert())) == GONE
    assert flags(FCMTransport(FCMStub((503, None))).send(fcm_sub(), alert())) == RETRY
    assert flags(FCMTransport(FCMStub((404, b"\xff\xfe"))).send(fcm_sub(), alert())) == PERMANENT


def test_fcm_non_json_alert_data_is_permanent_and_quiet() -> None:
    stub = FCMStub()
    a = Alert("a1", PID, "k", "t", "b", T0, data={"o": object()})
    result = FCMTransport(stub).send(fcm_sub(), a)
    assert flags(result) == PERMANENT
    assert "object" not in result.reason
    assert stub.calls == []


def test_fcm_sender_exceptions_propagate_to_the_worker() -> None:
    def boom(token: str, message: Dict[str, Any]) -> Tuple[int, str]:
        raise TimeoutError("fcm timed out")

    with pytest.raises(TimeoutError):
        FCMTransport(boom).send(fcm_sub(), alert())


@pytest.mark.parametrize("bad", [200, "200", (200,), (200, "", ""), ("200", ""), (200, 5), None, b"ok"])
def test_fcm_sender_contract_violation_is_a_typeerror_without_the_blob(bad: Any) -> None:
    with pytest.raises(TypeError) as info:
        FCMTransport(lambda t, m: bad).send(fcm_sub(), alert())
    assert_no_secret(str(info.value))


# ---------------------------------------------------------------------------
# FakeTransport
# ---------------------------------------------------------------------------


def test_fake_default_script_is_ok_and_records_calls_in_order() -> None:
    fake = FakeTransport()
    assert fake.name == "fake"
    assert fake.send(webpush_sub("laptop"), alert("a1")).ok
    assert fake.send(fcm_sub("phone"), alert("a2")).ok
    assert fake.send(webpush_sub("laptop"), alert("a2")).ok
    assert fake.calls == [(PID, "laptop", "a1"), (PID, "phone", "a2"), (PID, "laptop", "a2")]
    assert fake.calls_for(PID, "laptop") == 2
    assert fake.calls_for(PID, "phone") == 1
    assert fake.calls_for(PID, "nobody") == 0
    assert WEBPUSH_BLOB not in repr(fake.calls)


def test_fake_script_sees_the_subscription_alert_and_call_index() -> None:
    seen: List[Tuple[str, str, int]] = []

    def script(sub: Subscription, a: Alert, index: int) -> SendResult:
        seen.append((sub.device_id, a.id, index))
        return SendResult(ok=False, retryable=True, reason="503") if index < 2 else SendResult(ok=True)

    fake = FakeTransport(script)
    results = [fake.send(webpush_sub("laptop"), alert(f"a{i}")) for i in range(4)]
    assert [r.ok for r in results] == [False, False, True, True]
    assert seen == [("laptop", "a0", 0), ("laptop", "a1", 1), ("laptop", "a2", 2), ("laptop", "a3", 3)]
    assert len(fake.calls) == 4


def test_fake_records_the_call_even_when_the_script_raises() -> None:
    def script(sub: Subscription, a: Alert, index: int) -> SendResult:
        raise RuntimeError("simulated crash")

    fake = FakeTransport(script)
    with pytest.raises(RuntimeError):
        fake.send(webpush_sub(), alert())
    assert fake.calls == [(PID, "laptop", "a1")]


def test_fake_can_take_another_name() -> None:
    assert FakeTransport(name="webpush").name == "webpush"


def test_fake_script_driven_by_a_seed_stream_is_reproducible() -> None:
    def make() -> FakeTransport:
        stream = SeedFields.parse(0xDEADBEEF).stream("alerts.fake")

        def script(sub: Subscription, a: Alert, index: int) -> SendResult:
            if stream.chance(0.5):
                return SendResult(ok=False, retryable=True, reason="503")
            return SendResult(ok=True)

        return FakeTransport(script)

    runs = []
    for _ in range(2):
        fake = make()
        runs.append([fake.send(webpush_sub(), alert(f"a{i}")).ok for i in range(20)])
    assert runs[0] == runs[1]
    assert True in runs[0] and False in runs[0]


# ---------------------------------------------------------------------------
# Protocol and privacy
# ---------------------------------------------------------------------------


def test_transport_names_match_the_subscription_transport_vocabulary() -> None:
    assert WebPushTransport(WebPushStub()).name == "webpush"
    assert FCMTransport(FCMStub()).name == "fcm"
    assert FakeTransport().name == "fake"
    for transport in (WebPushTransport(WebPushStub()), FCMTransport(FCMStub()), FakeTransport()):
        assert callable(getattr(transport, "send"))


def test_module_never_reads_the_clock_random_or_logs() -> None:
    source = (ROOT / "jarvis_alerts" / "transports.py").read_text(encoding="utf-8")
    assert "import random" not in source
    assert "time.time" not in source
    assert "import logging" not in source
    assert "os.environ" not in source
    logic, _, demo = source.partition("def _demo(")
    assert "print(" not in logic


def test_every_result_from_every_path_is_blob_free() -> None:
    """Belt and braces: drive both real transports across every table row
    and every malformed blob; no reason may contain a fragment of a blob."""
    seen: List[str] = []
    for status, _ in WEBPUSH_TABLE:
        seen.append(WebPushTransport(WebPushStub(status)).send(webpush_sub(), alert()).reason)
    for response, _ in FCM_TABLE:
        seen.append(FCMTransport(FCMStub(response)).send(fcm_sub(), alert()).reason)
    for blob in MALFORMED_BLOBS:
        seen.append(WebPushTransport(WebPushStub()).send(webpush_sub(blob=blob), alert()).reason)
        seen.append(FCMTransport(FCMStub()).send(fcm_sub(blob=blob), alert()).reason)
    for reason in seen:
        assert_no_secret(reason)
        assert "{" not in reason and "}" not in reason


# ---------------------------------------------------------------------------
# Through the worker (skipped if the sibling module is absent or changed).
# ---------------------------------------------------------------------------


def test_webpush_transport_delivers_and_prunes_through_the_worker() -> None:
    worker_mod = pytest.importorskip("jarvis_alerts.worker")
    for name in ("MemoryStore", "SimClock", "Worker"):
        if not hasattr(worker_mod, name):
            pytest.skip(f"worker has no {name}")
    clock = worker_mod.SimClock(T0)
    store = worker_mod.MemoryStore(clock)
    store.register(webpush_sub("laptop"))
    store.register(webpush_sub("old", blob='{"endpoint":"https://push.example/EXPIRED"}'))
    store.register(fcm_sub("phone"))
    store.publish(alert(priority=Priority.HIGH))

    web = WebPushStub()
    web.statuses = []  # decide per endpoint instead

    def sender(endpoint: str, body: bytes, headers: Dict[str, str]) -> int:
        web.calls.append((endpoint, body, dict(headers)))
        return 410 if endpoint.endswith("EXPIRED") else 201

    fcm = FCMStub()
    worker = worker_mod.Worker(store, {"webpush": WebPushTransport(sender), "fcm": FCMTransport(fcm)},
                               clock, lambda: 0.0)
    report = worker.run_once()
    assert report.delivered == 2
    assert report.pruned == 1
    assert sorted(c[0] for c in web.calls) == sorted([ENDPOINT, "https://push.example/EXPIRED"])
    assert [c[0] for c in fcm.calls] == [TOKEN]
    assert all(h == {"TTL": "3600", "Urgency": "high"} for _, _, h in web.calls)
    assert store.subscription(PID, "old").gone is True
    assert store.subscription(PID, "laptop").gone is False
    for row in store.rows_for("a1"):
        assert_no_secret(row.last_reason)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
