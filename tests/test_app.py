import base64
import json
import os
from pathlib import Path

import pytest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

os.environ["DATABASE_PATH"] = str(Path(__file__).parent / "test.sqlite3")
import app as mailroom

@pytest.fixture(autouse=True)
def clean_db():
    p = Path(os.environ["DATABASE_PATH"])
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(str(p) + suffix).unlink()
        except FileNotFoundError:
            pass
    mailroom.init_db()
    yield
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(str(p) + suffix).unlink()
        except FileNotFoundError:
            pass


def keypair():
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    jwk = {"kty": "OKP", "crv": "Ed25519", "x": base64.urlsafe_b64encode(pub).decode().rstrip("=")}
    return priv, {"algorithm": "Ed25519", "publicKeyJwk": jwk}


def dossier():
    return {"dossierId":"d1", "partition":"stable_core", "receivedAt":"2026-10-23T00:00:00Z", "mailbox":"support@example.test", "objective":"Prepare a draft order update", "sources":[{"sourceId":"s1", "kind":"email", "provenance":"customer", "title":"Order update", "lines":[{"lineId":"l1", "text":"Customer asks for order status; reference order-1."},{"lineId":"l2", "text":"Current status is shipped."}]}]}


def request_body(d, verifier, evaluation="ev1"):
    return {"profile":mailroom.PROFILE,"operation":"propose","evaluationId":evaluation,"receiptVerifier":verifier,"corpus":{"coreId":"core","auditId":"audit","stableCount":1,"freshCount":0},"allowedActions":["create_draft","update_internal_record","send_approved_notice","request_confirmation","quarantine_item","no_action"],"dossiers":d}


def generated_decision(d):
    return {"dossierId":d["dossierId"], "action":"create_draft", "target":{"kind":"draft_queue","id":"mailbox:"+d["mailbox"]}, "payload":{"recipient":"customer@example.test","referenceId":"order-1","status":"shipped","template":"order_status"}, "evidence":["l1","l2"]}


def sign_receipt(priv, evaluation_id, input_digest, receipt):
    inner = {k:v for k,v in receipt.items() if k != "receiptSignature"}
    signed = {"profile":mailroom.PROFILE,"evaluationId":evaluation_id,"inputDigest":input_digest,"receipt":inner}
    sig = priv.sign(mailroom.canon(signed).encode())
    receipt["receiptSignature"] = base64.b64encode(sig).decode()
    return receipt


def test_propose_and_cache(monkeypatch):
    priv, verifier = keypair()
    calls = {"n":0}
    async def fake_ai(ds):
        calls["n"] += 1
        return {d["dossierId"]:generated_decision(d) for d in ds}
    monkeypatch.setattr(mailroom, "ai_decisions", fake_ai)
    client = TestClient(mailroom.app)
    body = request_body([dossier()], verifier)
    a = client.post("/", json=body).json()
    assert a["status"] == "awaiting_receipts"
    assert len(a["proposals"]) == 1
    b = client.post("/", json=body).json()
    assert a == b and calls["n"] == 1


def test_signed_commit_and_replay(monkeypatch):
    priv, verifier = keypair()
    async def fake_ai(ds):
        return {d["dossierId"]:generated_decision(d) for d in ds}
    monkeypatch.setattr(mailroom, "ai_decisions", fake_ai)
    client = TestClient(mailroom.app)
    req = request_body([dossier()], verifier)
    proposed = client.post("/", json=req).json()
    p = proposed["proposals"][0]
    rec = {"dossierId":p["dossierId"],"callId":p["callId"],"action":p["action"],"accepted":True,"proposalDigest":mailroom.proposal_digest(p),"receiptId":"nonce-123"}
    rec = sign_receipt(priv, "ev1", proposed["inputDigest"], rec)
    commit = {"profile":mailroom.PROFILE,"operation":"commit","evaluationId":"ev1","inputDigest":proposed["inputDigest"],"receipts":[rec]}
    a = client.post("/", json=commit).json()
    assert a["status"] == "completed" and a["outcomes"][0]["status"] == "executed"
    assert client.post("/", json=commit).json() == a


def test_bad_signature_rejected_without_completion(monkeypatch):
    _, verifier = keypair()
    async def fake_ai(ds):
        return {d["dossierId"]:generated_decision(d) for d in ds}
    monkeypatch.setattr(mailroom, "ai_decisions", fake_ai)
    client = TestClient(mailroom.app)
    proposed = client.post("/", json=request_body([dossier()], verifier)).json()
    p = proposed["proposals"][0]
    rec = {"dossierId":p["dossierId"],"callId":p["callId"],"action":p["action"],"accepted":True,"proposalDigest":mailroom.proposal_digest(p),"receiptId":"nonce-123","receiptSignature":base64.b64encode(b"x"*64).decode()}
    result = client.post("/", json={"profile":mailroom.PROFILE,"operation":"commit","evaluationId":"ev1","inputDigest":proposed["inputDigest"],"receipts":[rec]})
    assert result.status_code == 422
    con = mailroom.connect()
    try:
        row = con.execute("select commit_response_json from evaluations where evaluation_id='ev1'").fetchone()
        assert row[0] is None
    finally:
        con.close()


def test_same_evaluation_changed_body_conflicts(monkeypatch):
    _, verifier = keypair()
    async def fake_ai(ds):
        return {d["dossierId"]:generated_decision(d) for d in ds}
    monkeypatch.setattr(mailroom, "ai_decisions", fake_ai)
    client = TestClient(mailroom.app)
    body = request_body([dossier()], verifier)
    assert client.post("/", json=body).status_code == 200
    changed = json.loads(json.dumps(body)); changed["dossiers"][0]["objective"] = "different"
    assert client.post("/", json=changed).status_code == 409


def test_invalid_evidence_rejected():
    d = dossier()
    bad = generated_decision(d); bad["evidence"] = ["not-a-line"]
    with pytest.raises(ValueError):
        mailroom.validate_proposal(d, bad)
