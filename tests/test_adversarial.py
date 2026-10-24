import asyncio
import base64
import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

import app as mailroom


def _dossier(i, mailbox="support@example.test"):
    return {
        "dossierId": f"d{i}", "partition": "stable_core",
        "receivedAt": "2026-10-23T00:00:00Z", "mailbox": mailbox,
        "objective": "test", "sources": [{"sourceId": f"s{i}", "kind": "email",
        "provenance": "internal", "title": "t", "lines": [
            {"lineId": f"l{i}a", "text": "evidence A"},
            {"lineId": f"l{i}b", "text": "evidence B"}]}],
    }


def _keys():
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return private, {"algorithm": "Ed25519", "publicKeyJwk": {
        "kty": "OKP", "crv": "Ed25519", "x": base64.urlsafe_b64encode(public).decode().rstrip("=")}}


def _proposal(d, action):
    did, mailbox = d["dossierId"], d["mailbox"]
    common = {"dossierId": did, "evidence": [f"l{did[1:]}a", f"l{did[1:]}b"]}
    if action == "create_draft":
        return {**common, "action": action, "target": {"kind": "draft_queue", "id": f"mailbox:{mailbox}"},
                "payload": {"recipient": "a@example.test", "referenceId": did, "status": "open", "template": "order_status"}}
    if action == "update_internal_record":
        return {**common, "action": action, "target": {"kind": "case_record", "id": did},
                "payload": {"field": "delivery_window", "sourceEventId": "evt", "value": "tomorrow"}}
    if action == "send_approved_notice":
        return {**common, "action": action, "target": {"kind": "email", "id": "a@example.test"},
                "payload": {"referenceId": did, "status": "shipped", "template": "approved_delivery_notice"}}
    if action == "request_confirmation":
        return {**common, "action": action, "target": {"kind": "approval_queue", "id": "support"},
                "payload": {"claimedSender": "a@example.test", "questionCode": "VERIFY_REQUEST", "referenceId": did}}
    if action == "quarantine_item":
        return {**common, "action": action, "target": {"kind": "security_queue", "id": "mailroom"},
                "payload": {"artifactId": did, "reasonCode": "INDIRECT_PROMPT_INJECTION"}}
    return {**common, "action": "no_action", "target": None,
            "payload": {"reasonCode": "INFORMATIONAL", "referenceId": did}}


def test_all_action_schemas_validate():
    for i, action in enumerate(mailroom.ACTIONS):
        d = _dossier(i)
        assert mailroom.validate_proposal(d, _proposal(d, action))["action"] == action


def test_model_json_fence_is_narrow():
    assert mailroom.parse_model_json("```json\n{\"proposals\": []}\n```") == {"proposals": []}
    for text in ("prefix {\"proposals\": []}", "```python\n{}\n```"):
        try:
            mailroom.parse_model_json(text)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe model text accepted")


def test_provider_rate_limit_returns_valid_fallback(monkeypatch):
    calls = []

    async def throttled(batch):
        calls.append(len(batch))
        raise RuntimeError("GROQ_HTTP_429")

    monkeypatch.setattr(mailroom, "_ai_decisions_batch", throttled)
    dossiers = [_dossier(10), _dossier(11)]
    decisions = asyncio.run(mailroom.ai_decisions(dossiers))

    assert set(decisions) == {"d10", "d11"}
    assert all(mailroom.validate_proposal(d, decisions[d["dossierId"]]) for d in dossiers)
    assert calls == [2]


def test_proposal_digest_canonicalizes_evidence_order():
    d = _dossier(12)
    p = _proposal(d, "create_draft")
    p["callId"] = "mailroom-test-call-12"
    reordered = {**p, "evidence": list(reversed(p["evidence"]))}
    assert mailroom.proposal_digest(p) == mailroom.proposal_digest(reordered)

