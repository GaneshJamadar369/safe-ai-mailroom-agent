import base64
import json

from fastapi.testclient import TestClient

import invoice_app as ia


def _headers(token="owner-a"):
    return {"Authorization": f"Bearer {token}", "A2A-Version": "1.0", "Content-Type": "application/a2a+json"}


def _request(message_id="m1"):
    package = {
        "packageId": "pkg-1",
        "vendorName": "Acme Ltd",
        "invoiceNumber": "INV-1",
        "amountMinor": 12500,
        "currency": "INR",
        "documents": "[invoice-main] reconciled and within delegated authority. [approval] [ledger]",
    }
    return {
        "message": {"messageId": message_id, "role": "ROLE_USER", "parts": [
            {"mediaType": ia.IN_BATCH, "data": {"batchId": "batch-1", "policyRevision": "p1", "packages": [package]}}
        ]},
        "configuration": {"returnImmediately": False, "historyLength": 20,
                           "acceptedOutputModes": [ia.OUT_PROPOSALS, ia.OUT_RECEIPTS]},
    }


def test_invoice_card_auth_and_full_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("INVOICE_DATABASE_PATH", str(tmp_path / "invoice.sqlite3"))
    monkeypatch.setattr(ia, "AUTH_TOKEN", "owner-a")
    with TestClient(ia.app) as client:
        card = client.get("/.well-known/agent-card.json")
        assert card.status_code == 200
        assert card.json()["skills"][0]["id"] == "invoice_action_agent"
        assert client.get("/tasks", headers=_headers()).status_code == 200
        assert client.get("/tasks", headers={"Authorization": "Bearer owner-a", "A2A-Version": "1.0", "Content-Type": "application/json"}).status_code == 400
        proposed = client.post("/message:send", headers=_headers(), json=_request())
        assert proposed.status_code == 200
        task = proposed.json()["task"]
        assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        assert len(task["artifacts"]) == 1
        assert client.post("/message:send", headers=_headers(), json=_request()).json() == {"task": task}
        continuation = {"message": {"messageId": "m2", "taskId": task["id"], "contextId": task["contextId"], "role": "ROLE_USER", "parts": [{
            "mediaType": ia.IN_RESULTS, "data": {"batchId": "batch-1", "results": [{
                "packageId": "pkg-1", "actionId": task["artifacts"][0]["parts"][0]["data"]["proposals"][0]["actionId"],
                "action": task["artifacts"][0]["parts"][0]["data"]["proposals"][0]["action"], "outcome": "ACCEPTED", "receiptNonce": "nonce-1"
            }]}
        }]}}
        completed = client.post("/message:send", headers=_headers(), json=continuation)
        assert completed.status_code == 200
        assert completed.json()["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert len(completed.json()["task"]["artifacts"]) == 2
        assert client.post("/message:send", headers={**_headers("owner-b"),}, json=continuation).status_code == 404


def test_invoice_idempotency_conflict_and_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv("INVOICE_DATABASE_PATH", str(tmp_path / "invoice2.sqlite3"))
    monkeypatch.setattr(ia, "AUTH_TOKEN", "owner-a")
    with TestClient(ia.app) as client:
        first = client.post("/message:send", headers=_headers(), json=_request("same"))
        assert first.status_code == 200
        changed = _request("same")
        changed["message"]["parts"][0]["data"]["packages"][0]["amountMinor"] = 999
        assert client.post("/message:send", headers=_headers(), json=changed).status_code == 409
        task_id = first.json()["task"]["id"]
        canceled = client.post(f"/tasks/{task_id}:cancel", headers=_headers())
        assert canceled.status_code == 200
        assert canceled.json()["task"]["status"]["state"] == "TASK_STATE_CANCELED"
        assert client.post(f"/tasks/{task_id}:cancel", headers=_headers()).status_code == 409
