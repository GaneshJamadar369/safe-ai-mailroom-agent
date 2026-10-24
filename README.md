# Safe AI Mailroom

FastAPI service for the GA5 `ga5-mailroom-action-gate/v2` propose/commit contract.

## Local run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:GROQ_API_KEY = "your-new-key"
$env:DATABASE_PATH = ".\data\mailroom.sqlite3"
uvicorn app:app --host 0.0.0.0 --port 8000
```

`.env.example` is documentation only; FastAPI does not load it automatically. Set `GROQ_API_KEY` in the process environment or your hosting provider's secret manager, and never commit a real key.

For local development, `groqapikeys.txt` may contain one key per line and is ignored by git. In Render, configure `GROQ_API_KEYS` as a comma/newline-separated secret instead. On HTTP 401, 403, or 429, the service rotates to the next configured key; it does not retry payload-size failures.

Render logs are JSON lines. Search by `request_id` or `evaluation_id`; set `LOG_LEVEL=DEBUG` temporarily for framework-level diagnostics. Logs intentionally exclude raw dossier text, model prompts, API keys, receipt signatures, and other secret material.

Model requests are automatically split into bounded concurrent batches (`GROQ_BATCH_BYTES`, default `60000`) so a full grader Check does not exceed the provider request limit.

The single public POST endpoint is `/`. Send `Content-Type: application/json` and the JSON envelopes from the assignment. The service uses AI only for semantic decisions; schema checks, canonical hashing, persistence, replay/conflict handling, and Ed25519 receipt validation use deterministic code.

## Deployment

Use Render or another host with writable storage. The included free-plan Render configuration uses `/tmp/safe-ai-mailroom.sqlite3`; this survives requests while the instance is running but is reset when Render replaces the instance. For durable evaluation caching across restarts, attach a paid persistent disk mounted at `/var/data` and set `DATABASE_PATH=/var/data/safe-ai-mailroom.sqlite3`. Add `GROQ_API_KEY` through the dashboard; never commit credentials. Submit the HTTPS service root URL, with no query string or fragment.

## Important limits

This is a starting implementation, not a guarantee of passing hidden evaluations. Test against the real endpoint locally with synthetic dossiers and generated Ed25519 keys before using any grader Check. LLM decision quality and latency on the full batch remain uncertain and should be measured. No outbound email is sent by this service; outcomes record accepted/rejected proposals only, as the grader contract describes.
