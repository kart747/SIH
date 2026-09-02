Minimal FastAPI server for streaming smartphone sensors and EKF fail-safe.

Run:

```
python -m uvicorn app.main:app --reload --port 8000
```

Endpoints:
- `POST /ingest` - JSON sensor payload
- `GET /health` - health check
