from fastapi.testclient import TestClient

from tradingagents.web.app import app


def test_health_endpoint() -> None:
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_active_run_endpoint_returns_null_without_running_task(monkeypatch) -> None:
    class EmptyRunManager:
        def active(self):
            return None

    monkeypatch.setattr(app.state, "run_manager", EmptyRunManager())
    response = TestClient(app).get("/api/runs/active")

    assert response.status_code == 200
    assert response.json() is None


def test_active_run_endpoint_returns_snapshot(monkeypatch) -> None:
    class ActiveRecord:
        def snapshot(self):
            return {"run_id": "active-run", "status": "running"}

    class ActiveRunManager:
        def active(self):
            return ActiveRecord()

    monkeypatch.setattr(app.state, "run_manager", ActiveRunManager())
    response = TestClient(app).get("/api/runs/active")

    assert response.status_code == 200
    assert response.json() == {"run_id": "active-run", "status": "running"}
