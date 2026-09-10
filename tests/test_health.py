"""Tests for the liveness endpoint uptime monitors poll."""

from fastapi.testclient import TestClient

from nba_winprob.api.server import app


def test_healthz_answers_get():
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_answers_head():
    """UptimeRobot and most monitors probe with HEAD by default.

    FastAPI's ``@app.get`` does not register HEAD, so a GET-only route answers
    405 and the monitor reports a false outage.
    """
    with TestClient(app) as client:
        response = client.request("HEAD", "/healthz")
    assert response.status_code == 200


def test_healthz_is_not_rate_limited():
    with TestClient(app) as client:
        codes = {client.get("/healthz").status_code for _ in range(60)}
    assert codes == {200}
