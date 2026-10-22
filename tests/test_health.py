from app.main import app, health_check


def test_health_route_is_registered_and_returns_ready_status() -> None:
    assert any(route.path == "/api/health" for route in app.routes)
    assert health_check() == {"status": "ok"}

