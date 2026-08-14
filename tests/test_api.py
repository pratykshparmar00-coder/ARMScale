from fastapi.testclient import TestClient
from backend.api.main import app, engine
from unittest.mock import patch

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "architecture" in data
    assert "inference_available" in data

def test_platform_endpoint():
    response = client.get("/api/platform")
    assert response.status_code == 200
    data = response.json()
    assert "provider" in data
    assert "architecture" in data
    assert "physical_cores" in data

def test_system_endpoint():
    response = client.get("/api/system")
    assert response.status_code == 200
    data = response.json()
    assert "architecture" in data
    assert "ram_gb" in data

def test_api_model_unloaded():
    engine.is_loaded = False
    response = client.get("/api/model")
    assert response.status_code == 200
    assert response.json()["loaded_status"] is False

def test_generate_unavailable():
    engine.is_loaded = False
    response = client.post("/api/generate", json={"prompt": "Hello"})
    assert response.status_code == 503

@patch("backend.api.main.engine.generate")
def test_generate_success(mock_generate):
    engine.is_loaded = True
    mock_generate.return_value = {
        "response": "Hi",
        "latency_ms": 50.0,
        "tokens_generated": 2,
        "tokens_per_second": 40.0,
        "model": "test",
        "runtime": "mock",
        "architecture": "x86_64",
        "local": True
    }
    
    response = client.post("/api/generate", json={"prompt": "Hello"})
    assert response.status_code == 200
    assert response.json()["response"] == "Hi"
