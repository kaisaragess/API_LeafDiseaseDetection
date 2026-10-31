import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client: TestClient):
    """FR-01: Health & Runtime Metadata Check."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"

    expected_models = ["KNN", "SVM", "RandomForest", "ExtraTrees", "Ensemble_Voting"]
    for m in expected_models:
        assert m in data["models_loaded"]

    expected_pipelines = ["B0", "B1", "B2", "P1"]
    for p in expected_pipelines:
        assert p in data["pipelines_supported"]

    env = data["environment"]
    assert "python_version" in env
    assert "opencv_version" in env
    assert "scikit_learn_version" in env


def test_root_endpoint(client: TestClient):
    """Test Root route redirects or returns API welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation" in data
