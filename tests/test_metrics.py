import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_experiment_metrics_endpoint(client: TestClient):
    """FR-05: Experiment Metrics & Statistical Dashboard."""
    response = client.get("/api/v1/experiment-metrics")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert "validation_scheme" in data
    assert data["random_seed"] == 42

    dataset_summary = data["dataset_summary"]
    assert dataset_summary["total_images"] == 120
    assert "Bacterial leaf blight" in dataset_summary["classes"]
    assert "Brown spot" in dataset_summary["classes"]
    assert "Leaf smut" in dataset_summary["classes"]

    macro_f1 = data["macro_f1_summary"]
    assert len(macro_f1) >= 4
    for item in macro_f1:
        assert item["pipeline"] in ["B0", "B1", "B2", "P1"]
        assert 0.0 <= item["mean_macro_f1"] <= 1.0
        assert len(item["ci_95"]) == 2

    stat_tests = data["statistical_tests"]
    assert "friedman_p_value" in stat_tests
    assert stat_tests["friedman_p_value"] < 0.05

    wilcoxon = stat_tests["posthoc_wilcoxon_p1_vs_b0"]
    assert wilcoxon["delta_macro_f1"] > 0
    assert wilcoxon["significant"] is True
