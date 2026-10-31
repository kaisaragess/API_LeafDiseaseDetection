from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_LEAF = FIXTURES_DIR / "brown_spot.jpg"
BLIGHT_LEAF = FIXTURES_DIR / "bacterial_leaf_blight.jpg"
SMUT_LEAF = FIXTURES_DIR / "leaf_smut.jpg"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_predict_default_pipeline_and_classifier(client: TestClient):
    """
    FR-02: Single-Image Diagnosis & Visual Inspection.
    Tests default parameters (P1 + SVM).
    """
    assert SAMPLE_LEAF.exists(), "Sample test image fixture is missing."
    with open(SAMPLE_LEAF, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/v1/predict",
        files={"image": ("brown_spot.jpg", img_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"

    # Meta telemetry check
    meta = data["meta"]
    assert meta["pipeline_used"] == "P1"
    assert meta["classifier_used"] == "SVM"
    assert meta["image_resize_shape"] == [384, 128]
    assert meta["preprocessing_time_ms"] > 0
    assert meta["prediction_time_ms"] > 0
    assert meta["total_features_extracted"] == 142

    # Prediction check
    pred = data["prediction"]
    assert pred["predicted_class"] in [
        "Bacterial leaf blight",
        "Brown spot",
        "Leaf smut",
    ]
    assert 0.0 <= pred["confidence_score"] <= 1.0
    probs = pred["probabilities"]
    assert len(probs) == 3
    assert abs(sum(probs.values()) - 1.0) < 0.05

    # Feature summary check
    feats = data["feature_summary"]
    assert feats["leaf_area_ratio"] >= 0.08
    assert feats["lesion_area_ratio"] >= 0.0
    assert "dominant_hsv_mean" in feats
    assert "glcm_contrast_mean" in feats
    assert "estimated_lesion_blobs" in feats

    # 6-step Base64 visualizations check
    vis = data["visualizations_base64"]
    required_steps = [
        "1_resized_input",
        "2_gray_world",
        "3_clahe_lab",
        "4_leaf_mask_morphology",
        "5_lesion_mask",
        "6_final_segmented",
    ]
    for step in required_steps:
        assert step in vis
        assert vis[step].startswith("data:image/")

    # Scientific disclaimer check
    assert "purwarupa akademik" in data["disclaimer"].lower()


@pytest.mark.parametrize("pipeline", ["B0", "B1", "B2", "P1"])
def test_predict_all_pipelines(client: TestClient, pipeline: str):
    """Verifies that all 4 pipeline scenarios execute successfully."""
    with open(SAMPLE_LEAF, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/v1/predict",
        files={"image": ("sample.jpg", img_bytes, "image/jpeg")},
        data={"pipeline_mode": pipeline, "classifier": "SVM"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["pipeline_used"] == pipeline
    assert data["status"] == "success"


@pytest.mark.parametrize(
    "classifier", ["KNN", "SVM", "RandomForest", "ExtraTrees", "Ensemble_Voting"]
)
def test_predict_all_classifiers(client: TestClient, classifier: str):
    """Verifies that all 5 machine learning estimators execute successfully."""
    with open(SAMPLE_LEAF, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/v1/predict",
        files={"image": ("sample.jpg", img_bytes, "image/jpeg")},
        data={"pipeline_mode": "P1", "classifier": classifier},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["classifier_used"] == classifier
    assert data["prediction"]["confidence_score"] > 0


def test_predict_with_perturbations(client: TestClient):
    """Tests the on-the-fly robustness simulator (brightness, saturation, blur)."""
    with open(SAMPLE_LEAF, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/v1/predict",
        files={"image": ("sample.jpg", img_bytes, "image/jpeg")},
        data={
            "pipeline_mode": "P1",
            "classifier": "SVM",
            "brightness_shift": 0.15,
            "saturation_shift": -0.10,
            "apply_blur": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    pert = data["perturbation_applied"]
    assert pert["brightness_shift"] == 0.15
    assert pert["saturation_shift"] == -0.10
    assert pert["apply_blur"] is True


def test_compare_pipelines_endpoint(client: TestClient):
    """
    FR-04: Multi-Pipeline Comparative Stress Test.
    Runs 1 leaf image through 4 pipelines clean vs perturbed (-20% brightness).
    """
    with open(SAMPLE_LEAF, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/v1/compare-pipelines",
        files={"image": ("sample.jpg", img_bytes, "image/jpeg")},
        data={"classifier": "SVM", "brightness_shift": -0.20},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["classifier_tested"] == "SVM"
    assert "Brightness -20%" in data["perturbation_condition"]

    comparisons = data["comparisons"]
    for p in ["B0", "B1", "B2", "P1"]:
        assert p in comparisons
        item = comparisons[p]
        assert "clean_class" in item
        assert "perturbed_class" in item
        assert "clean_confidence" in item
        assert "perturbed_confidence" in item
        assert "is_consistent" in item
        assert "confidence_drop" in item

    assert "scientific_insight" in data
    assert len(data["scientific_insight"]) > 10


@pytest.mark.parametrize(
    "fixture_filename",
    ["bacterial_leaf_blight.jpg", "brown_spot.jpg", "leaf_smut.jpg"],
)
def test_predict_all_disease_sample_fixtures(client: TestClient, fixture_filename: str):
    """Tests inference on all 3 real disease class sample fixtures."""
    fixture_path = FIXTURES_DIR / fixture_filename
    assert fixture_path.exists(), f"Fixture {fixture_filename} missing."

    with open(fixture_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/v1/predict",
        files={"image": (fixture_filename, img_bytes, "image/jpeg")},
        data={"pipeline_mode": "P1", "classifier": "SVM"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["prediction"]["predicted_class"] in [
        "Bacterial leaf blight",
        "Brown spot",
        "Leaf smut",
    ]
    assert data["feature_summary"]["leaf_area_ratio"] >= 0.08
