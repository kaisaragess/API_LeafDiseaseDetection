import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_guard_rejects_empty_file(client: TestClient):
    """Verifies that an empty upload is rejected with 400 Bad Request."""
    response = client.post(
        "/api/v1/predict",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "INVALID_IMAGE_FILE"


def test_guard_rejects_fake_corrupt_format(client: TestClient):
    """Verifies that a non-image file is rejected with 400 Bad Request."""
    fake_bytes = b"Hello, this is just plain text content disguised as an image."
    response = client.post(
        "/api/v1/predict",
        files={"image": ("fake.jpg", fake_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "INVALID_IMAGE_FILE"


def test_guard_rejects_non_leaf_image_422(client: TestClient):
    """
    FR-03: Out-of-Distribution Guard.
    An image without rice leaf content (e.g. blank background where leaf_area_ratio < 0.08)
    must be rejected with HTTP 422 Unprocessable Entity and INVALID_LEAF_OBJECT code.
    """
    # Plain white background image (0% leaf area)
    white_img = np.full((384, 128, 3), 255, dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", white_img)
    img_bytes = buffer.tobytes()

    response = client.post(
        "/api/v1/predict",
        files={"image": ("blank_white.jpg", img_bytes, "image/jpeg")},
        data={"pipeline_mode": "P1", "classifier": "SVM"},
    )

    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "INVALID_LEAF_OBJECT"
    assert "leaf_area_ratio" in data
    assert data["leaf_area_ratio"] < 0.08
