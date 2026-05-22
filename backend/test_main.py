import os
import shutil
import pytest
from fastapi.testclient import TestClient

# Add backend directory to path so `main` can be imported directly
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app, DATA_DIR, MODELS_DIR

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_teardown():
    """Ensure a clean data directory before and after each test."""
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    yield

    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)


# ── Health Check ──────────────────────────────────────────

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── Upload Samples ────────────────────────────────────────

def test_upload_sample():
    """Upload a single sample and verify it lands on disk."""
    dummy_image = b"fake_image_data"
    response = client.post(
        "/upload-sample",
        data={"class_name": "test_class"},
        files={"image_data": ("test.jpg", dummy_image, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    class_dir = os.path.join(DATA_DIR, "test_class")
    assert os.path.exists(class_dir)
    assert len(os.listdir(class_dir)) == 1


def test_upload_multiple_samples():
    """Upload two images to the same class."""
    for i in range(2):
        client.post(
            "/upload-sample",
            data={"class_name": "multi"},
            files={"image_data": (f"img{i}.jpg", b"data", "image/jpeg")},
        )
    class_dir = os.path.join(DATA_DIR, "multi")
    assert len(os.listdir(class_dir)) == 2


# ── Classes Info ──────────────────────────────────────────

def test_get_classes_info():
    client.post(
        "/upload-sample",
        data={"class_name": "class_a"},
        files={"image_data": ("test1.jpg", b"fake", "image/jpeg")},
    )
    client.post(
        "/upload-sample",
        data={"class_name": "class_b"},
        files={"image_data": ("test2.jpg", b"fake", "image/jpeg")},
    )

    response = client.get("/classes/info")
    assert response.status_code == 200
    classes = response.json()["classes"]
    assert "class_a" in classes
    assert classes["class_a"] == 1
    assert "class_b" in classes


def test_get_classes_info_empty():
    """With no uploads the classes dict should be empty (models dir excluded)."""
    response = client.get("/classes/info")
    assert response.status_code == 200
    classes = response.json()["classes"]
    assert "models" not in classes
    assert classes == {}


# ── Delete Class ──────────────────────────────────────────

def test_delete_class():
    client.post(
        "/upload-sample",
        data={"class_name": "class_to_delete"},
        files={"image_data": ("test.jpg", b"fake", "image/jpeg")},
    )

    response = client.delete("/class/class_to_delete")
    assert response.status_code == 200
    assert not os.path.exists(os.path.join(DATA_DIR, "class_to_delete"))

    # Deleting a non-existent class returns 404
    response = client.delete("/class/nonexistent")
    assert response.status_code == 404


# ── Samples Endpoint ─────────────────────────────────────

def test_get_samples_empty():
    """Requesting samples for a class that doesn't exist returns an empty list."""
    response = client.get("/samples/no_such_class")
    assert response.status_code == 200
    assert response.json()["samples"] == []


# ── Training ─────────────────────────────────────────────

def test_train_model_not_enough_classes():
    """Training with fewer than 2 classes should fail."""
    client.post(
        "/upload-sample",
        data={"class_name": "class_a"},
        files={"image_data": ("test1.jpg", b"fake", "image/jpeg")},
    )
    response = client.post("/train")
    assert response.status_code == 400
    assert "At least 2 classes are required" in response.json()["message"]


# ── Prediction ────────────────────────────────────────────

def test_predict_no_model():
    """Without a trained model the predict endpoint should return 400."""
    response = client.post(
        "/predict",
        files={"image_data": ("test.jpg", b"fake", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Model not trained yet." in response.json()["message"]
