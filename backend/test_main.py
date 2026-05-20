import os
import shutil
import pytest
from fastapi.testclient import TestClient

# We need to run tests from the root directory so imports work
# We will run `pytest backend/test_main.py` from New folder
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app, DATA_DIR, MODELS_DIR

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup: ensure clean slate
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    yield
    
    # Teardown
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

def test_upload_sample():
    dummy_image = b"fake_image_data"
    response = client.post(
        "/upload-sample",
        data={"class_name": "test_class"},
        files={"image_data": ("test.jpg", dummy_image, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    class_dir = os.path.join(DATA_DIR, "test_class")
    assert os.path.exists(class_dir)
    assert len(os.listdir(class_dir)) == 1

def test_get_classes_info():
    client.post(
        "/upload-sample",
        data={"class_name": "class_a"},
        files={"image_data": ("test1.jpg", b"fake", "image/jpeg")}
    )
    client.post(
        "/upload-sample",
        data={"class_name": "class_b"},
        files={"image_data": ("test2.jpg", b"fake", "image/jpeg")}
    )
    
    response = client.get("/classes/info")
    assert response.status_code == 200
    classes = response.json()["classes"]
    assert "class_a" in classes
    assert classes["class_a"] == 1
    assert "class_b" in classes

def test_delete_class():
    client.post(
        "/upload-sample",
        data={"class_name": "class_to_delete"},
        files={"image_data": ("test.jpg", b"fake", "image/jpeg")}
    )
    
    response = client.delete("/class/class_to_delete")
    assert response.status_code == 200
    assert not os.path.exists(os.path.join(DATA_DIR, "class_to_delete"))
    
    response = client.delete("/class/nonexistent")
    assert response.status_code == 404

def test_train_model_not_enough_classes():
    client.post(
        "/upload-sample",
        data={"class_name": "class_a"},
        files={"image_data": ("test1.jpg", b"fake", "image/jpeg")}
    )
    response = client.post("/train")
    assert response.status_code == 400
    assert "At least 2 classes are required" in response.json()["message"]

def test_predict_no_model():
    # Without training, model.pkl shouldn't exist
    response = client.post(
        "/predict",
        files={"image_data": ("test.jpg", b"fake", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Model not trained yet." in response.json()["message"]
