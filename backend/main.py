import os
import uuid
import shutil
import pickle
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="Teachable Machine Backend")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "backend_data"
MODELS_DIR = os.path.join(DATA_DIR, "models")

# Ensure base directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Machine Learning Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.MobileNet_V3_Small_Weights.DEFAULT
backbone = models.mobilenet_v3_small(weights=weights).features
backbone.to(device)
backbone.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        features = backbone(input_batch)
    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
    return features.flatten().cpu().numpy()

@app.post("/upload-sample")
async def upload_sample(class_name: str = Form(...), image_data: UploadFile = File(...)):
    """
    Endpoint to receive an image and save it under the specified class name directory.
    """
    # Create directory for the specific class
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Generate unique filename
    file_ext = image_data.filename.split(".")[-1] if "." in image_data.filename else "jpg"
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(class_dir, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image_data.file, buffer)
        
    return {"status": "success", "message": f"Image uploaded to {class_name}", "filename": unique_filename}

@app.get("/classes/info")
async def get_classes_info():
    """
    Get current classes and the number of samples in each.
    """
    classes_info = {}
    if os.path.exists(DATA_DIR):
        for d in os.listdir(DATA_DIR):
            d_path = os.path.join(DATA_DIR, d)
            if os.path.isdir(d_path) and d != "models":
                num_samples = len([f for f in os.listdir(d_path) if os.path.isfile(os.path.join(d_path, f))])
                classes_info[d] = num_samples
    return {"classes": classes_info}

@app.delete("/class/{class_name}")
async def delete_class(class_name: str):
    """
    Delete a class directory and all its samples.
    """
    class_dir = os.path.join(DATA_DIR, class_name)
    if os.path.exists(class_dir) and os.path.isdir(class_dir):
        shutil.rmtree(class_dir)
        return {"status": "success", "message": f"Class {class_name} deleted."}
    return JSONResponse(status_code=404, content={"status": "error", "message": "Class not found."})

@app.post("/train")
async def train_model():
    """
    Endpoint to trigger model training.
    """
    X = []
    y = []
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d != "models"]
    if len(classes) < 2:
        return JSONResponse(status_code=400, content={"status": "error", "message": "At least 2 classes are required for training."})
        
    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            try:
                features = extract_features(img_path)
                X.append(features)
                y.append(cls)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
                
    if not X:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No valid images found for training."})
        
    # Train Logistic Regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    
    # Save the model
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    
    return {"status": "training_started", "message": "Model trained successfully", "classes": classes}

@app.post("/predict")
async def predict(image_data: UploadFile = File(...)):
    """
    Endpoint to predict the class of an uploaded image.
    """
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    if not os.path.exists(model_path):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Model not trained yet."})
        
    try:
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
            
        # Save temporarily
        temp_path = os.path.join(DATA_DIR, f"temp_{uuid.uuid4()}.jpg")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image_data.file, buffer)
            
        features = extract_features(temp_path)
        os.remove(temp_path)
        
        prediction = clf.predict([features])[0]
        probabilities = clf.predict_proba([features])[0]
        
        prob_dict = {cls: float(prob) for cls, prob in zip(clf.classes_, probabilities)}
        
        return {"prediction": prediction, "probabilities": prob_dict}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
