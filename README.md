# 🧠 Teachable Machine Clone

> **Train your own image classifier — no ML experience required.**
> Just point your webcam, capture a few samples per category, hit "Train", and get real-time predictions in seconds.

---

## 📌 Table of Contents

- [Problem It Solves](#-problem-it-solves)
- [What Is This Project?](#-what-is-this-project)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How It Works (Architecture)](#-how-it-works-architecture)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [How to Use](#-how-to-use)
- [API Reference](#-api-reference)
- [Running Tests](#-running-tests)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Problem It Solves

### The Challenge

Building an image classification model traditionally requires:

- Deep knowledge of machine learning and neural networks
- Proficiency in Python, TensorFlow/PyTorch, and data preprocessing
- A powerful GPU and hours of training time
- Hundreds or thousands of labeled images
- Understanding of model architecture, hyperparameters, and evaluation metrics

**This puts AI out of reach for students, designers, hobbyists, and non-technical creators** who have great ideas but lack the technical expertise to bring them to life.

### The Solution

This project is a **Google Teachable Machine clone** — a beginner-friendly web application that lets anyone train a custom image classification model in **3 simple steps**:

1. **Capture** — Use your webcam or upload images for each category
2. **Train** — Click one button and wait a few seconds
3. **Predict** — Point your webcam at something and see real-time results

No code. No terminal. No ML knowledge. Just a browser.

### Real-World Use Cases

| Use Case | Example |
|----------|---------|
| 🏫 **Education** | Students learning how AI classification works hands-on |
| ♻️ **Waste Sorting** | Train a model to distinguish recyclable vs non-recyclable items |
| 🐱 **Pet Recognition** | Classify different pet breeds from webcam |
| 🤟 **Sign Language** | Detect hand signs for basic sign language translation |
| 🏭 **Quality Control** | Identify defective vs good products on a production line |
| 🎨 **Art Style** | Classify paintings by art style or artist |

---

## 💡 What Is This Project?

This is a full-stack web application inspired by [Google's Teachable Machine](https://teachablemachine.withgoogle.com/). It uses **Transfer Learning** with a pre-trained **MobileNetV3** neural network as a feature extractor, combined with a fast **Logistic Regression** classifier on top — giving you the power of deep learning with the speed of classical ML.

**In plain English:**
- MobileNetV3 (trained by Google on millions of images) looks at your photos and extracts meaningful "features" — patterns like edges, textures, shapes
- Logistic Regression takes those features and learns to separate them into your custom classes
- The result: a model that can be trained in seconds on just a handful of images

---

## ✨ Key Features

- 📷 **Webcam capture** — Take training samples directly from your browser
- 📁 **File upload** — Drag and drop existing images (JPG, JPEG, PNG)
- ➕ **Dynamic classes** — Add, rename, or delete as many categories as you want
- 🏋️ **One-click training** — Train your model with a single button press
- 🔮 **Real-time prediction** — Test via live webcam or image upload with confidence bars
- 🖼️ **Sample preview** — See thumbnails of uploaded training images
- 🩺 **Health check API** — Verify backend connectivity at any time
- 🧪 **Test suite** — 9 automated tests covering all API endpoints

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | [Streamlit](https://streamlit.io/) | Interactive web UI with webcam integration |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance REST API |
| **ML Model** | [PyTorch](https://pytorch.org/) + [scikit-learn](https://scikit-learn.org/) | Feature extraction (MobileNetV3) + Classification (Logistic Regression) |
| **Image Processing** | [Pillow](https://pillow.readthedocs.io/) | Image loading and preprocessing |
| **Server** | [Uvicorn](https://www.uvicorn.org/) | ASGI server for FastAPI |
| **Testing** | [pytest](https://docs.pytest.org/) | Unit testing framework |

---

## 📁 Project Structure

```
Teachable-Machine/
│
├── backend/
│   ├── main.py              # FastAPI application — all API endpoints
│   └── test_main.py         # Pytest test suite (9 tests)
│
├── frontend/
│   └── app.py               # Streamlit UI — webcam, upload, train, predict
│
├── backend_data/             # (auto-created at runtime, gitignored)
│   ├── class_name_1/         #   Images for class 1
│   ├── class_name_2/         #   Images for class 2
│   └── models/
│       └── model.pkl         #   Trained classifier
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # You are here!
```

> **Note:** The `backend_data/` folder is created automatically when you start the backend. It stores your training images and the trained model. It is **gitignored** so your images won't be pushed to GitHub.

---

## ⚙️ How It Works (Architecture)

```
┌─────────────────────┐         HTTP         ┌─────────────────────────┐
│                     │  ◄──────────────────► │                         │
│   Streamlit UI      │   POST /upload-sample │   FastAPI Backend       │
│   (localhost:8501)  │   POST /train         │   (localhost:8000)      │
│                     │   POST /predict       │                         │
│  ┌───────────────┐  │   GET  /classes/info  │  ┌───────────────────┐  │
│  │ 📷 Webcam     │  │   GET  /samples/{cls} │  │ MobileNetV3       │  │
│  │ 📁 Upload     │  │   DELETE /class/{cls} │  │ (Feature Extract) │  │
│  │ 🏋️ Train Btn  │  │                       │  │         ↓         │  │
│  │ 🔮 Predict    │  │                       │  │ Logistic Regress. │  │
│  └───────────────┘  │                       │  │ (Classification)  │  │
│                     │                       │  └───────────────────┘  │
└─────────────────────┘                       └─────────────────────────┘
```

**Step-by-step flow:**
1. You capture/upload images through the Streamlit UI
2. Images are sent to FastAPI via `POST /upload-sample` and saved to disk
3. When you click "Train", the backend extracts features from all images using MobileNetV3
4. A Logistic Regression model is trained on those features and saved as `model.pkl`
5. For predictions, the uploaded/captured image is feature-extracted and classified against the trained model
6. Confidence percentages for each class are returned and displayed as progress bars

---

## 📋 Prerequisites

Before starting, make sure you have:

| Requirement | Version | Check Command |
|------------|---------|---------------|
| **Python** | 3.9 or higher | `python --version` |
| **pip** | Latest | `pip --version` |
| **Git** | Any | `git --version` |
| **Webcam** | Built-in or USB | — |
| **Browser** | Chrome / Edge / Firefox | — |

> **💡 Tip:** If `python --version` shows Python 2.x or gives an error, try `python3 --version` instead. On Windows, you can install Python from [python.org](https://www.python.org/downloads/).

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Aayan-Ahmed-17/Teachable-Machine.git
cd Teachable-Machine
```

### Step 2: Create a Virtual Environment

A virtual environment keeps this project's dependencies separate from your system Python.

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (Git Bash):**
```bash
python -m venv venv
source venv/Scripts/activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> After activation, you should see `(venv)` at the beginning of your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs FastAPI, Streamlit, PyTorch, scikit-learn, and all other required packages. It may take a few minutes (PyTorch is a large download).

### Step 4: Start the Backend Server

Open a terminal, activate the virtual environment, and run:

```bash
cd backend
uvicorn main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Started reloader process
```

> **✅ Keep this terminal open.** The backend must be running for the frontend to work.

### Step 5: Start the Frontend (New Terminal)

Open a **second** terminal, activate the virtual environment, and run:

```bash
cd frontend
streamlit run app.py
```

You should see:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Your browser should open automatically. If not, go to **http://localhost:8501**.

---

## 🎮 How to Use

### Step 1: Create Your Classes

When the app opens, you'll see two default classes: **"Class 1"** and **"Class 2"**.

- **Rename** a class by clicking on its name and typing a new one (e.g., "Cat", "Dog")
- **Add more classes** by clicking the **"➕ Add a class"** button at the bottom
- **Delete** a class by clicking the **🗑️** icon next to its name

> **💡 Tip:** Use descriptive names! The class name is what the model will predict (e.g., "Thumbs Up" vs "Thumbs Down").

### Step 2: Add Training Images

For each class, you have two options:

#### Option A: Webcam Capture (Recommended)
1. Click the **"📷 Webcam"** tab under a class
2. Allow browser camera access when prompted
3. Position the object/gesture in front of your webcam
4. Click the **camera button** to capture
5. Repeat to add more samples (5-10 per class is a good starting point)

#### Option B: File Upload
1. Click the **"📁 Upload"** tab under a class
2. Drag and drop images, or click to browse your files
3. Select one or multiple images (JPG, JPEG, PNG)
4. Click **"Upload Files"** to send them to the backend

> **⚠️ Important:** You need **at least 2 classes** with **at least 1 image each** to train. For best results, aim for **5-20 images per class**.

### Step 3: Train Your Model

1. Look at the **"Training"** column in the middle
2. Once you have enough samples, the **"Train Model"** button becomes active
3. Click it and wait a few seconds
4. You'll see **"Model Trained ✅"** when it's done

> **💡 Tip:** The advanced settings (Epochs, Batch Size, Learning Rate) are for display — the current backend uses Logistic Regression which trains instantly. These would apply if you extend the project to use fine-tuning.

### Step 4: Test Your Model

1. Look at the **"Preview"** column on the right
2. Choose **"📷 Live Webcam"** or **"📁 Test Image"**
3. Capture or upload an image to classify
4. The model will show **prediction confidence bars** for each class

**Example output:**
```
Cat:  92.3%  ████████████████████░░░
Dog:   7.7%  ██░░░░░░░░░░░░░░░░░░░░
```

### Step 5: Iterate and Improve

Not happy with the accuracy? Here's what to do:

- ✅ Add **more training images** (variety helps!)
- ✅ Make sure images are **well-lit** and **focused**
- ✅ Include different **angles** and **backgrounds**
- ✅ Retrain after adding more samples
- ❌ Avoid blurry or dark images
- ❌ Avoid classes that look too similar with very few samples

---

## 📡 API Reference

The FastAPI backend exposes the following REST endpoints. You can also explore them interactively at **http://localhost:8000/docs** (Swagger UI).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/upload-sample` | Upload an image to a class |
| `GET` | `/classes/info` | Get all classes and their sample counts |
| `DELETE` | `/class/{class_name}` | Delete a class and all its images |
| `GET` | `/samples/{class_name}` | Get image URLs for a specific class |
| `POST` | `/train` | Train the model on uploaded data |
| `POST` | `/predict` | Predict the class of an uploaded image |

### Example API Calls (using `curl`)

**Upload a sample:**
```bash
curl -X POST http://localhost:8000/upload-sample \
  -F "class_name=cat" \
  -F "image_data=@photo.jpg"
```

**Train the model:**
```bash
curl -X POST http://localhost:8000/train
```

**Make a prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "image_data=@test_image.jpg"
```

**Check health:**
```bash
curl http://localhost:8000/health
```

---

## 🧪 Running Tests

The project includes 9 automated tests that verify all backend endpoints.

```bash
cd backend
python -m pytest test_main.py -v
```

**Expected output:**
```
test_main.py::test_health_check                      PASSED  [ 11%]
test_main.py::test_upload_sample                     PASSED  [ 22%]
test_main.py::test_upload_multiple_samples           PASSED  [ 33%]
test_main.py::test_get_classes_info                  PASSED  [ 44%]
test_main.py::test_get_classes_info_empty            PASSED  [ 55%]
test_main.py::test_delete_class                      PASSED  [ 66%]
test_main.py::test_get_samples_empty                 PASSED  [ 77%]
test_main.py::test_train_model_not_enough_classes    PASSED  [ 88%]
test_main.py::test_predict_no_model                  PASSED  [100%]

========================= 9 passed in ~10s =========================
```

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><strong>❌ "command not found: uvicorn"</strong></summary>

Your virtual environment is not activated. Run:
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Git Bash
source venv/Scripts/activate

# macOS / Linux
source venv/bin/activate
```
Then try again.
</details>

<details>
<summary><strong>❌ "Error loading ASGI app. Could not import module 'main'"</strong></summary>

You're running `uvicorn` from the wrong directory. Make sure you are inside the `backend/` folder:
```bash
cd backend
uvicorn main:app --reload
```
</details>

<details>
<summary><strong>❌ "ConnectionError: Cannot connect to backend"</strong></summary>

The FastAPI backend isn't running. Start it first in a separate terminal:
```bash
cd backend
uvicorn main:app --reload
```
Then start the frontend.
</details>

<details>
<summary><strong>❌ Webcam not working in Streamlit</strong></summary>

- Make sure your browser has **camera permissions** enabled
- Try using **Chrome** or **Edge** (best Streamlit compatibility)
- Check that no other app is using your webcam
- If using a URL other than `localhost`, Streamlit requires **HTTPS** for camera access
</details>

<details>
<summary><strong>❌ "At least 2 classes are required for training"</strong></summary>

You need to upload images to at least 2 different classes before training. Each class needs at least 1 image.
</details>

<details>
<summary><strong>❌ Low prediction accuracy</strong></summary>

- Add more training images (10-20 per class)
- Ensure good lighting and focus
- Include different angles and backgrounds
- Make sure your classes are visually distinct
</details>

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** this repository
2. **Create a branch**: `git checkout -b feature/my-feature`
3. **Make your changes** and test them
4. **Commit**: `git commit -m "Add: my new feature"`
5. **Push**: `git push origin feature/my-feature`
6. **Open a Pull Request**

### Ideas for Contribution
- 🌙 Dark mode for the Streamlit UI
- 📊 Model accuracy/metrics dashboard
- 💾 Export/import trained models
- 🎥 Continuous webcam prediction (video stream)
- 🔄 Fine-tuning option for better accuracy
- 📱 Mobile-responsive layout

---

## 📄 License

This project is open source and available for educational purposes.

---

<div align="center">

**Built with ❤️ using FastAPI + Streamlit + PyTorch**

⭐ Star this repo if it helped you learn something new!

</div>
