# 🌿 AgroDoc: Plant Leaf Disease Detection System

An end-to-end Deep Learning web application designed to detect plant leaf diseases accurately using a fine-tuned ResNet18 model trained on the PlantVillage dataset. The system features a PyTorch backend pipeline, a high-performance FastAPI REST API, and an interactive Streamlit user interface.

---

## 🚀 Features
* **Deep Learning Model:** Fine-tuned ResNet18 architecture capable of classifying 38 different plant health conditions with high accuracy (~98%).
* **FastAPI Backend:** Lightweight and fast REST API endpoint to handle image uploads and return disease predictions with confidence scores.
* **Streamlit Frontend:** User-friendly web interface for real-time leaf image uploading and instant disease diagnosis.
* **GPU Accelerated:** Optimized for CUDA-enabled devices (tested on NVIDIA RTX 2050).

---

## 🛠️ Tech Stack
* **Core:** Python, PyTorch, Torchvision
* **Backend:** FastAPI, Uvicorn, PIL
* **Frontend:** Streamlit, Requests
* **Environment:** Anaconda, Jupyter Notebook

---

## 📂 Project Directory Structure
```text
AgroDoc-Project/
│
├── data/                    # Dataset directory (train/valid/test)
├── saved_models/            # Trained model weights (.pth)
├── src/
│   ├── dataset.py           # Data loading & preprocessing pipelines
│   ├── model.py             # ResNet18 architecture setup
│   ├── train.py             # Training loop script
│   ├── inference.py         # Prediction logic helper class
│   └── main.py              # FastAPI application server
│
├── app.py                   # Streamlit frontend UI
├── requirements.txt         # Project dependencies
├── .gitignore               # Ignored files/folders
└── README.md                # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository & navigate:**
   ```bash
   git clone https://github.com/your-username/AgroDoc-Project.git
   cd AgroDoc-Project
   ```

2. **Create and activate Conda environment:**
   ```bash
   conda create -n agrodoc_env python=3.10 -y
   conda activate agrodoc_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Application

### Step 1: Start the FastAPI Backend
```bash
cd src
python -m uvicorn main:app --reload
```

### Step 2: Start the Streamlit Frontend
```bash
streamlit run app.py
```

---

## 📊 Model Performance
* **Validation Accuracy:** ~98.56%
* **Classes Supported:** 38 Plant & Disease Categories (Apple, Corn, Grape, Potato, Tomato, etc.)
