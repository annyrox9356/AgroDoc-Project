from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import AgroDocPredictor
import os

app = FastAPI(title="AgroDoc API", description="Plant Leaf Disease Detection using PyTorch & ResNet18")

# CORS middleware enable karna taaki Streamlit ya koi bhi frontend easily connect kar sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 38 Classes ki list (PlantVillage dataset ki standard classes)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Model path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'agrodoc_best_model.pth')

# Predictor instance initialize karna (Agar model train hokar save ho gaya hoga toh load ho jayega)
predictor = None
if os.path.exists(MODEL_PATH):
    predictor = AgroDocPredictor(model_path=MODEL_PATH, num_classes=len(CLASS_NAMES))
    print("[*] AgroDoc model successfully loaded into FastAPI!")
else:
    print("[!] Warning: Model weights (.pth) not found yet. Please complete training first.")

@app.get("/")
def home():
    return {"message": "Welcome to AgroDoc API! Use /predict endpoint to classify plant leaf diseases."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model is not loaded yet. Train the model first.")
    
    try:
        # Upload ki hui image ke bytes read karna
        image_bytes = await file.read()
        
        # Inference class se prediction nikalna
        predicted_class, confidence = predictor.predict_image(image_bytes, CLASS_NAMES)
        
        return {
            "filename": file.filename,
            "predicted_disease": predicted_class,
            "confidence_score": round(confidence, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))