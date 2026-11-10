from fastapi import FastAPI, File , UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
import shutil
import os 
import uuid
import pickle
from train_model import train_model
from audio_utils import extract_mfcc


app = FastAPI()

# ------------------ PATH SETUP ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATASET_DIR, exist_ok=True)

UPLOAD_FOLDER =os.path.join(BASE_DIR, "audio")
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

# Load model + scaler
model = None
scaler = None
def load_model():
    global model, scaler
    with open(os.path.join(BASE_DIR,"model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

#Load once at startup
load_model()
#-----------Cors----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#------------Routes---------------
@app.get("/")
def home():
    return {"message": "Backend is running 🚀"}

#------------train---------------
@app.post("/train/")
async def retrain_model():

    try:
        train_model()

        # Load the newly trained model
        load_model()

        return {
            "message": "Model retrained successfully!"
        }

    except ValueError as e:
        return {
            "error": str(e)
        }
#------------Register------------
@app.post("/register/")
async def register(user: str, file: UploadFile = File(...)):
    user_folder = os.path.join(DATASET_DIR, user)
    os.makedirs(user_folder, exist_ok=True)
    
    file_path = os.path.join(user_folder, f"{uuid.uuid4()}.wav")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"Saved: {file_path}. Starting Retraining...")
    #---optional Auto retrain -----
    #train_model()
    #Load_model()
    return {"message": f"{user} registered✅"} 

#-------------Login-------------
@app.post("/login/")
async def login(file: UploadFile = File(...)):

    #safety check
    if model is None or scaler is None:
        return {"error": "Model not loaded"}
    temp_path = os.path.join(BASE_DIR,f"temp_{uuid.uuid4().hex}.wav")


    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract MFCC
        mfcc = extract_mfcc(temp_path)
        mfcc_scaled = scaler.transform([mfcc])

        # Predict speaker
        prediction = model.predict(mfcc_scaled)[0]

        # Get probabilities
        probabilities = model.predict_proba(mfcc_scaled)[0]

        # Sort probabilities
        sorted_probs = np.sort(probabilities)[::-1]

        # Highest probability
        confidence = sorted_probs[0] * 100

        # Difference between top two probabilities
        if len(sorted_probs) > 1:
            margin = (sorted_probs[0] - sorted_probs[1]) * 100
        else:
            margin = 0


        print("Prediction:", prediction)
        print("Confidence:", confidence)
        print("Margin:", margin)

        # Cleanup temp file
        os.remove(temp_path)

        # Reject logic
        if confidence < 70:
            return {
                "status": "REJECTED",
                "confidence": round(confidence, 2)
            }

        return {
            "status": "AUTHENTICATED",
            "user": prediction,
            "confidence": round(confidence, 2),
            "margin": round(margin, 2)
        }
    finally:
        # Ensure temp file is removed even if an error occurs
        if os.path.exists(temp_path):
            os.remove(temp_path)


