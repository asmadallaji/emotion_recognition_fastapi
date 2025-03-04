from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.api.load_model import model, EMOTIONS
from src.api.image_processing import preprocess_image
import numpy as np
import io
from PIL import Image

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API de reconnaissance des émotions !"}

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)
        emotion = EMOTIONS[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return JSONResponse(content={"emotion": emotion, "confidence": confidence})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
