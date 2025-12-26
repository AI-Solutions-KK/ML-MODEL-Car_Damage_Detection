from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from app.inference import predict_damage

app = FastAPI(title="Vehicle Damage Detection API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prediction = predict_damage(image)
    return {"prediction": prediction}
