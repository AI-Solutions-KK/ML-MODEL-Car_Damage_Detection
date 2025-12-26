# 🚗 Vehicle Damage Detection API

## Overview
Vehicle Damage Detection API is a **production-ready machine learning REST service** built with **FastAPI** and **PyTorch**.

The API accepts a vehicle image and returns an AI-predicted damage category such as **Front Breakage**, **Rear Breakage**, or **No Damage**.

---

## Features
- 🧠 Deep Learning model (ResNet50 – PyTorch)
- ⚡ FastAPI for high‑performance inference
- 📸 Image upload via REST API
- 📊 Swagger UI documentation
- ☁️ Cloud-ready (Azure App Service)
- 🔒 Stateless inference

---

## Project Structure
```
vehicle-damage-detection-api/
├── app/
│   ├── main.py
│   ├── inference.py
│   └── schemas.py
├── models/
│   └── saved_model.pth
├── sample.jpg
├── run_local.py
├── requirements.txt
└── README.md
```

---

## API Usage

### POST /predict
Upload vehicle image and receive damage class.

```bash
curl -X POST http://127.0.0.1:8000/predict   -F "file=@sample.jpg"
```

Response:
```json
{ "prediction": "Front Breakage" }
```

---

## Run Locally
```bash
python run_local.py
```
Swagger UI:
http://127.0.0.1:8000/docs

---

## License
MIT License
