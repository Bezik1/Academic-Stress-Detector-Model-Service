from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import StressInput
from .crud import get_stress_level_prediction

app = FastAPI()

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/api/model/predict")
def predict_stress(input_data: StressInput):
    try:
        prediction = get_stress_level_prediction(input_data)
        return {
            "status": 200,
            "message": "Prediction evaluated successfully",
            "data": {
                "stress_level": prediction
            }
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")