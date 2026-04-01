from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger import get_logger

logger = get_logger("App")

app = FastAPI(title="Fake News Classifier API", description="API to classify news as real or fake and trigger retraining")

# Pydantic model for receiving text prediction queries
class NewsRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "message": "Welcome to Fake News Classifier API",
        "endpoints": {
            "/train": "Trigger model training pipeline",
            "/predict": "Predict whether the news text is real or fake"
        }
    }

@app.post("/train")
def train_route():
    try:
        logger.info("Received request to train the model")
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        return {"message": "Model Training was successful!"}
    except Exception as e:
        logger.error(f"Error occurred during training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_route(request: NewsRequest):
    try:
        logger.info(f"Received prediction request")
        prediction_pipeline = PredictionPipeline()
        # 1 denotes fake and 0 denotes real, assuming the model training is standardized that way
        result = "Fake News" if prediction_pipeline.predict(request.text) == 1 else "Real News"
        return {
            "prediction": result,
            "text": request.text
        }
    except Exception as e:
        logger.error(f"Error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
