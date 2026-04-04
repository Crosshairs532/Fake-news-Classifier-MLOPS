import time
from fastapi import FastAPI, HTTPException, Request, Response, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger import get_logger



load_dotenv()



logger = get_logger("App")

app = FastAPI(title="Fake News Classifier API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

REQUEST_COUNT = Counter(
    "http_requests_total", 
    "Total HTTP Requests", 
    ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", 
    "HTTP request latency in seconds", 
    ["method", "endpoint"]
)
PREDICTION_RESULTS = Counter(
    "news_classification_total", 
    "Count of classifications", 
    ["type"]
)




class NewsRequest(BaseModel):
    text: str

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method, 
        endpoint=request.url.path, 
        http_status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=request.method, 
        endpoint=request.url.path
    ).observe(process_time)
    
    return response

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

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

@app.post("/predict-web")
def predict_web(request: Request, text: str = Form(...)):
    prediction_pipeline = PredictionPipeline()
    prediction = prediction_pipeline.predict(text)
    result = "Fake News" if prediction == 1 else "Real News"
    
    return templates.TemplateResponse("index.html", {"request": request, "prediction": result})
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)