import os
import argparse
import logging
from typing import List, Dict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class PredictionRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, float]]

class ModelDeployer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        logger.info(f"Loading model from {self.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        try:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = []
            for probs in probabilities:
                prediction = {f"class_{i}": float(prob) for i, prob in enumerate(probs)}
                predictions.append(prediction)
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

deployer = None

@app.on_event("startup")
async def startup_event():
    global deployer
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise ValueError("MODEL_PATH environment variable is not set")
    deployer = ModelDeployer(model_path)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        predictions = deployer.predict(request.texts)
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    parser = argparse.ArgumentParser(description="Deploy a model with FastAPI")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    
    args = parser.parse_args()
    
    os.environ["MODEL_PATH"] = args.model_path
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
