from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from .manager import ModelManager
from .store import ModelStore

app = FastAPI(title="Multi-Model Serving Demo")

# Initialize components
model_store = ModelStore("config/models.json")
model_manager = ModelManager(model_store)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    input_data: Any

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Get model worker
    worker = model_manager.get_model_worker(request.model_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    
    # Make prediction
    try:
        result = worker.predict(request.input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    return {
        "available_models": model_store.list_models(),
        "loaded_models": model_manager.list_loaded_models()
    }
