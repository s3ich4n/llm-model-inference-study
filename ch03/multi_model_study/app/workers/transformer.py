from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.workers.base import ModelWorker


class TransformerWorker(ModelWorker):
    def __init__(self, model_metadata):
        self.tokenizer: AutoTokenizer | None = None
        super().__init__(model_metadata)

    def _load_model(self):
        if self.model is None:  # Only load if not already loaded
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_metadata.name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_metadata.name)

    def predict(self, input_data: Any) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized")
        inputs = self.tokenizer(
            input_data,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        return {"predictions": predictions.tolist()}
