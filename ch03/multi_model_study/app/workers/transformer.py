from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.workers.base import ModelWorker


class TransformerWorker(ModelWorker):
    def __init__(self, model_metadata):
        self.tokenizer: AutoTokenizer | None = None
        # super().__init__() 안에서 _load_model()이 도니 그 전에 준비해 둔다
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model_metadata)

    def _load_model(self):
        if self.model is None:  # Only load if not already loaded
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.model_metadata.name
                )
                .to(self.device)
                .eval()
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
        ).to(self.device)  # 입력도 모델과 같은 장치로
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        return {"predictions": predictions.cpu().tolist()}
