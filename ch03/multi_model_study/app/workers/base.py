from abc import ABC, abstractmethod
from typing import Any

import torch


class ModelWorker(ABC):
    def __init__(self, model_metadata):
        self.model_metadata = model_metadata
        self.model: torch.nn.Module | None = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> dict[str, Any]:
        pass
