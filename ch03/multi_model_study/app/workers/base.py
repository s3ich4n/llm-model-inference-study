from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

import torch


class ModelWorker(ABC):
    def __init__(self, model_metadata):
        self.model_metadata = model_metadata
        self.model: Optional[torch.nn.Module] = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Dict[str, Any]:
        pass
