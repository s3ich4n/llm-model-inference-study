from app.store import ModelMetadata
from app.workers.base import ModelWorker
from app.workers.torch_vision import TorchVisionWorker
from app.workers.transformer import TransformerWorker
from app.workers.triton import TritonWorker


class ModelEngine:
    def __init__(self):
        self.workers = {}  # {model_id: worker}

    def get_worker(self, model_id: str) -> ModelWorker | None:
        return self.workers.get(model_id)

    def create_worker(self, model_metadata: ModelMetadata) -> ModelWorker:
        if model_metadata.id not in self.workers:
            if model_metadata.framework == "transformers":
                self.workers[model_metadata.id] = TransformerWorker(model_metadata)
            elif model_metadata.framework == "torchvision":
                self.workers[model_metadata.id] = TorchVisionWorker(model_metadata)
            elif model_metadata.framework == "triton":
                self.workers[model_metadata.id] = TritonWorker(model_metadata)
            else:
                raise ValueError(f"Unsupported framework: {model_metadata.framework}")
        return self.workers[model_metadata.id]

    def delete_worker(self, model_id: str):
        if model_id in self.workers:
            del self.workers[model_id]
