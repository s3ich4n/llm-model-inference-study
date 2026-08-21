import gc
from collections import OrderedDict

import torch

from .engine import ModelEngine
from .store import ModelStore
from .workers.base import ModelWorker


class ModelManager:
    def __init__(self, model_store: ModelStore, max_models: int = 2):
        self.model_store = model_store
        self.max_models = max_models
        self.model_cache = (
            OrderedDict()
        )  # OrderedDict to track least recently used, id -> worker
        self.model_engine = ModelEngine()

    def get_model_worker(self, model_id: str) -> ModelWorker | None:
        # Check if model is in cache
        if model_id in self.model_cache:
            # Move to end (most recently used)
            self.model_cache.move_to_end(model_id)
            return self.model_engine.get_worker(model_id)

        # Get model metadata
        model_metadata = self.model_store.get_model(model_id)
        if not model_metadata:
            return None

        # Check if we need to remove least used model
        if len(self.model_cache) >= self.max_models:
            # Remove least recently used model.
            # 참조를 끊는 것과 VRAM을 실제로 돌려주는 것은 다른 문제라,
            # 다음 모델을 로드하기 전에 세 단계를 모두 거쳐야 한다.
            evicted_id, evicted_worker = self.model_cache.popitem(last=False)
            self.model_engine.delete_worker(evicted_id)
            del evicted_worker  # 마지막 참조 제거 (TritonWorker면 여기서 원격 unload)
            gc.collect()  # 순환 참조까지 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # allocator가 쥔 블록을 드라이버에 반환

        # Download model if not already downloaded
        # Skip the downlaod implementation for simplicity
        # if not self.model_store.model_exists(model_id):
        #     self.model_store.download_model(model_id)

        # Create and cache new model worker
        self.model_cache[model_id] = self.model_engine.create_worker(model_metadata)
        return self.model_cache[model_id]

    def list_loaded_models(self) -> dict[str, str]:
        return {
            model_id: worker.model_metadata.name
            for model_id, worker in self.model_cache.items()
        }
