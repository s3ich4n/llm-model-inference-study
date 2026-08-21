"""테스트 전역 픽스처.

바깥 환경에 기대는 테스트는 마커로 갈라 둔다.

- `@pytest.mark.gpu`    : CUDA가 없으면 건너뛴다
- `@pytest.mark.triton` : Triton 서버가 안 떠 있으면 건너뛴다

그래서 `pytest -m "not triton"` 한 줄로 Triton 없이 돌릴 수 있다.
"""

import gc
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import requests
import torch
import tritonclient.http as httpclient
from fastapi.testclient import TestClient
from PIL import Image

from app import server
from app.engine import ModelEngine
from app.manager import ModelManager
from app.store import ModelStore
from tests.model_ids import (
    DENSENET,
    DENSENET_INPUT,
    DENSENET_MODEL_NAME,
)

CONFIG_PATH = "config/models.json"
TRITON_URL = "0.0.0.0:8009"
IMAGES_DIR = Path(__file__).parent / "images"


# --------------------------------------------------------------------------
# 마커 기반 건너뛰기
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def triton_is_ready() -> bool:
    """세션당 한 번만 물어본다."""
    try:
        response = requests.get(f"http://{TRITON_URL}/v2/health/ready", timeout=2)
    except requests.RequestException:
        return False
    return response.status_code == 200


def pytest_runtest_setup(item):
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("CUDA를 쓸 수 없는 환경")
    if item.get_closest_marker("triton") and not triton_is_ready():
        pytest.skip(f"Triton 서버({TRITON_URL})가 떠 있지 않음")


# --------------------------------------------------------------------------
# 공통 자원
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def store() -> ModelStore:
    """읽기 전용 카탈로그라 세션 하나로 충분하다."""
    return ModelStore(CONFIG_PATH)


@pytest.fixture(scope="session")
def cat_image_path() -> str:
    path = IMAGES_DIR / "cat1.jpg"
    assert path.exists(), f"테스트 이미지가 없다: {path}"
    return str(path)


@pytest.fixture(scope="session")
def densenet_payload(cat_image_path) -> dict:
    """DenseNet이 기대하는 (3, 224, 224) FP32 텐서를 JSON 실을 수 있는 모양으로.

    `config.pbtxt` 가 `max_batch_size: 0` 이라 배치 축을 붙이지 않는다.
    """
    image = Image.open(cat_image_path).resize((224, 224))
    array = np.array(image).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    return {
        DENSENET_INPUT: {
            "shape": list(array.shape),
            "data": array.tolist(),
            "dtype": "float32",
        }
    }


@pytest.fixture(autouse=True)
def clean_vram():
    """테스트마다 남은 텐서를 치워서 측정값이 서로 새지 않게 한다."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 서비스 계층
# --------------------------------------------------------------------------


def _drain(manager: ModelManager) -> None:
    """캐시를 비우면서 VRAM까지 실제로 돌려준다."""
    while manager.model_cache:
        evicted_id, evicted_worker = manager.model_cache.popitem(last=False)
        manager.model_engine.delete_worker(evicted_id)
        del evicted_worker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def manager(store) -> ModelManager:
    """테스트마다 새 LRU 캐시. 끝나면 비운다."""
    instance = ModelManager(store, max_models=2)
    yield instance
    _drain(instance)


@pytest.fixture
def client() -> TestClient:
    """`app.server` 의 전역 매니저를 그대로 쓰는 HTTP 클라이언트.

    전역이라 테스트 사이에 캐시가 남는다. 시작 상태가 중요한 테스트는
    `empty_app_cache` 를 같이 요청하면 된다.
    """
    return TestClient(server.app)


@pytest.fixture
def empty_app_cache():
    """`/predict` 를 빈 캐시에서 시작하고 싶을 때."""
    _drain(server.model_manager)
    yield
    _drain(server.model_manager)


@pytest.fixture
def predict(client):
    """`POST /predict` 호출 한 번을 감싼다."""

    def _predict(model_id: str, input_data, expect_status: int = 200):
        response = client.post(
            "/predict", json={"model_id": model_id, "input_data": input_data}
        )
        assert response.status_code == expect_status, response.text
        return response.json()

    return _predict


@pytest.fixture
def snapshot_at_next_load(monkeypatch):
    """다음 `create_worker()` 가 불리기 **직전**의 값을 한 번 찍어 둔다.

    축출과 새 모델 로드 사이의 짧은 구간을 들여다보는 용도다.
    반환된 dict의 `value` 키에 측정값이 들어온다.
    """

    def _install(measure):
        recorded: dict = {}
        original = ModelEngine.create_worker

        def spy(self, model_metadata):
            recorded.setdefault("value", measure())
            return original(self, model_metadata)

        monkeypatch.setattr(ModelEngine, "create_worker", spy)
        return recorded

    return _install


# --------------------------------------------------------------------------
# Triton
# --------------------------------------------------------------------------


class TritonHarness:
    """Triton 관리 API를 직접 두드리는 얇은 래퍼."""

    def __init__(self, url: str = TRITON_URL, model: str = DENSENET_MODEL_NAME):
        self.url = url
        self.model = model
        self.client = httpclient.InferenceServerClient(url=url)

    def _repository(self, action: str) -> requests.Response:
        return requests.post(
            f"http://{self.url}/v2/repository/models/{self.model}/{action}", timeout=60
        )

    def load(self) -> requests.Response:
        return self._repository("load")

    def unload(self) -> requests.Response:
        return self._repository("unload")

    def state(self) -> str | None:
        """지금 Triton이 이 모델을 어떤 상태로 들고 있는지."""
        response = requests.post(
            f"http://{self.url}/v2/repository/index", timeout=10
        ).json()
        for entry in response:
            if entry["name"] == self.model:
                return entry.get("state")
        return None

    def wait_for_state(self, expected: str, timeout: float = 30.0) -> str | None:
        """LOADING / UNLOADING 을 거쳐 자리를 잡을 때까지 기다린다."""
        deadline = time.monotonic() + timeout
        observed = self.state()
        while observed != expected and time.monotonic() < deadline:
            time.sleep(0.05)
            observed = self.state()
        return observed


@pytest.fixture
def triton():
    """Triton 하네스. 테스트가 끝나면 모델을 내려 둔다."""
    harness = TritonHarness()
    yield harness
    harness.unload()


@pytest.fixture
def densenet_id() -> str:
    return DENSENET
