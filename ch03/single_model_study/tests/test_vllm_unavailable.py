import pytest
from fastapi.testclient import TestClient

from containers import container
from llm import VLLMUnavailableError
from main import app


class _StubEngineNoGPU:
    def generate_vllm(self, prompts):
        raise VLLMUnavailableError("vLLM is not available: no GPU detected")


@pytest.fixture
def client_without_gpu():
    # Overrides the DI-provided LLMEngine so this test never loads a real
    # model; exercises the exception-handler wiring, not the model itself.
    container.llm_engine.override(_StubEngineNoGPU())
    try:
        yield TestClient(app)
    finally:
        container.llm_engine.reset_override()


def test_generate_vllm_returns_503_without_gpu(client_without_gpu):
    response = client_without_gpu.post("/generate_vllm", json={"prompts": ["hi"]})

    assert response.status_code == 503
    assert response.json() == {"detail": "vLLM is not available: no GPU detected"}
