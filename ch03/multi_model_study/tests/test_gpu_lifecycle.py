"""GPU 배치와 제거 시 VRAM 회수를 검증한다.

두 가지를 나눠서 본다.

1. 워커가 실제로 GPU에 올라가는가 (그리고 입력 텐서도 같은 장치로 따라가는가)
2. LRU에서 제거된 모델의 VRAM이 **다음 모델을 로드하기 전에** 회수되는가

2번이 핵심이다. 캐시 상한을 지켰는데도 OOM이 나는 전형적인 원인이,
제거된 워커가 지역 변수에 붙들린 채 다음 모델이 로드되면서 두 모델이
잠깐 공존하는 상황이기 때문이다.
"""

import gc
import json

import pytest
import torch

from app.workers.torch_vision import TorchVisionWorker
from app.workers.transformer import TransformerWorker
from tests.model_ids import DENSENET, MOBILENET, SENTIMENT, SPAM


def _mib(n: int) -> str:
    return f"{n / 2**20:.1f}MiB"


# --------------------------------------------------------------------------
# 1. 워커가 GPU에 올라가는가
# --------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("worker_class", "model_id"),
    [
        pytest.param(TransformerWorker, SENTIMENT, id="transformer"),
        pytest.param(TorchVisionWorker, MOBILENET, id="torchvision"),
    ],
)
def test_worker_loads_onto_cuda(store, worker_class, model_id):
    worker = worker_class(store.get_model(model_id))
    device = next(worker.model.parameters()).device
    assert device.type == "cuda", f"모델이 {device}에 있다"


@pytest.mark.gpu
def test_transformer_predict_survives_gpu_and_returns_plain_json(store):
    """모델만 GPU로 옮기고 입력을 안 옮기면 여기서 device mismatch가 난다."""
    worker = TransformerWorker(store.get_model(SENTIMENT))
    result = worker.predict("This movie was great!")

    json.dumps(result)  # CUDA 텐서가 섞여 있으면 직렬화가 깨진다
    probabilities = result["predictions"][0]
    assert len(probabilities) == 2
    assert probabilities[1] > probabilities[0], "긍정 문장인데 부정 확률이 더 높다"


@pytest.mark.gpu
def test_torchvision_predict_survives_gpu_and_returns_plain_json(store, cat_image_path):
    """전처리 텐서도 모델과 같은 장치로 따라가야 한다."""
    worker = TorchVisionWorker(store.get_model(MOBILENET))
    result = worker.predict(cat_image_path)

    json.dumps(result)
    probabilities = result["predictions"][0]
    assert len(probabilities) == 1000
    assert abs(sum(probabilities) - 1.0) < 1e-3, "softmax 결과의 합이 1이 아니다"


# --------------------------------------------------------------------------
# 2. 제거된 모델의 VRAM이 다음 로드 전에 회수되는가
# --------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("measure", "what"),
    [
        pytest.param(
            torch.cuda.memory_allocated,
            "제거된 모델의 VRAM이 새 모델 로드 전에 회수되지 않았다",
            id="allocated",
        ),
        pytest.param(
            torch.cuda.memory_reserved,
            "allocator가 제거된 모델의 블록을 계속 쥐고 있다",
            id="reserved",
        ),
    ],
)
def test_eviction_frees_vram_before_the_next_model_loads(
    manager, snapshot_at_next_load, measure, what
):
    """참조를 끊는 것과 VRAM을 실제로 돌려주는 것은 다른 문제다.

    참조만 끊으면 `memory_allocated()` 는 줄지만 PyTorch caching allocator가
    블록을 쥐고 있어서 `memory_reserved()` 와 nvidia-smi 수치는 그대로다.
    그래서 두 지표를 따로 본다.
    """
    manager.get_model_worker(SENTIMENT)
    manager.get_model_worker(SPAM)

    gc.collect()
    baseline = measure()
    assert baseline > 0, "사전 조건: 모델이 VRAM을 쓰고 있어야 한다"

    recorded = snapshot_at_next_load(measure)
    manager.get_model_worker(MOBILENET)  # SENTIMENT 제거 후 MOBILENET 로드

    assert "value" in recorded, "create_worker가 불리지 않았다"
    assert recorded["value"] < baseline, (
        f"{what}. 제거 전 {_mib(baseline)} → 새 모델 로드 직전 {_mib(recorded['value'])}"
    )


@pytest.mark.gpu
def test_vram_does_not_grow_across_repeated_cycles(manager):
    """같은 모델들을 두 바퀴 돌려도 VRAM이 계단식으로 쌓이지 않아야 한다."""
    order = [SENTIMENT, SPAM, MOBILENET, SENTIMENT, SPAM, MOBILENET]

    marks = []
    for model_id in order:
        manager.get_model_worker(model_id)
        assert len(manager.model_cache) <= 2
        gc.collect()
        marks.append(torch.cuda.memory_allocated())

    first_cycle = max(marks[:3])
    second_cycle = max(marks[3:])
    assert second_cycle <= first_cycle * 1.1, (
        "같은 모델을 다시 돌렸는데 VRAM 점유가 늘었다. 제거된 모델이 남아 있다. "
        f"1회차 {_mib(first_cycle)} → 2회차 {_mib(second_cycle)}"
    )


@pytest.mark.triton
def test_triton_unload_fires_at_eviction_not_at_function_exit(
    manager, triton, snapshot_at_next_load
):
    """제거 즉시 원격 unload가 나가야 한다.

    `__del__` 에 기대고 있어서, 제거된 워커를 지역 변수가 붙들고 있으면
    unload 가 `get_model_worker()` 가 끝날 때까지 밀린다.
    """
    manager.get_model_worker(DENSENET)
    manager.get_model_worker(SENTIMENT)
    assert triton.state() == "READY", "사전 조건: densenet이 올라가 있어야 한다"

    recorded = snapshot_at_next_load(triton.state)
    manager.get_model_worker(SPAM)  # DENSENET 제거

    # UNLOADING 이든 UNAVAILABLE 이든 unload 가 이미 나갔다는 뜻이다.
    # 패치 전에는 이 시점에 READY 로 남아 있었다.
    assert recorded.get("value") != "READY", (
        f"Triton이 아직 모델을 들고 있다. 관측된 상태: {recorded.get('value')}"
    )
