"""Triton 서버를 앱을 거치지 않고 직접 두드린다.

`TritonWorker` 가 하는 일(관리 API로 load/unload, tritonclient로 infer)을
그대로 손으로 해 보는 대조군이다. 전부 `@pytest.mark.triton` 이라
서버가 안 떠 있으면 통째로 건너뛴다.
"""

import numpy as np
import pytest
import tritonclient.http as httpclient
from PIL import Image

from tests.model_ids import DENSENET_INPUT, DENSENET_OUTPUT

pytestmark = pytest.mark.triton


def test_model_loading(triton):
    """explicit 모드라 관리 API로 밀어 올려야 올라온다."""
    assert triton.load().status_code == 200
    assert triton.wait_for_state("READY") == "READY"


def test_model_unloading(triton):
    triton.load()
    assert triton.unload().status_code == 200
    # unload 는 비동기라 UNLOADING 을 한 번 거쳐 간다
    assert triton.wait_for_state("UNAVAILABLE") == "UNAVAILABLE"


def test_model_inference(triton, cat_image_path):
    triton.load()

    image = Image.open(cat_image_path).resize((224, 224))
    array = np.array(image).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    # config.pbtxt 가 max_batch_size: 0 이라 배치 축은 붙이지 않는다

    input_tensor = httpclient.InferInput(DENSENET_INPUT, array.shape, "FP32")
    input_tensor.set_data_from_numpy(array)

    response = triton.client.infer(
        model_name=triton.model,
        inputs=[input_tensor],
        outputs=[httpclient.InferRequestedOutput(DENSENET_OUTPUT)],
    )

    logits = response.as_numpy(DENSENET_OUTPUT)
    assert logits is not None
    assert logits.shape == (1000,)
    assert isinstance(np.argmax(logits), np.integer)


def test_inference_rejects_a_batch_dimension(triton, cat_image_path):
    """`max_batch_size: 0` 이라 (1, 3, 224, 224) 를 보내면 서버가 거절한다."""
    triton.load()

    image = Image.open(cat_image_path).resize((224, 224))
    array = np.array(image).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    input_tensor = httpclient.InferInput(DENSENET_INPUT, array.shape, "FP32")
    input_tensor.set_data_from_numpy(array)

    with pytest.raises(Exception, match="(?i)shape|dimension|unexpected"):
        triton.client.infer(
            model_name=triton.model,
            inputs=[input_tensor],
            outputs=[httpclient.InferRequestedOutput(DENSENET_OUTPUT)],
        )
