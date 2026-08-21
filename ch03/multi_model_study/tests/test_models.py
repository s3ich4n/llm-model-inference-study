"""`/predict` 와 `/models` 를 HTTP 경계에서 본다.

모델마다 `input_data` 에 넣어야 하는 것이 완전히 다르다는 게 이 파일의 핵심이다.
문자열, 서버 로컬 파일 경로, 텐서 딕셔너리가 같은 엔드포인트로 들어간다.
"""

import numpy as np
import pytest
from transformers import AutoConfig

from tests.model_ids import (
    ALL,
    DENSENET,
    DENSENET_OUTPUT,
    MOBILENET,
    SENTIMENT,
    SPAM,
)


def _argmax(probabilities: list[float]) -> int:
    return max(range(len(probabilities)), key=probabilities.__getitem__)


@pytest.fixture(scope="session")
def sentiment_id2label() -> dict[int, str]:
    """감성 모델은 응답에 라벨 이름을 안 실어 준다. 모델 설정에서 따로 가져온다."""
    config = AutoConfig.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    return config.id2label


# --------------------------------------------------------------------------
# 카탈로그
# --------------------------------------------------------------------------


def test_catalog_ids_match_config(store):
    """models.json 이 어긋나면 나머지 테스트가 통째로 무의미해지니 먼저 잠근다."""
    for label, model_id in ALL.items():
        assert store.get_model(model_id) is not None, f"{label} 이 카탈로그에 없다"


def test_list_models(client):
    response = client.get("/models")
    assert response.status_code == 200

    data = response.json()
    assert "available_models" in data
    assert "loaded_models" in data
    assert set(ALL.values()) <= set(data["available_models"])


# --------------------------------------------------------------------------
# 텍스트 모델: 문자열 하나면 끝
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_label"),
    [
        pytest.param(
            "This movie was great! I really enjoyed it.", "POSITIVE", id="positive"
        ),
        pytest.param(
            "This movie was terrible. I hated every minute of it.",
            "NEGATIVE",
            id="negative",
        ),
    ],
)
def test_sentiment_model(predict, sentiment_id2label, text, expected_label):
    data = predict(SENTIMENT, text)

    probabilities = data["predictions"][0]
    assert len(probabilities) == 2, "이 모델은 클래스가 둘이다"
    assert sentiment_id2label[_argmax(probabilities)] == expected_label


@pytest.mark.parametrize(
    ("text", "expected_label"),
    [
        pytest.param("Hi, can we meet tomorrow at 2pm?", "LABEL_0", id="ham"),
        # 원본 테스트를 그대로 옮겼다. 여기 기대 라벨이 None 인 건 의도된 공백이다.
        # 이 모델은 SMS Spam Collection 문체가 아니면 잘 못 맞춰서,
        # 이 문장을 실제로는 LABEL_0(정상)으로 분류한다. 원본도 그래서 ham 만 검사한다.
        pytest.param(
            "WIN A FREE IPHONE NOW! CLICK HERE!", None, id="spam-label-unchecked"
        ),
    ],
)
def test_spam_model(predict, text, expected_label):
    data = predict(SPAM, text)

    probabilities = data["predictions"][0]
    assert len(probabilities) == 2
    if expected_label is not None:
        assert f"LABEL_{_argmax(probabilities)}" == expected_label


def test_text_model_takes_a_list_as_a_batch(predict):
    """`input_data: Any` 라서 리스트를 넣으면 배치 추론이 그대로 된다."""
    data = predict(SENTIMENT, ["I love it", "I hate it"])

    predictions = data["predictions"]
    assert len(predictions) == 2, "바깥 리스트가 배치 축이다"
    assert predictions[0][1] > predictions[0][0]
    assert predictions[1][0] > predictions[1][1]


# --------------------------------------------------------------------------
# 이미지 모델
# --------------------------------------------------------------------------


def test_torchvision_model_takes_a_server_side_path(predict, cat_image_path):
    """파일을 올리는 통로가 없다. 서버 프로세스가 읽을 경로 문자열을 보낸다."""
    data = predict(MOBILENET, cat_image_path)

    probabilities = data["predictions"][0]
    assert len(probabilities) == 1000, "ImageNet 1000 클래스"
    assert abs(sum(probabilities) - 1.0) < 1e-3, "softmax 결과의 합이 1이 아니다"


def test_torchvision_model_rejects_a_missing_path(predict):
    data = predict(MOBILENET, "there-is-no-such-file.jpg", expect_status=500)
    assert "No such file" in data["detail"]


@pytest.mark.triton
def test_densenet_model_takes_a_whole_tensor(predict, densenet_payload):
    """Triton 쪽은 전처리를 클라이언트가 끝내서 텐서째로 보낸다."""
    data = predict(DENSENET, densenet_payload)

    assert DENSENET_OUTPUT in data, "응답 키가 predictions 가 아니라 출력 텐서 이름이다"
    logits = np.array(data[DENSENET_OUTPUT])
    assert logits.shape == (1000,), "max_batch_size: 0 이라 배치 축이 없다"
    assert abs(logits.sum() - 1.0) > 1e-3, "여긴 softmax를 안 걸어서 확률이 아니다"


# --------------------------------------------------------------------------
# 라우팅과 캐시
# --------------------------------------------------------------------------


def test_unknown_model_id_returns_404(predict):
    predict("invalid-id", "test input", expect_status=404)


def test_cache_never_holds_more_than_two_models(
    predict, client, empty_app_cache, cat_image_path
):
    predict(SENTIMENT, "This movie was great!")
    predict(SPAM, "Hi, can we meet tomorrow at 2pm?")
    predict(MOBILENET, cat_image_path)  # 여기서 sentiment 가 축출된다

    loaded = client.get("/models").json()["loaded_models"]
    assert len(loaded) <= 2
    assert SENTIMENT not in loaded, "가장 오래된 모델이 빠져야 한다"
    assert set(loaded) == {SPAM, MOBILENET}
