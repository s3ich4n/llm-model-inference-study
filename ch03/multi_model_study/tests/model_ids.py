"""테스트가 참조하는 모델 ID.

`config/models.json` 과 짝을 맞춘다. 여기 값이 카탈로그와 어긋나면
`test_models.py::test_catalog_ids_match_config` 가 먼저 깨진다.
"""

SENTIMENT = "550e8400-e29b-41d4-a716-446655440000"
SPAM = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
MOBILENET = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
DENSENET = "8ba7b810-9dad-11d1-80b4-00c04fd430c9"

ALL = {
    "sentiment": SENTIMENT,
    "spam": SPAM,
    "mobilenet": MOBILENET,
    "densenet": DENSENET,
}

DENSENET_MODEL_NAME = "densenet_onnx"
DENSENET_INPUT = "data_0"
DENSENET_OUTPUT = "fc6_1"
