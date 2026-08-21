# multi model study

A small multi-model serving service: several models, one process, and only
enough room for two of them at a time. The service loads models on demand and
evicts the least recently used one when the cache is full.

Independent of `ch03/single_model_study` - its own `pyproject.toml`, its own
`uv.lock`, its own `.venv`.

## Features

- On-demand model loading
- LRU (Least Recently Used) model caching, capped at 2 resident models
- Text and image models behind one generic `/predict` interface
- Model metadata kept in a config file, not in code
- Framework-specific workers: Transformers, TorchVision, Triton

## Project structure

```
.
├── app/
│   ├── server.py      # FastAPI server and endpoints
│   ├── store.py       # Model metadata management
│   ├── manager.py     # Model caching and lifecycle (the LRU part)
│   ├── engine.py      # Model worker factory
│   └── workers/       # Abstract worker + framework-specific implementations
│       ├── base.py         # ModelWorker (ABC)
│       ├── transformer.py  # TransformerWorker
│       ├── torch_vision.py # TorchVisionWorker
│       └── triton.py       # TritonWorker
├── config/
│   └── models.json    # Model configurations
├── model_dir/         # Triton model repository (gitignored, see below)
├── tests/
├── compose.yml        # Triton Inference Server
└── pyproject.toml
```

## How to run

```shell
uv sync
uv run uvicorn app.server:app --host 0.0.0.0 --port 8001
```

`python -m app.server` also works and reads `PORT` from the environment.

Note that `ModelStore` is constructed with the relative path
`config/models.json`, so the server has to be started from this directory.

## How to test

```shell
uv run pytest
```

Three of the four models are downloaded from Hugging Face / torchvision on
first use, so the first run takes a while.

Tests that need something outside this process carry a marker, and are skipped
automatically when that thing is missing. Select them explicitly to run a
subset:

| Marker | Needs | Run without it |
| --- | --- | --- |
| `triton` | a Triton server on `:8009` | `uv run pytest -m "not triton"` |
| `gpu` | a CUDA device | `uv run pytest -m "not gpu"` |

```shell
uv run pytest -m "not triton"        # 18 of 24, no container required
uv run pytest -m "gpu"               # only the device / VRAM lifecycle tests
uv run pytest tests/test_models.py   # only the HTTP surface
```

The layout:

```
tests/
├── conftest.py             # fixtures + marker-based skipping
├── model_ids.py            # the four model ids, shared by every test
├── test_models.py          # /predict and /models over HTTP
├── test_triton_densenet.py # Triton's own API, without going through the app
└── test_gpu_lifecycle.py   # device placement and VRAM reclaimed on eviction
```

## Triton Inference Server

The `densenet_onnx` model is served by Triton rather than loaded in-process.
`model_dir/` is the model repository Triton reads, and it is gitignored - the
ONNX weights are 33 MB, so fetch them instead of tracking them:

```shell
git clone -b r25.05 https://github.com/triton-inference-server/server.git /tmp/triton-server
(cd /tmp/triton-server/docs/examples && ./fetch_models.sh)
mkdir -p model_dir
cp -r /tmp/triton-server/docs/examples/model_repository/densenet_onnx model_dir/
```

Then start the server. `compose.yml` runs it in explicit model-control mode,
which is what lets the app load and unload the model on demand:

```shell
docker compose up -d
```

The image is `nvcr.io/nvidia/tritonserver:24.12-py3` and is large (tens of GB),
so the first pull takes a long time.

Once it is up:

```shell
http :8009/v2/health/ready
uv run pytest tests/test_triton_densenet.py
```

## API usage

Examples use [HTTPie](https://httpie.io/). `:8001` is shorthand for
`http://localhost:8001`, and `key=value` becomes a JSON string field while
`key:=value` passes raw JSON through. Every request and response below was
captured from a real run.

```shell
uv tool install httpie      # or: brew install httpie / pipx install httpie
```

The service has two endpoints.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/models` | the catalog, plus what is resident right now |
| `POST` | `/predict` | inference, routed by `model_id` |

### `GET /models`

```shell
http :8001/models
```

```http
HTTP/1.1 200 OK
content-type: application/json
```

```json
{
  "available_models": {
    "550e8400-e29b-41d4-a716-446655440000": {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "distilbert-base-uncased-finetuned-sst-2-english",
      "type": "text",
      "framework": "transformers",
      "version": "1.0.0",
      "description": "Sentiment analysis model"
    },
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8": { ... },
    "7c9e6679-7425-40de-944b-e07fc1f90ae7": { ... },
    "8ba7b810-9dad-11d1-80b4-00c04fd430c9": { ... }
  },
  "loaded_models": {}
}
```

The three elided entries carry the same six fields:

| `id` prefix | `name` | `framework` | `type` |
| --- | --- | --- | --- |
| `550e8400…` | `distilbert-base-uncased-finetuned-sst-2-english` | transformers | text |
| `6ba7b810…` | `mrm8488/bert-tiny-finetuned-sms-spam-detection` | transformers | text |
| `7c9e6679…` | `pytorch/vision:mobilenet_v2` | torchvision | image |
| `8ba7b810…` | `densenet_onnx` | triton | image |

`available_models` is everything in `config/models.json` and never changes.
`loaded_models` is what the LRU cache is holding, and it never grows past two
entries. Fire a few `/predict` calls and read it again to watch models being
evicted:

```json
{
  "loaded_models": {
    "7c9e6679-7425-40de-944b-e07fc1f90ae7": "pytorch/vision:mobilenet_v2",
    "8ba7b810-9dad-11d1-80b4-00c04fd430c9": "densenet_onnx"
  }
}
```

### `POST /predict`

The request schema is only two fields, and the second one is `Any`:

```python
class PredictionRequest(BaseModel):
    model_id: str
    input_data: Any
```

So what belongs in `input_data` depends entirely on which `model_id` you pick.
Four models, four shapes:

| `model_id` | `input_data` | Response key | Values |
| --- | --- | --- | --- |
| `550e8400…` sentiment | a string, or a list of strings | `predictions` | probabilities |
| `6ba7b810…` spam | a string, or a list of strings | `predictions` | probabilities |
| `7c9e6679…` mobilenet | a **server-side file path** | `predictions` | probabilities |
| `8ba7b810…` densenet | `{tensor_name: {shape, data}}` | **`fc6_1`** | **logits** |

#### Sentiment analysis (transformers)

```shell
http POST :8001/predict \
  model_id=550e8400-e29b-41d4-a716-446655440000 \
  input_data="This movie was great!"
```

```json
{
  "predictions": [
    [0.0001321966847172007, 0.9998677968978882]
  ]
}
```

The outer list is the batch, the inner one is the classes. Labels are not in
the response; read them off the model config
(`{0: "NEGATIVE", 1: "POSITIVE"}`), so this one is `POSITIVE` at 99.99%.

#### Spam detection (transformers)

```shell
http POST :8001/predict \
  model_id=6ba7b810-9dad-11d1-80b4-00c04fd430c8 \
  input_data="Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)"
```

```json
{
  "predictions": [
    [0.10327281057834625, 0.8967271447181702]
  ]
}
```

This model's config carries no real label names (`LABEL_0` / `LABEL_1`), so
index 1 winning means spam. It was fine-tuned on the SMS Spam Collection and
only recognises that register: a short shout like `"WIN A FREE IPHONE NOW!"`
comes back `[0.932, 0.068]`, i.e. *not* spam.

#### Batching, for free

`input_data` is `Any` and the tokenizer is called with `padding=True`, so a
list of strings goes through as one batch. Use `:=` to send raw JSON:

```shell
http POST :8001/predict \
  model_id=550e8400-e29b-41d4-a716-446655440000 \
  input_data:='["I love it", "I hate it", "meh"]'
```

```json
{
  "predictions": [
    [0.00012001487630186602, 0.9998799562454224],
    [0.9996398687362671, 0.0003601564676500857],
    [0.020960895344614983, 0.9790391325950623]
  ]
}
```

Only the two text models support this. Both image workers handle exactly one
image per request.

#### Image classification (torchvision)

There is no upload endpoint. `input_data` is a path the *server process* opens
with `Image.open()`, so a URL or a base64 data URI will not work, and the path
is resolved relative to wherever the server was started:

```shell
http POST :8001/predict \
  model_id=7c9e6679-7425-40de-944b-e07fc1f90ae7 \
  input_data=tests/images/cat1.jpg
```

```json
{
  "predictions": [
    [0.0005560, 0.0005305, ...]
  ]
}
```

That inner list is 1000 floats, one per ImageNet class, summing to 1.

Resolve the top index against `MobileNet_V2_Weights.DEFAULT.meta["categories"]`.
For `cat1.jpg` the top five are `tabby`, `Egyptian cat`, `tiger cat`,
`Persian cat`, `plastic bag` — all under 5%, because the subject is a tuxedo
cat and ImageNet has no class for one.

#### Image classification (Triton)

This one takes neither a path nor an upload: the caller does the preprocessing
and sends the whole tensor. `data_0` is the input name declared in
`model_dir/densenet_onnx/config.pbtxt`, and because that file sets
`max_batch_size: 0` with `dims: [3, 224, 224]`, the array must **not** carry a
batch dimension.

That is 150,528 floats, roughly 2.9 MiB of JSON, so build the body into a file
and let HTTPie read it from stdin:

```shell
uv run python - <<'PY'
import json
import numpy as np
from PIL import Image

arr = np.array(Image.open("tests/images/cat1.jpg").resize((224, 224))).astype(np.float32) / 255.0
arr = arr.transpose(2, 0, 1).astype(np.float32)   # HWC -> CHW
json.dump(
    {
        "model_id": "8ba7b810-9dad-11d1-80b4-00c04fd430c9",
        "input_data": {"data_0": {"shape": list(arr.shape), "data": arr.tolist()}},
    },
    open("/tmp/densenet_payload.json", "w"),
)
PY

http POST :8001/predict < /tmp/densenet_payload.json
```

```json
{
  "fc6_1": [-1.6152821779251099, -0.3497679829597473, ...]
}
```

Two things differ from the other three models. The key is the output tensor
name `fc6_1` rather than `predictions`, and the values are raw logits:
`TransformerWorker` and `TorchVisionWorker` both apply `softmax` before
returning, `TritonWorker` hands back whatever the model produced. Map the top
indices onto `model_dir/densenet_onnx/densenet_labels.txt` to read it — for
`cat1.jpg` that is `EGYPTIAN CAT` (11.56), `TIGER CAT` (9.59), `LYNX` (9.52).

### Error responses

```shell
http POST :8001/predict model_id=nope input_data=hello
```

```json
HTTP/1.1 404 Not Found

{"detail": "Model nope not found"}
```

Anything a worker raises during `predict()` is flattened into a 500 with the
exception text, which is how a bad path surfaces:

```shell
http POST :8001/predict \
  model_id=7c9e6679-7425-40de-944b-e07fc1f90ae7 input_data=nope.jpg
```

```json
HTTP/1.1 500 Internal Server Error

{"detail": "[Errno 2] No such file or directory: 'nope.jpg'"}
```

A malformed body is caught by pydantic before the handler runs:

```shell
http POST :8001/predict input_data=hello
```

```json
HTTP/1.1 422 Unprocessable Entity

{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "model_id"],
      "msg": "Field required",
      "input": {"input_data": "hello"}
    }
  ]
}
```

### Triton's own API

`TritonWorker` does not implement loading and unloading, it delegates to
Triton's model repository API. The same calls by hand:

```shell
http :8009/v2/health/ready                                   # 200, empty body
http POST :8009/v2/repository/models/densenet_onnx/load      # 200
http POST :8009/v2/repository/models/densenet_onnx/unload    # 200
http POST :8009/v2/repository/index
```

```json
[
  {"name": "densenet_onnx", "version": "1", "state": "READY"}
]
```

After an unload — which the worker triggers from `__del__` when the LRU cache
evicts it — the same call reports:

```json
[
  {"name": "densenet_onnx", "version": "1", "state": "UNAVAILABLE", "reason": "unloaded"}
]
```

Unloading is asynchronous, so a `state` of `UNLOADING` in between is normal.

One caveat if you drive these by hand: nothing keeps the app's LRU cache and
Triton's repository in sync. `_load_model()` runs once, when the worker object
is constructed, so unloading behind the app's back leaves a cached worker
pointing at a model Triton no longer has:

```shell
http POST :8009/v2/repository/models/densenet_onnx/unload
http :8001/models      # loaded_models still lists densenet_onnx
http POST :8001/predict < /tmp/densenet_payload.json
```

```json
HTTP/1.1 500 Internal Server Error

{"detail": "[404] Request for unknown model: 'densenet_onnx' has no available versions"}
```

Requesting two other models evicts that stale worker, and the next call builds
a fresh one that loads the model again.

## Architecture

Five components, each with one job:

1. **Server** (`server.py`): HTTP endpoints
2. **Store** (`store.py`): model metadata read from `config/models.json`
3. **Manager** (`manager.py`): the LRU cache and the load/evict decisions
4. **Engine** (`engine.py`): factory that picks a worker class by framework
5. **Worker** (`workers/`): the actual inference
   - `base.py` / `ModelWorker`: abstract base class defining the interface
   - `transformer.py` / `TransformerWorker`: transformer-based models
   - `torch_vision.py` / `TorchVisionWorker`: torchvision models
   - `triton.py` / `TritonWorker`: models served by a Triton server over HTTP

A request comes in, the manager either finds the worker in its cache and moves
it to the front, or evicts the oldest one and asks the engine to build a new
worker for it.
