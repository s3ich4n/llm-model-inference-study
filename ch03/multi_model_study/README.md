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
│   └── worker.py      # Abstract worker + framework-specific implementations
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
uv run pytest -s
```

Three of the four models are downloaded from Hugging Face / torchvision on
first use, so the first run takes a while.

`tests/test_triton_densenet.py` and
`tests/test_models.py::TestModelServing::test_image2_triton_model` need a
running Triton server (see below). Without one, skip them:

```shell
uv run pytest -s tests/test_models.py \
  --deselect tests/test_models.py::TestModelServing::test_image2_triton_model
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
curl http://localhost:8009/v2/health/ready
uv run pytest -s tests/test_triton_densenet.py
```

## API usage

### List available models

```shell
curl http://localhost:8001/models
```

The response has two parts: `available_models` is everything in
`config/models.json`, `loaded_models` is what is currently resident. The second
list never grows past two entries.

### Make predictions

Sentiment analysis:

```shell
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "550e8400-e29b-41d4-a716-446655440000", "input_data": "This movie was great!"}'
```

Spam detection:

```shell
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8", "input_data": "Win a free iPhone now!"}'
```

Image classification (the path is read by the server process, not uploaded):

```shell
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "input_data": "tests/images/cat1.jpg"}'
```

## Architecture

Five components, each with one job:

1. **Server** (`server.py`): HTTP endpoints
2. **Store** (`store.py`): model metadata read from `config/models.json`
3. **Manager** (`manager.py`): the LRU cache and the load/evict decisions
4. **Engine** (`engine.py`): factory that picks a worker class by framework
5. **Worker** (`worker.py`): the actual inference
   - `ModelWorker`: abstract base class defining the interface
   - `TransformerWorker`: transformer-based models
   - `TorchVisionWorker`: torchvision models
   - `TritonWorker`: models served by a Triton server over HTTP

A request comes in, the manager either finds the worker in its cache and moves
it to the front, or evicts the oldest one and asks the engine to build a new
worker for it.
