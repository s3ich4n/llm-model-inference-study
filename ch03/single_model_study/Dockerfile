FROM python:3.12-slim

# torch/numpy pull in OpenMP at runtime; slim images don't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

# Install dependencies first so this layer is cached unless the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application code.
COPY containers.py dtos.py endpoints.py logs.py main.py ./
COPY llm ./llm

EXPOSE 8000

# A single worker on purpose: each uvicorn worker would load its own copy of
# the model (ModelExecutor's worker process, and vLLM's engine if a GPU is
# visible) - this image is sized for one model instance per container.
#
# GPU vs CPU is not a build-time choice: llm/model_worker.py and llm/llm.py
# both auto-detect via torch.cuda.is_available(). Run with `docker run
# --gpus all` (nvidia-container-toolkit on the host) to use a GPU, including
# the vLLM-backed /generate_vllm endpoint; run without it for CPU-only, in
# which case /generate_vllm returns 503 and the other three endpoints still
# work normally.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
