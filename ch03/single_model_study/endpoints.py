import asyncio

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from containers import Container
from dtos import (
    BatchGenerateRequest,
    BatchGenerateResponse,
    GenerateRequest,
    GenerateResponse,
)
from llm import LLMEngine

router = APIRouter()


@router.post("/generate_stream")
@inject
async def generate_stream(
    request: GenerateRequest,
    llm: LLMEngine = Depends(Provide[Container.llm_engine]),
):
    async def event_generator():
        loop = asyncio.get_event_loop()
        async for token in llm.event_generator(loop, request.prompt):
            # token = 'data: {"token": " a", "sequence_id": "8310f5e1-6f6f-480e-b2f9-c8144a12cc17"}\n\n'
            yield token

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


# process 1 request with only one prompt at a time.
@router.post("/basic_generate", response_model=GenerateResponse)
@inject
async def basic_generate(
    request: GenerateRequest,
    llm: LLMEngine = Depends(Provide[Container.llm_engine]),
):
    generated_text = llm.basic_generate(request.prompt)
    return GenerateResponse(generated_text=generated_text)


# process multiple prompts in a request
@router.post("/generate", response_model=BatchGenerateResponse)
@inject
async def generate(
    request: BatchGenerateRequest,
    llm: LLMEngine = Depends(Provide[Container.llm_engine]),
):
    generated_texts = llm.generate(request.prompts)
    return BatchGenerateResponse(generated_texts=generated_texts)


@router.post("/generate_vllm", response_model=BatchGenerateResponse)
@inject
async def generate_vllm(
    request: BatchGenerateRequest,
    llm: LLMEngine = Depends(Provide[Container.llm_engine]),
):
    """
    Generate text using vLLM for multiple prompts.
    This endpoint uses vLLM's efficient batched inference capabilities.
    """
    generated_texts = llm.generate_vllm(request.prompts)
    return BatchGenerateResponse(generated_texts=generated_texts)
