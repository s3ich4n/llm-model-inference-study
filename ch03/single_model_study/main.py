from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from containers import container
from endpoints import router
from llm import VLLMUnavailableError


@asynccontextmanager
async def lifespan(app: FastAPI):
    container.init_resources()
    yield
    container.shutdown_resources()


app = FastAPI(lifespan=lifespan)
app.include_router(router)


@app.exception_handler(VLLMUnavailableError)
async def vllm_unavailable_handler(request: Request, exc: VLLMUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# Wiring must happen after all @inject-decorated routes above are defined,
# since container.wire() only binds providers to functions it can already
# see in the module namespace at call time. Which modules to wire is
# declared on the Container itself (wiring_config); this just triggers it.
container.wire()
