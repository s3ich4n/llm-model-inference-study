from contextlib import asynccontextmanager

from fastapi import FastAPI

from containers import container
from endpoints import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    container.init_resources()
    yield
    container.shutdown_resources()


app = FastAPI(lifespan=lifespan)
app.include_router(router)


# Wiring must happen after all @inject-decorated routes above are defined,
# since container.wire() only binds providers to functions it can already
# see in the module namespace at call time.
container.wire(modules=["endpoints"])
