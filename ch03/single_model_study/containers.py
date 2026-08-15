from collections.abc import Iterator

from dependency_injector import containers, providers

from llm import LLMEngine
from logs import logger


def _llm_engine() -> Iterator[LLMEngine]:
    engine = LLMEngine()
    try:
        yield engine
    finally:
        try:
            engine._cleanup()
        except Exception:  # noqa: BLE001 - cleanup must never raise
            logger.exception("Failed to clean up LLM engine")


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=["endpoints"])

    llm_engine = providers.Resource(_llm_engine)


container = Container()
