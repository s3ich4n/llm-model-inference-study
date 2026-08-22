"""테스트가 공유하는 픽스처.

컨테이너의 settings와 openai_client 프로바이더를 갈아끼우면 `.env`도
네트워크도 없이 구성 요소 전체를 조립할 수 있다. 실제 API를 쓰는 테스트는
`@pytest.mark.integration`을 달고 `real_container`를 받는다.
"""

import hashlib
from types import SimpleNamespace

import pytest
from dependency_injector import providers

from config import load_mock_settings
from containers import Container

EMBEDDING_DIM = 16


def _deterministic_embedding(
    text: str,
) -> list[float]:
    """같은 글자에는 늘 같은 벡터를 준다.

    무작위로 만들면 검색 순위가 실행마다 달라져 테스트가 흔들린다. 내용을
    해시해서 벡터를 만들면 비슷한 글자끼리 가까워지지는 않지만, 적어도
    '같은 문서가 같은 자리에 온다'는 것은 확인할 수 있다.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i % len(digest)] / 255.0 for i in range(EMBEDDING_DIM)]


class FakeOpenAI:
    """OpenAI 클라이언트 중 이 프로젝트가 실제로 쓰는 두 갈래만 흉내낸다.

    모양은 실제 SDK를 보고 맞췄다.

    - `create`의 인자는 전부 KEYWORD_ONLY다. 우리 코드가 실수로 위치 인자를
      넘기면 진짜 클라이언트는 TypeError를 내므로 가짜도 그래야 한다.
    - 임베딩 응답의 각 항목에는 `embedding` 말고 `index`도 있다.
    - `message.content`는 `Optional[str]`이라 None이 올 수 있다.
      `completion_text=None`으로 그 경우를 재현할 수 있다.
    """

    def __init__(
        self,
        completion_text: str | None = "가짜 응답",
    ):
        self.completion_text = completion_text
        self.embedding_calls: list[list[str]] = []
        self.completion_calls: list[dict] = []
        self.embeddings = SimpleNamespace(create=self._create_embeddings)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion),
        )

    def _create_embeddings(
        self,
        *,
        input: list[str],
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ):
        self.embedding_calls.append(list(input))
        return SimpleNamespace(
            object="list",
            data=[
                SimpleNamespace(
                    embedding=_deterministic_embedding(text),
                    index=index,
                    object="embedding",
                )
                for index, text in enumerate(input)
            ],
            model=model,
            usage=SimpleNamespace(prompt_tokens=0, total_tokens=0),
        )

    def _create_completion(
        self,
        *,
        model: str,
        messages: list[dict],
        max_completion_tokens: int,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ):
        kwargs = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        self.completion_calls.append(kwargs)
        return SimpleNamespace(
            id="chatcmpl-fake",
            object="chat.completion",
            created=0,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    index=0,
                    message=SimpleNamespace(
                        content=self.completion_text,
                        role="assistant",
                    ),
                ),
            ],
            model=model,
            usage=SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )


@pytest.fixture
def fake_openai():
    return FakeOpenAI()


@pytest.fixture
def container(
    fake_openai,
):
    """가짜 설정과 가짜 OpenAI 클라이언트가 물린 컨테이너."""
    c = Container()
    c.settings.override(providers.Singleton(load_mock_settings))
    c.openai_client.override(providers.Object(fake_openai))
    yield c
    c.reset_override()


@pytest.fixture
def settings(
    container,
):
    return container.settings()


@pytest.fixture
def rag_system(
    container,
):
    return container.rag_system()


@pytest.fixture
def llm_manager(
    container,
):
    return container.llm_manager()


@pytest.fixture
def planner(
    container,
):
    return container.planner()


@pytest.fixture
def action_executor(
    container,
):
    return container.action_executor()


@pytest.fixture
def agent(
    container,
):
    return container.agent()


@pytest.fixture
def built_rag_system(
    rag_system,
):
    """PDF를 읽지 않고 손으로 채워넣은 작은 벡터 DB."""
    documents = [
        {"content": "5-level paging extends linear addresses to 57 bits.",
         "source": "paging.pdf", "file_path": "/x/paging.pdf", "chunk_id": 0},
        {"content": "Patricia tries compress single-child chains.",
         "source": "tries.pdf", "file_path": "/x/tries.pdf", "chunk_id": 0},
        {"content": "OLAP cubes summarise facts across dimensions.",
         "source": "olap.pdf", "file_path": "/x/olap.pdf", "chunk_id": 1},
    ]
    rag_system.documents = documents
    rag_system.embeddings = [
        _deterministic_embedding(doc["content"]) for doc in documents
    ]
    rag_system.metadata = [
        {"source": doc["source"], "chunk_id": doc["chunk_id"]} for doc in documents
    ]
    return rag_system


@pytest.fixture
def real_container():
    """`.env`의 진짜 키를 읽는 컨테이너. integration 테스트 전용."""
    from pydantic import ValidationError

    c = Container()
    try:
        c.settings()
    except ValidationError as exc:
        pytest.skip(f"설정을 읽지 못해 건너뛴다: {exc}")
    return c
