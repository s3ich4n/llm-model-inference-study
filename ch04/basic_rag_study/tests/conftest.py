"""테스트가 공유하는 픽스처.

컨테이너의 settings와 openai_client 프로바이더를 갈아끼우면 `.env`도
네트워크도 없이 구성 요소 전체를 조립할 수 있다. 실제 API를 쓰는 테스트는
`@pytest.mark.integration`을 달고 `real_container`를 받는다.
"""

import hashlib
from types import SimpleNamespace

import pytest
from dependency_injector import providers
from pydantic_settings import SettingsConfigDict

from config import Settings
from containers import Container

EMBEDDING_DIM = 16


class MockSettings(Settings):
    """환경을 아예 보지 않는 Settings.

    `_env_file=None`만으로는 부족하다. `.env` 읽기는 막아도 OS 환경변수는
    그대로 들어오기 때문이다. 그래서 설정 출처를 생성자 인자 하나로 줄인다.
    넘기지 않은 필드는 `Settings`에 선언된 기본값을 그대로 쓰므로, 필드가
    늘어나도 여기를 따라 고칠 필요가 없다.

    검사 규칙은 그대로 살아 있다. 테스트가 통과시키는 설정은 실제로도
    통과할 수 있는 설정이어야 한다.
    """

    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        frozen=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """TC 구동 시 별도 값이 들어오지 않도록 방어하기 위해 오버라이드

        - mise 를 쓰는 환경
        - CI환경 등
        """
        return (init_settings,)


def load_mock_settings(**overrides) -> Settings:
    """가짜 설정을 만든다. 바꾸고 싶은 필드만 키워드로 넘기면 된다."""
    return MockSettings(
        openai_api_key="sk-test-mock-key",
        **overrides,
    )


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


@pytest.fixture(autouse=True)
def _isolate_settings_environment(monkeypatch, request):
    """셸에 export된 설정 값이 테스트에 새어들지 않게 막는다.

    개발자 환경에 `CHUNK_SIZE=7` 같은 게 남아 있으면 기본값을 확인하는
    테스트가 엉뚱하게 깨진다. 다만 integration 테스트는 실제 mise/.env
    설정과 API 키를 사용해야 하므로 환경을 그대로 둔다.
    """
    if request.node.get_closest_marker("integration") is not None:
        return

    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


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
