"""에이전트 설정 값과, 그 값을 어디서 읽어올지 정하는 로더들.

이 모듈은 임포트되는 시점에는 아무것도 읽지 않는다. 환경변수를 실제로 읽는
시점은 컨테이너가 `load_config()`를 호출하는 구동 시점이다. 덕분에 테스트는
`.env`나 OS 환경변수와 무관하게 `load_mock_config()`로 갈아끼울 수 있다.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

DEFAULT_USER_PROFILE = {
    "expertise_level": "intermediate",
    "background": "technical",
    "preferred_detail_level": "moderate",
}


@dataclass(frozen=True)
class Config:
    """에이전트 구성 요소들이 공유하는 설정 묶음."""

    openai_api_key: str
    llm_model: str = "gpt-4.1-nano"
    embedding_model: str = "text-embedding-3-small"
    vector_db_path: str = "./vector_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    knowledge_folder: str = "./knowledge_files"
    max_tokens: int = 4096
    temperature: float = 0.7
    default_user_profile: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_USER_PROFILE)
    )


def load_config() -> Config:
    """`.env`와 OS 환경변수를 읽어 실제 설정을 만든다.

    키가 없으면 여기서 바로 멈춘다. 뒤늦게 401을 받고 원인을 되짚는 것보다
    구동 시점에 터지는 편이 낫다.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )

    return Config(
        openai_api_key=api_key,
        llm_model=os.getenv("LLM_MODEL", "gpt-4.1-nano"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        vector_db_path=os.getenv("VECTOR_DB_PATH", "./vector_db"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        knowledge_folder=os.getenv("KNOWLEDGE_FOLDER", "./knowledge_files"),
        max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
    )


def load_mock_config() -> Config:
    """환경변수를 전혀 읽지 않는 고정 설정.

    테스트에서 컨테이너의 config 프로바이더를 이걸로 덮어쓰면 `.env` 없이도
    구성 요소를 만들 수 있다. 키는 형식만 맞춘 가짜라 실제 호출은 못 한다.
    """
    return Config(
        openai_api_key="sk-test-mock-key",
        llm_model="gpt-4.1-nano",
        embedding_model="text-embedding-3-small",
        vector_db_path="./vector_db",
        chunk_size=1000,
        chunk_overlap=200,
        knowledge_folder="./knowledge_files",
        max_tokens=4096,
        temperature=0.7,
    )
