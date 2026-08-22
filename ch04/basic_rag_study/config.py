"""에이전트 설정.

`BaseSettings`는 인스턴스를 만드는 순간 `.env`와 OS 환경변수를 읽는다.
모듈을 임포트하는 시점에는 아무 일도 일어나지 않으므로, 환경을 실제로
읽는 시점은 컨테이너가 이 클래스를 처음 부르는 구동 시점이 된다.
"""

from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_USER_PROFILE = {
    "expertise_level": "intermediate",
    "background": "technical",
    "preferred_detail_level": "moderate",
}


class Settings(BaseSettings):
    """에이전트 구성 요소들이 공유하는 설정.

    필드 이름이 그대로 환경변수 이름이 된다(대소문자 구분 없음). 즉
    `chunk_size`는 `CHUNK_SIZE`를 읽고, 문자열을 int로 바꾸는 것도
    pydantic이 한다.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env에 모르는 키가 있어도 무시한다
        frozen=True,  # 조립이 끝난 뒤에는 아무도 못 바꾼다
    )

    # 기본값이 없으므로 없으면 ValidationError가 난다
    openai_api_key: str

    llm_model: str = "gpt-4.1-nano"
    embedding_model: str = "text-embedding-3-small"
    vector_db_path: str = "./vector_db"
    knowledge_folder: str = "./knowledge_files"

    # 허용된 이름만 받는다. 오타는 ValidationError로 바로 드러난다.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    default_user_profile: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_USER_PROFILE)
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        # .env에 debug라고 적어도 받아준다
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_chunk(self) -> "Settings":
        # RAGSystem._split_text는 (chunk_size - chunk_overlap)만큼 전진한다.
        # 이 값이 0 이하면 같은 자리를 맴돌며 청크를 무한히 만든다.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap({self.chunk_overlap})은 "
                f"chunk_size({self.chunk_size})보다 작아야 한다"
            )
        return self


def load_settings() -> Settings:
    """`.env`와 OS 환경변수를 읽어 실제 설정을 만든다."""
    return Settings()


def load_mock_settings() -> Settings:
    """환경을 전혀 타지 않는 고정 설정.

    pydantic-settings는 생성자 인자를 환경변수보다 우선하므로, 모든 필드를
    직접 넘기면 `.env`가 있든 없든 항상 같은 값이 나온다. `_env_file=None`은
    `.env` 파일 자체를 읽지 않겠다는 뜻이다.
    """
    return Settings(
        _env_file=None,
        openai_api_key="sk-test-mock-key",
        llm_model="gpt-4.1-nano",
        embedding_model="text-embedding-3-small",
        vector_db_path="./vector_db",
        log_level="INFO",
        knowledge_folder="./knowledge_files",
        chunk_size=1000,
        chunk_overlap=200,
        max_tokens=4096,
        temperature=0.7,
    )
