"""테스트가 실제 환경변수 대신 가짜 설정을 쓰도록 컨테이너를 갈아끼운다.

`Settings()`는 .env와 OPENAI_API_KEY를 요구하지만, 여기서 config
프로바이더를 `load_mock_settings()`로 덮어쓰면 키 없이도 구성 요소가 만들어진다.
실제 API를 호출하는 테스트는 integration 마커를 달아 따로 돌린다.
"""

import pytest
from dependency_injector import providers

from config import load_mock_settings
from containers import Container


@pytest.fixture
def container():
    """가짜 설정이 주입된 컨테이너. 네트워크 호출은 하지 못한다."""
    c = Container()
    c.settings.override(providers.Singleton(load_mock_settings))
    yield c
    c.settings.reset_override()


@pytest.fixture
def real_container():
    """.env의 진짜 키를 읽는 컨테이너. integration 테스트 전용."""
    c = Container()
    try:
        c.settings()
    except ValueError as exc:
        pytest.skip(str(exc))
    return c


@pytest.fixture
def mock_settings():
    return load_mock_settings()
