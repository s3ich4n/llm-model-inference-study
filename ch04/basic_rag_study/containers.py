"""에이전트를 조립하는 DI 컨테이너.

구성 요소들은 서로를 직접 만들지 않고 생성자로 받는다. 무엇을 넘길지는
여기서만 정해지므로, 테스트는 `container.settings.override(...)` 한 줄로
설정과 OpenAI 클라이언트를 통째로 갈아끼울 수 있다.
"""

from dependency_injector import containers, providers
from openai import OpenAI

from actions import ActionExecutor
from agent import Agent
from config import load_settings
from llm_manager import LLMManager
from planner import Planner
from rag_system import RAGSystem


class Container(containers.DeclarativeContainer):
    # 환경변수는 컨테이너를 만들 때가 아니라 settings를 처음 꺼낼 때 읽힌다.
    # BaseSettings는 인자 없이 부르면 환경을 읽으므로 Singleton이 그대로 받는다.
    settings = providers.Singleton(load_settings)

    openai_client = providers.Singleton(
        OpenAI,
        api_key=settings.provided.openai_api_key,
    )

    rag_system = providers.Singleton(
        RAGSystem,
        settings=settings,
        client=openai_client,
    )

    llm_manager = providers.Singleton(
        LLMManager,
        settings=settings,
        client=openai_client,
    )

    planner = providers.Singleton(
        Planner,
        llm_manager=llm_manager,
    )

    action_executor = providers.Singleton(
        ActionExecutor,
        rag_system=rag_system,
        llm_manager=llm_manager,
        settings=settings,
    )

    # Agent만 Factory다. 사용자 프로필이 다른 에이전트를 여러 개 만들 수 있어야
    # 하고, 그때도 무거운 구성 요소들은 위의 Singleton을 그대로 공유한다.
    agent = providers.Factory(
        Agent,
        settings=settings,
        rag_system=rag_system,
        llm_manager=llm_manager,
        planner=planner,
        action_executor=action_executor,
    )


container = Container()
