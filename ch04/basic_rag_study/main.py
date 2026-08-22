"""에이전트 진입점.

`.env`를 읽는 것도, 구성 요소를 조립하는 것도 전부 컨테이너가 한다.
여기서는 컨테이너에서 에이전트를 꺼내 쓰기만 한다.
"""

import logging

from containers import container

logging.basicConfig(level=logging.INFO)


def main():
    # container.agent()를 부르는 순간 config 프로바이더가 처음 평가되고,
    # 그때 .env와 환경변수를 읽는다. 키가 없으면 여기서 멈춘다.
    agent = container.agent()

    print("🔨 Building knowledge base from PDF files...")
    agent.build_knowledge_base()

    agent.interactive_mode()


if __name__ == "__main__":
    main()
