"""에이전트 진입점.

환경을 읽는 것도, 구성 요소를 조립하는 것도 전부 컨테이너가 한다.
여기서는 컨테이너에서 에이전트를 꺼내 쓰기만 한다.
"""

from pydantic import ValidationError

from containers import container


def _report_settings_error(
    error: ValidationError,
) -> None:
    """어느 설정이 왜 틀렸는지 사람이 읽을 형태로 알려준다."""
    print("❌ 설정을 읽지 못했습니다. 아래 항목을 확인하세요.\n")
    for err in error.errors():
        field = ".".join(str(part) for part in err["loc"]) or "(전체)"
        print(f"   {field.upper()}: {err['msg']}")
    print("\n💡 env_example.txt의 환경변수들을 mise.local.toml 형식으로 구성해주세요.")


def main():
    # container.agent()를 부르는 순간 settings 프로바이더가 처음 평가되고,
    # 그때 .env와 환경변수를 읽는다. 값이 잘못됐으면 여기서 멈춘다.
    try:
        container.init_resources()
        agent = container.agent()
    except ValidationError as e:
        _report_settings_error(e)
        raise SystemExit(1) from None

    print("🔨 Building knowledge base from PDF files...")
    agent.build_knowledge_base()
    agent.interactive_mode()


if __name__ == "__main__":
    main()
