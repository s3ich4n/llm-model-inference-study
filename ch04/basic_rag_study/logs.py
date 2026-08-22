"""로깅 설정.

ch03/single_model_study/logs.py와 같은 방식이다. 핸들러와 포매터를 임포트
시점에 한 번 붙여두고, 각 모듈은 여기서 로거를 받아 쓴다. 모듈마다
`logging.basicConfig`를 부르지 않으므로 설정이 한 곳에만 있다.

ch03과 두 군데 다르다.

첫째, ch03은 `logs`라는 이름의 로거 하나를 모두가 공유하는데 그러면
`%(name)s` 자리에 항상 `logs`만 찍힌다. ch04는 모듈이 여섯이고 PDF 적재부터
검색, 계획, 실행까지 흐름을 따라가야 해서 `get_logger(__name__)`으로
모듈별 이름을 살린다.

둘째, 레벨을 `Settings.log_level`에서 받는다. 다만 이 모듈은 `config`를
임포트하지 않는다. 그러면 임포트 시점에 환경을 읽게 되어 지금까지 옮겨놓은
것이 도로 무너진다. 대신 레벨을 나중에 바꿀 수 있게 열어두고, 값을 넣는
일은 컨테이너가 한다.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO

# 핸들러는 하나만 만들어 모든 로거가 공유한다. 모듈마다 새로 만들면
# 같은 줄이 여러 번 찍힌다.
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(LOG_FORMAT))
_handler.setLevel(DEFAULT_LOG_LEVEL)


def get_logger(
    name: str,
) -> logging.Logger:
    """모듈 이름을 단 로거를 돌려준다.

    로거 자체는 DEBUG로 열어두고 걸러내는 일은 핸들러에 맡긴다. 그래야
    이미 만들어진 로거들을 일일이 찾아다니지 않고 `configure_logging()`
    한 번으로 레벨을 바꿀 수 있다.
    """
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    if _handler not in log.handlers:
        log.addHandler(_handler)
    return log


def configure_logging(
    level: str | int,
) -> None:
    """구동 시점에 `Settings.log_level`을 반영한다.

    이 함수를 부르기 전에 만들어진 로거에도 그대로 적용된다. 걸러내는
    주체가 로거가 아니라 공유 핸들러이기 때문이다.
    """
    _handler.setLevel(level)


# ch03처럼 `from logs import logger`로 바로 쓰고 싶을 때를 위한 기본 로거
logger = get_logger(__name__)
