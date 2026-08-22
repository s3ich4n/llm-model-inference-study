### Model Serving in an Agentic World

- 에이전트 모델 서빙이 다른점 소개
    - 핵심 전환점
        - 앞선 챕터들(2~3장)에서는 **모델을 독립적인 예측 엔진으**로 다뤘습니다. **요청 하나에 추론 한 번, 응답 반환**.
        - 이번 절부터는 모델이 에이전트 시스템 안에 내장되면서 이 전제가 깨집니다:
            - **모델이 요청당 한 번이 아니라, 제어 루프(control loop) 안에서 반복 호출됨**
            - 그 루프 안에서 정보를 검색하고, 중간 결과를 추론하고, 도구를 실행하고, 출력을 다듬는 과정을 거쳐야 최종 답변이 나옴
    - 구조적 복잡도
        - 에이전트 하나만으로도 정교한 작업을 수행할 수 있지만, 실무에서는 여러 에이전트가 서로 위에 쌓이거나(build on top of one another), 협업하거나, 기능별로 특화되는 에이전트 네트워크/플랫폼 형태로 조직되는 경우가 많습니다.
        - 이런 계층 구조는 뛰어난 자율성을 주지만, 그만큼 설계·운영상 난제(특히 서빙 관점에서)를 만듭니다.
    - 서빙에 미치는 구체적 영향 : **사용자 상호작용 한 번이 다음을 촉발할 수 있습니다**
        - 다중 LLM 호출
        - 더 긴 컨텍스트 윈도우
        - 검색 연산 (RAG)
        - 메모리 재사용 (CAG)
    - 이 모든 게 토큰 사용량을 늘리고, 체인으로 연결된 호출들에서 지연시간을 증폭시키고, 트래픽 패턴을 더 동적으로 만듭니다.
    - 결과적으로 서빙 = 모델을 효율적으로 실행하는 것을 넘어, **오케스트레이션·메모리 관리·시스템 레벨 조정까지 지원**해야 하는 일이 됩니다.
    - 책 원문 번역
        - 이 섹션에서는 샘플 에이전트를 사용해 LLM을 활용하여 소프트웨어와 서비스가 상호작용하면서 동시에 자율적으로 작동하는 에이전트형 사용자 경험을 만드는 방법을 보여줍니다.
        - 이전 장들에서는 모델을 독립적인 예측 엔진으로 제공하는 데 초점을 맞췄지만, 현대의 LLM 활용에서는 모델이 에이전트 시스템 내부에 점점 더 깊이 통합되고 있습니다. 이러한 시스템에서는 모델이 요청당 한 번만 호출되는 것이 아닙니다. 대신, 정보를 수집하고 중간 결과를 분석하며 도구를 실행하고 출력물을 다듬은 뒤 최종 답변을 내놓는 제어 루프 내에서 반복적으로 호출될 수 있습니다. 이러한 변화는 모델 서비스 시스템에 요구되는 조건을 근본적으로 바꾸고 있습니다.
        - 에이전트는 강력하면서도 복잡합니다. 하나의 에이전트가 복잡한 작업을 수행할 수 있지만, 실제로는 여러 에이전트가 서로 위에 쌓이거나 협력하거나 각기 다른 기능을 전문으로 하는 에이전트 네트워크나 플랫폼으로 구성되는 경우가 많습니다. 이러한 계층적 아키텍처는 뛰어난 자율성을 가능하게 하지만, 특히 모델 서빙 측면에서 설계와 운영에 상당한 도전 과제를 동반합니다.
        - 예를 들어, 하나의 사용자 상호작용이 여러 번의 LLM 호출, 더 긴 컨텍스트 윈도우, 검색 연산(RAG), 또는 메모리 재사용(CAG)을 유발할 수 있습니다. 이러한 행위는 토큰 사용량을 증가시키고, 체인된 호출 간 지연 시간을 늘리며, 더 동적인 트래픽 패턴을 만듭니다. 따라서 서비스는 단순히 모델을 효율적으로 실행하는 것을 넘어서, 오케스트레이션, 메모리 관리, 시스템 수준의 조정을 지원해야 합니다.
        - 에이전트 개발 맥락에서 모델 서빙을 이해하는 데 도움을 주기 위해, 여기서는 기본 개념에 중점을 둡니다. 기본적인 에이전트를 예로 들어, 모델 서빙이 에이전트 자율성의 핵심 메커니즘을 어떻게 지원하는지 보여줍니다. 이 기초를 바탕으로 스스로 더 고급스럽고 다중 에이전트 시스템을 탐구하고 이해할 수 있는 역량이 향상될 것입니다.
    
- 에이전트의 정의
    - **에이전트의 정의** : 다음이 가능한 자율적인 LLM 기반 시스템으로 정의
        1. 상위 수준 목표를 이해
        2. 그 목표를 달성할 방법을 추론
        3. 외부 도구나 데이터 소스를 선택·호출
        4. 중간 결과를 바탕으로 적응·반복
        5. 사람 개입을 최소화(또는 아예 없이) 최종 결과물 산출
        
    - 에이전트 유형 예시
        - 리서치 에이전트: 논문을 읽고, 핵심 발견을 추출해 문헌 리뷰를 종합
        - 코딩 에이전트: 코드를 생성·리뷰·테스트·디버깅하고, 저장소로부터 배포까지 관리
        - 비즈니스 운영 에이전트: 데이터 입력을 자동화하고, 보고서를 생성해 이해관계자에게 배포
        
    - 핵심 차이: **전통적 어시스턴트 vs LLM 에이전트**
    - 규칙 기반 챗봇 같은 전통적 시스템은 개발자가 미리 짜둔 워크플로우나 사용자의 단계별 지시에 의존합니다.
    - 반면 LLM 기반 에이전트는 다음 능력 덕분에 복잡한 워크플로우를 독립적으로 실행할 수 있습니다:
        - 자연어 지시 해석
        - 추론 및 행동 계획
        - 도구 선택 및 사용
        - 환경/사람의 피드백에 적응
        - 컨텍스트와 사용자 선호도 유지
    - 즉 차이의 본질은 자율성입니다 : "무엇을 어떻게 할지"까지 시스템 스스로 판단한다는 점.
    
- (참고) **악분**님: ‘**멀티턴 대화** 동작 원리’를 간결하게 정리 - Blog
    
    !image.png
    
- (참고) **악분**님: AI 모델은 인터넷 검색은 못하는데, 어떻게 인터넷 검색을 할까? : **tool call** - Blog
    - 방법 1: 클라이언트에서 실행하기 - 클라이언트에서 처리하는 방식은 모델의 tool call을 애플리케이션이 직접 실행하는 구조
        
        !image.png
        
    - 방법 2: AI gateway에서 실행하기 - AI gateway에서 처리하는 방식은 gateway가 모델의 검색 tool call을 가로채 실행하는 구조
        
        !image.png
        
    - 방법 3: AI provider에서 실행하기 - AI provider 방식은 모델을 제공하는 서비스가 검색 tool까지 실행하는 구조
        
        !image.png
        
    - AI모델 인터넷검색의 핵심은 tool call을 어디에서 실행하고 결과를 어떻게 다시 전달하는지입니다.
    - tool call 실행하는 곳을 분류하고 장점만 나열하면,
        1. 클라이언트는 실행 흐름을 세밀하게 제어하기 좋고,
        2. AI gateway는 여러 모델에 공통 정책을 적용하기 좋으며,
        3. AI provider 방식은 구현을 줄이는 대신 provider의 지원 범위에 의존합니다.
    
- Knowledge Agent : PDF 파일들을 질의·분석하는 지식 에이전트 코드 구현
    
    !https://devlos.tistory.com/129
    
    https://devlos.tistory.com/129
    
    - **설계 특징**
        - 이식성(portability) 최우선
            - 로컬 모델 호스팅이나 별도 DB 없이, **OpenAI API(LLM 추론 + 임베딩 둘 다)**를 쓰고 **모든 정보는 인메모리에 저장**
        - 지식 베이스 = 로컬 PDF 폴더
            - **knowledge_files/에 PDF를 넣어두면 에이전트가 기동 시 자동으로 처리**합니다.
    - **지원 기능**
        1. 문서 질의(direct query) : 예: "5-level paging이 뭐고 어떻게 동작하나?"
        2. 복잡한 질문을 위한 지능형 플래닝
        3. 요약(summarization)
        4. 문서 간 분석/비교 : 예: "DB 쿼리 최적화와 자료구조 최적화를 상세 비교해줘"
    - **실행** : **python agent.py로 대화형 CLI**처럼 질문을 던지고 답을 받는 방식
        
        
    - 이 절에서는 실제 예를 통해 모델 서빙이 PDF 파일에서 정보를 조회하고 분석하도록 설계된 지식 에이전트를 어떻게 구동할 수 있는지 설명합니다.
    - 이 샘플 에이전트는 의도적으로 단순하지만, 모델 서빙 위에 에이전트를 구축하는 일반적인 아키텍처 패턴을 잘 보여줍니다.
    - 간단하게 하기 위해 이 지식 에이전트를 "Knowledge Agent"라고 부르겠습니다
    - 이식성을 극대화하기 위해 Knowledge Agent는 로컬 모델 호스팅과 데이터베이스를 사용하지 않습니다.
    - 대신, 그것은 **Open에 의존**합니다.
    - LLM 추론과 임베딩을 위한 AI의 API이며, 모든 정보를 메모리에 저장합니다.
    - 소스 코드와 설정 지침은 이 책의 GitHub 저장소 내 KnowledgeAgent 폴더에서 확인할 수 있습니다.
    - 이 장의 나머지 내용을 읽기 전에, 먼저 이 지침을 읽고 본인의 컴퓨터에 Knowledge Agent를 설치해 예제를 따라 해 보세요.
    - 더 깊이 이해하려면 직접 샘플 에이전트를 실행해 보고 다양한 유형의 질문을 시도해 보길 권장합니다.
    - 샘플 지식 에이전트는 문서 질의, 복잡한 질문에 대한 지능형 계획, 요약, 문서 분석을 지원합니다.
    - 지식 베이스는 로컬의 knowledge_files 폴더에 저장된 PDF 파일들로 구성되어 있습니다.
    - 이 **폴더에 본인의 PDF를 추가하면, 에이전트가 시작 시 자동으로 처리**합니다:
        
        ```bash
        ~/llm-model-serving/ch04/KnowledgeAgent/ **ls knowledge_files**
          5-Level Paging and 5-Level EPT.pdf
          A Brief Introduction to the SAL.pdf
          A Brief Tutorial on Database Queries.pdf
          ...
        ```
        
    - 에이전트가 실행되면 다양한 질문을 할 수 있습니다. 예를 들어 다음과 같은 직접 정보 쿼리가 될 수 있습니다:
        - "5단계 페이징이란 무엇이며 어떻게 작동하나요?"
        - "패트리샤는 무엇을 시도하며, 그것들은 어떻게 최적화되나요?"
        
    - 또한 다음과 같은 고급 요약 및 분석 작업일 수도 있습니다:
        - "이 기술들은 현대 컴퓨팅 시스템과 어떻게 관련이 있나요?"
        - "데이터베이스 쿼리 최적화와 데이터 구조 최적화 간의 상세 비교를 작성하십시오."
        
    - 다음 예시는 에이전트가 로컬에서 실행되는 모습을 보여줍니다:
        
        ```bash
        ~/llm-model-serving/ch04/KnowledgeAgent/ **python agent.py**
        Your question: **What are the main types of database queries discussed in the tutorial?**
        Response:
        This tutorial by Lutz Hamel explores the key tools used in modern
        relational database systems: database queries, data mining, and OLAP
        ...
        ```
        
    - 추가 예제는 README의 Example Queries 섹션에서 확인할 수 있습니다.
    
- **[옵션/실습]** Knowledge Agent - README.md **⇒ OpenAI API 키 필요**
    - **OpenAI API를 백엔드로 쓰는 RAG 기반 지식 에이전트 데모. 로컬 모델 없이 PDF 문서를 질의/분석.**
    - 구성 요소
        - `agent.py`(오케스트레이터) → `rag_system.py`(PDF 처리·임베딩·벡터 검색), `llm_manager.py`(OpenAI API/토큰 관리),
        - `planner.py`(LLM 기반 실행 계획 수립), `actions.py`(질의/요약/분석 등 액션 실행), `config.py`(중앙 설정).
        - 아키텍처는 **User Query → Agent → OpenAI API**로 단순하고, Agent가 내부적으로 4개 컴포넌트를 조율하는 구조.
        
        ```bash
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │   User Query    │───▶│     Agent       │───▶│   OpenAI API    │
        └─────────────────┘    └─────────────────┘    └─────────────────┘
                                      │
                                      ▼
                               ┌─────────────────┐
                               │   Components    │
                               │                 │
                               │ • **RAG System**    │
                               │ • LLM Manager   │
                               │ • Planner       │
                               │ • Actions       │
                               └─────────────────┘
        ```
        
    - **"5-Level Paging이 무엇인가?” 질문 시**
        
        ```mermaid
        sequenceDiagram
        
            participant U as User
            participant A as Agent
            participant P as Planner
            participant R as RAG
            participant L as LLM
        
            U->>A: 질문
        
            A->>P: 어떤 작업이 필요한가?
        
            P->>L: 실행 계획 생성 요청
            L-->>P: 검색 → 분석 → 요약
        
            P-->>A: 실행 계획
        
            A->>R: 관련 PDF 내용 검색
            R-->>A: 관련 Chunk
        
            A->>L: 질문 + 관련 문서
        
            L-->>A: 분석 결과
        
            A-->>U: 최종 답변
        ```
        
        - Agent는 관련 PDF 내용을 검색하고, 필요한 문맥을 LLM에게 전달해서 답을 만듭니다.
        - 이 Agent는 문서 질의, 요약, 분석, 여러 문서 간 비교 등을 지원하도록 설계되어 있습니다.
        - 여기서 중요한 것은 **LLM이 혼자 모든 것을 하는 게 아니라는 것**입니다.
            - *LLM = 생각하고 답을 생성*
            - *RAG = 필요한 자료 찾기*
            - *Planner = 작업 순서 결정*
            - *Agent = 전체 조정*
    - **테스트**
        - test_agent.py : 구조 검증, API 불필요
        - test_rag_system.py : 실제 RAG 파이프라인, API 필요
        - test_api_key.py : 키 유효성 진단
    - **비용/보안**
        - 임베딩 : ~$0.01–0.05/1000페이지, LLM 호출 : ~$0.10–0.50/쿼리 *← 실제 과금 발생*
        - .env는 절대 커밋 금지, 키 로테이션 권장
        
    - **Step 1: Setup Environment**
        
        ```bash
        # Clone the repository
        git clone <repository-url>
        cd KnowledgeAgent
        
        # Create and activate virtual environment
        python -m venv venv
        
        # On macOS/Linux:
        source venv/bin/activate
        
        # On Windows:
        # venv\Scripts\activate
        
        # Install dependencies in virtual environment
        pip install -r requirements.txt
        
        # Create environment file
        cp env_example.txt .env
        
        # Edit **.env** with your OpenAI API key
        OPENAI_API_KEY=your-actual-api-key-here
        ```
        
    - **python test_api_key.py 실행** : 소액 과금(최소 completion 1건 + embedding 1건 + 모델 목록 조회)이 발생, 키가 실제로 동작하는지 확인
        
        ```bash
        **python test_api_key.py**
        *...
        ✅ Valid OpenAI API key format
        
        🧪 Test 1: Testing completion API...
        ✅ Completion test passed: Hello, API test successful!
        
        🧪 Test 2: Testing embeddings API...
        ✅ Embeddings test passed: 1536 dimensions
        
        🧪 Test 3: Testing model access...
        ✅ GPT-4 available: True
        ✅ text-embedding-3-small available: True
        
        🎉 All API tests passed!*
        
        # API 키가 실제로 정상 동작합니다.
        - Completion API: 정상 응답 ("Hello, API test successful!")
        - Embeddings API: 정상 (1536차원 벡터)
        - 모델 접근 권한: GPT-4, text-embedding-3-small 둘 다 사용 가능
        ```
        
    - **python test_rag_system.py** 실행 : OpenAI API + 실제 PDF로 검증하는 통합 테스트 - 총 11개 테스트 메서드
        
        ```bash
        **테스트 분류**
        ...
        
        **11/11 전부 통과했습니다.
        ...**
        ```
        
    - **agent.py** : 4개 컴포넌트(RAGSystem, LLMManager, Planner, ActionExecutor)를 조율하는 Facade 역할의 오케스트레이터.
        
        ```bash
        # 코드 분석
        Agent.__init__: Config → RAGSystem/LLMManager/Planner/ActionExecutor 순으로 생성. 
        이 시점에 LLMManager 내부에서 OpenAI 클라이언트가 만들어지므로, 여기서부터 API 키가 필요합니다.
        
        # process_query(query, use_planning=True) — 핵심 진입점, 두 경로가 있습니다:
        - 플래닝 경로 (기본값): Planner.create_plan()이 LLM(GPT)에게 "이 질문에 어떤 액션들을 어떤 순서로 실행할지" JSON으로 계획을 짜게 시킵니다 ({"plan": [...], "reasoning": ..., "estimated_steps": ...}). 사용 가능한 액션은 4개뿐:
          - query_rag_with_context (RAG 검색 + 답변)
          - generate_profile_based_response (사용자 프로필 반영 답변)
          - generate_summary (요약)
          - generate_analysis (심층 분석)
        
        # LLM 응답 파싱이 실패하면 _create_fallback_plan()이 키워드 매칭("what/how"→RAG 1스텝, "summarize"→RAG+요약 2스텝, "analyze/compare"→RAG+분석 2스텝)으로 대체 계획을 만듭니다 — 플래닝 자체가 실패해도 완전히 멈추지 않는 방어적 설계입니다.
        - 비플래닝 경로 (use_planning=False): 플래닝을 건너뛰고 query_rag_with_context 하나만 바로 실행 — 빠르고 저렴한 대신 다단계 추론은 없음.
        _execute_action_sequence: 계획된 액션들을 순서대로 실행하면서 앞 액션의 결과(50자 이상일 때만)를 다음 액션의 context로 체이닝합니다. 즉 ["query_rag_with_context", "generate_summary"] 계획이면, 1단계는 RAG로 검색한 문서를 컨텍스트로 답변을 생성하고, 2단계는 그 답변 자체를 컨텍스트로 삼아 요약합니다 (원본 문서를 다시 검색하는 게 아님). 액션 단위로 try/except가 걸려 있어 한 스텝이 실패해도 에러 문자열만 결과에 남기고 나머지 스텝은 계속 진행됩니다.
        
        # 주의할 조용한 실패 지점: validate_action_prerequisites()는 len(self.rag_system.documents) > 0만 확인하는데, build_knowledge_base()를 먼저 호출하지 않으면(문서가 0개면) 모든 액션이 에러 없이 조용히 스킵되고 results가 빈 리스트로 나옵니다 — process_query가 success: True에 final_response: "No response generated"를 반환하므로, 겉보기엔 "성공했는데 답이 없다"는 헷갈리는 상태가 될 수 있습니다.
        
        # 최상위 안전망: process_query 전체가 try/except로 감싸져 있어 어떤 예외든 success: False + 에러 메시지 dict로 변환되고, interactive_mode()는 이 dict만 보고 성공/실패를 분기해 출력합니다 — 에이전트 프로세스 자체가 죽지 않습니다.
        
        # main(): .env 재로드(사실 config.py import 시 이미 한 번 로드되므로 약간 중복이지만, override=True라 최신값 보장 목적) → Agent() 생성 → build_knowledge_base()(PDF 42청크 임베딩, 실제 비용 발생 — 앞서 테스트에서 본 것과 동일) → interactive_mode() 진입.
        ```
        
    - **python agent.py** 로 실제 질의 : ***응답 하나에 실제로 발생한 API 호출 4회 확인!***
        
        !https://devlos.tistory.com/129
        
        https://devlos.tistory.com/129
        
        ```bash
        #
        cd ch04/KnowledgeAgent
        source venv/bin/activate
        **python agent.py**
        ----------------------
        # .env 로드 메시지 확인
        # PDF 4개를 42청크로 임베딩 (임베딩 API 호출)
        # Your question: 프롬프트가 뜨면 자연어로 질문 입력
        # 내부적으로 플래닝 LLM 호출 1회 + 액션당 LLM 호출 1~2회가 실행된 뒤 ✅ Response:로 최종 답변 출력
        # quit 입력 시 종료
        
        # 첫번째 질문
        *💬 Your question:* **"What is 5-level paging and how does it work?"**
        *🔄 Processing...
        INFO:__main__:Processing query: What is 5-level paging and how does it work?
        INFO:planner:Creating plan for query: What is 5-level paging and how does it work?
        **INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"**
        INFO:planner:Created plan: {'plan': ['generate_summary', 'generate_analysis'], 'reasoning': 'The plan begins with generating a concise summary to explain 5-level paging clearly, followed by an analysis to provide a detailed understanding of how it works. This approach ensures both a quick overview and an in-depth explanation.', 'estimated_steps': 2}
        INFO:__main__:Execution plan: {'plan': ['generate_summary', 'generate_analysis'], 'reasoning': 'The plan begins with generating a concise summary to explain 5-level paging clearly, followed by an analysis to provide a detailed understanding of how it works. This approach ensures both a quick overview and an in-depth explanation.', 'estimated_steps': 2}
        INFO:__main__:Action sequence: ['generate_summary', 'generate_analysis']
        INFO:__main__:Executing action 1/2: generate_summary
        INFO:actions:Executing action: generate_summary
        INFO:actions:Executing generate_summary action
        INFO:rag_system:Searching for query: What is 5-level paging and how does it work?
        **INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"**
        INFO:rag_system:Found 5 relevant documents
        **INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"**
        INFO:actions:Successfully generated summary
        INFO:__main__:Action generate_summary completed successfully
        INFO:__main__:Executing action 2/2: generate_analysis
        INFO:actions:Executing action: generate_analysis
        INFO:actions:Executing generate_analysis action
        **INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"**
        INFO:actions:Successfully generated analysis
        INFO:__main__:Action generate_analysis completed successfully
        
         Response:
        Certainly! Here is a comprehensive analysis of 5-level paging based on the provided context:
        
        ---
        
        ### 1. **Overview of 5-Level Paging**
        
        **Definition:**
        5-level paging is an architectural enhancement introduced to extend the virtual address space beyond the traditional limits imposed by 4-level paging. It adds an additional hierarchical level (PML5) to the paging structure, thereby increasing the maximum linear address size from 48 bits to 57 bits.
        ...*
        ```
        
        ```bash
        # 실행 흐름 추적
        
        1) 지식 베이스 빌드: PDF 4개 → 42청크, 임베딩 API 1회 호출 — 이전 test_rag_system.py 실행 때와 동일한 패턴입니다.
        
        2) 플래닝 단계 (Planner.create_plan, chat/completions 1회 호출):
        plan: ['generate_summary', 'generate_analysis']
        reasoning: "요약으로 개요를 먼저 주고, 분석으로 상세 설명을 이어가겠다"
        여기서 주목할 점 — 이 질문("What is X and how does it work?")은 _create_fallback_plan()의 키워드 규칙대로라면 "what"/"how"에 매칭돼 ["query_rag_with_context"] 1스텝으로 처리됐어야 합니다. 하지만 실제로는 LLM 플래너가 살아있었기 때문에(폴백으로 안 떨어짐) generate_summary + generate_analysis 2스텝이라는, 하드코딩된 휴리스틱과는 다른 독자적 판단을 내렸습니다. "LLM 기반 플래닝이 규칙 기반보다 더 똑똑한 선택을 한다"는 걸 실제 로그로 확인한 셈입니다.
        
        3) 액션 1/2 — generate_summary:
        - context가 비어있으므로 rag_system.get_context_for_query() 호출 → 쿼리 임베딩 1회(POST .../embeddings) → 관련 문서 5개 검색
        - 검색된 컨텍스트로 요약 프롬프트 생성 → chat/completions 1회 → 요약 텍스트 생성
        
        4) 액션 2/2 — generate_analysis:
        - 로그를 보면 이 단계 직전에 embeddings 호출이 없습니다 — chat/completions 딱 1번만 있습니다. 이게 바로 지난번 코드 분석에서 짚었던 컨텍스트 체이닝이 실제로 일어난 증거입니다: generate_summary의 결과(50자 초과)가 context로 재사용되면서, generate_analysis는 원본 문서를 다시 검색하지 않고 방금 생성된 요약문 자체를 입력 삼아 분석을 작성했습니다.
        
        5) 최종 출력의 함정: process_query는 results-1만 final_response로 반환하고, interactive_mode()는 그것만 화면에 찍습니다. 즉 1단계(generate_summary)의 실제 출력 텍스트는 화면에 전혀 안 보였고, 내부적으로 2단계의 입력 재료로만 소비됐습니다 — 사용자는 최종 분석 결과만 봤지만, 실제로는 "먼저 요약 → 그 요약을 바탕으로 분석"이라는 2단계 추론이 뒤에서 일어난 것입니다.
        
        **# 응답 하나에 실제로 발생한 API 호출**
        ┌───────────────────────────┬──────┐
        │           호출             │ 횟수  │
        ├───────────────────────────┼──────┤
        │ chat/completions (**플래**닝)   │ 1    │
        ├───────────────────────────┼──────┤
        │ embeddings (**쿼리 검색**)      │ 1    │
        ├───────────────────────────┼──────┤
        │ chat/completions (**요약**)    │ 1    │
        ├───────────────────────────┼──────┤
        │ chat/completions (**분석**)    │ 1    │
        ├───────────────────────────┼──────┤
        │ **합계**                       │ **4회**  │
        └───────────────────────────┴──────┘
        ***INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"**
        **INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"**
        **INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"**
        **INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"***
        
        # 사용자가 던진 질문은 단 하나인데, 실제로는 LLM 호출 3번 + 임베딩 1번이 연쇄적으로 일어났습니다
        이게 정확히 4장 도입부에서 읽으셨던 "단일 사용자 상호작용이 다중 LLM 호출·토큰 사용량 증가를 촉발한다"는 서술의 실제 사례입니다. 
        단순 request-response 서빙과 달리, 이 한 번의 질의를 처리하는 데 서빙 백엔드가 4번의 왕복을 감당해야 했다는 뜻이고,
        이게 tail latency가 체인 호출마다 누적되는 이유이기도 합니다.
        
        # 응답 품질
        결과물은 PML5, 57비트 주소, canonical address 규칙, EPT, CR4.LA57 활성화 등 실제 Intel 문서(knowledge_files/5-Level Paging...pdf)의
        기술적 세부사항을 정확히 반영하고 있고, "Examples from the Context" 섹션에서 원문을 직접 인용하는 형태까지 보여줘서
        RAG grounding이 정상적으로 동작하고 있음을 확인할 수 있습니다.
        ```
        
        ```bash
        # 두번째 질문
        *💬 Your question:* **"Compare the optimization techniques mentioned across all documents"**
        ...
        ```
        
    
- (참고) RAG 동작 - 악분님, 상세 , 베이스라인 RAG 구현: 같은 검색, 다른 응답
    - **RAG와 Model Customization의 관계**
        - RAG는 외부 지식 주입에 강하다.
            - 최신 정보나 사내 문서를 빠르게 반영할 수 있다.
            - vector database만 갱신하면 모델 재학습 없이 지식을 갱신할 수 있다.
            - context window 제한 때문에 관련도 높은 조각만 넣는 방식이 중요하다.
                
                !An example of RAG pipeline
                
                An example of RAG pipeline
                
                ![https://ryusstory.tistory.com/entry/도메인-특화-LLM을-위한-세-가지-접근법-프롬프트-RAG-파인튜닝](attachment:92a284a2-f75b-469f-a4c6-a0fe130927e7:image.png)
                
                https://ryusstory.tistory.com/entry/도메인-특화-LLM을-위한-세-가지-접근법-프롬프트-RAG-파인튜닝
                
                - RAG는 knowledge base를 문서 처리와 embedding model을 통해 vector database에 저장하고, 사용자 질의 시 semantic similarity로 관련 context를 찾아 LLM 입력에 주입한다.
        
    - RAG 동작 단계
        
        !https://docs.aws.amazon.com/ko_kr/prescriptive-guidance/latest/retrieval-augmented-generation-options/what-is-rag.html
        
        https://docs.aws.amazon.com/ko_kr/prescriptive-guidance/latest/retrieval-augmented-generation-options/what-is-rag.html
        
    - **핵심: LLM 입장에서는 input이 길어질 뿐 - Blog**
        - RAG를 처음 접하면 “검색된 문서가 LLM 내부의 Q 행렬에 들어간다”거나 “RAG의 query와 attention의 Q는 같다”는 직관을 갖기 쉽다. *틀렸다*.
        - LLM 입장에서 RAG는 **단순히 prompt 텍스트가 길어진 것일 뿐**이다. 외부에서 검색했든, 사람이 직접 붙여넣었든, LLM은 구분하지 못한다. 아래 흐름을 보자.
        
        !https://malwareanalysis.tistory.com/930
        
        https://malwareanalysis.tistory.com/930
        
        !image.png
        
        ```python
        사용자 query: "회색 정장에 어울리는 셔츠 추천해줘"
                                    ↓
                 [embedding model — RAG 전용, 예: text-embedding-ada-002]
                                    ↓
                          query 벡터 (1536차원)
                                    ↓
                      [Vector DB — cosine similarity 검색]
                                    ↓
                top-5 문서 (셔츠 카탈로그 row 5개의 텍스트)
                                    ↓
        ─────────────────────────────────────────────────
        여기부터는 그냥 "텍스트 합치기" — 행렬 주입이 아님
        
        prompt = f"""
        {retrieved_doc_1_text}
        {retrieved_doc_2_text}
        ...
        {retrieved_doc_5_text}
        
        Question: 회색 정장에 어울리는 셔츠 추천해줘
        """
        ─────────────────────────────────────────────────
                                    ↓
                        [LLM (gpt-3.5-turbo 등)]
                                    ↓
                                  토큰화
                                    ↓
                    X (입력 임베딩 행렬, row가 그만큼 늘어남)
                                    ↓
                       평소와 똑같은 self-attention
                                    ↓
                                   응답
        ```
        
        - *비유하자면, RAG는 **오픈북 시험**이다. 매 시험(요청)마다 책(외부 DB)에서 찾아서 답안(prompt)에 첨부한다. 반대 방향의 접근인 Fine-tuning(파인튜닝)은 **암기** — 가중치에 지식을 내재화하는 방식이다.*
        
- Knowledge Agent의 구조 Design
    - 먼저 Knowledge Agent의 구조를 살펴보겠습니다. 그림 4-1은 그 주요 구성 요소들을 시각적으로 개괄적으로 보여줍니다.
        
        !Figure 4-1. Knowledge Agent system overview
        
        Figure 4-1. Knowledge Agent system overview
        
    - Knowledge Agent는 두 개의 OpenAI 모델로 역할 분리 동작
        - **text-embedding-3-small**: 텍스트를 의미를 담은 숫자 벡터로 변환 ***⇒ Emdedding***
            - 시맨틱 검색·검색(retrieval)·콘텐츠 매칭에 쓰임.
            - 방금 실행 로그에서 본 POST .../embeddings 호출들이 전부 이 모델 (PDF 42청크 벡터화, 사용자 쿼리 벡터화).
        - **gpt-4.1-nano**: 에이전트의 추론 및 언어 생성 엔진 ***⇒ LLM***
            - 지시를 해석하고, 다음 행동을 계획하고, 자연어 응답을 만듭니다
            - 로그의 POST .../chat/completions 호출 3번(플래닝 1 + 요약 1 + 분석 1) 이 모델.
        
    - 모델 외에도 Knowledge Agent는 다음과 같은 핵심 구성 요소들로 이루어져 있습니다:
        
        ```bash
        사용자 질문 → **Agent.process_query()**
                      ├─ **Planner** (gpt-4.1-nano) → 실행 계획 수립
                      └─ **ActionExecutor 순차 실행**
                            ├─ **RAGSystem** (text-embedding-3-small) → 관련 문서 검색
                            └─ **LLMManager** (gpt-4.1-nano) → 각 액션별 응답 생성
        ```
        
        - *Knowledge Agent (orchestrator)* : 모든 구성 요소를 조정하는 중앙 컨트롤러
            - 전체를 조율하는 중앙 컨트롤러 : process_query()가 플래닝 → 액션 실행 → 최종 응답까지 지휘
        - *Retrieval-augmented generation (RAG) system* : PDF를 처리하고, 임베딩을 생성하며, 벡터 검색을 수행합니다
            - PDF 처리(42청크로 분할), 임베딩 생성, 벡터 검색(search()) 수행
        - *Planner* : 사용자 쿼리를 위한 지능적인 실행 계획을 생성하기 위해 LLM을 활용합니다
            - LLM을 활용해 지능형 실행 계획 수립
            - "What is 5-level paging...” 질문에 generate_summary → generate_analysis 2단계 계획을 LLM 스스로 결정
        - *Actions (executor)* : 질의, 요약, 분석과 같은 특정 작업을 수행합니다
            - 질의/요약/분석 등 구체적 작업 실행 : 순차 실행되며 앞 결과를 다음 컨텍스트로 체이닝
        
    - Agent는 보통 다음 구성 요소를 가진다.
        - **LLM manager**: planning과 generation을 담당한다.
        - **Action registry**: agent가 호출 가능한 action 목록을 관리한다.
        - **RAG component**: 문서 검색과 context retrieval을 수행한다.
        - **Workflow executor**: plan에 따라 action을 순서대로 실행한
    
- The Agent’s Internal Workflow 에이전트의 내부 워크플로우
    - 이제 Knowledge Agent의 내부 로직을 살펴보겠습니다. 그림 4-2는 사용자 쿼리를 처리하는 워크플로우를 보여줍니다.
    - 그래프에 나타난 'LLM 서빙'은 액션 시나리오에 따라 동일한 LLM 모델일 수도, 서로 다른 모델일 수도 있다는 점에 유의하세요.
        
        !Figure 4-2. Knowledge Agent가 사용자 query에 응답하는 내부 workflow
        
        Figure 4-2. Knowledge Agent가 사용자 query에 응답하는 내부 workflow
        
    - 그림 4-2에 나와 있는 9단계 워크플로우를 단계별로 살펴보겠습니다:
    1. **사용자 질문 입력** : 예를 들어, "데이터베이스 쿼리 최적화와 데이터 구조 최적화 간의 상세 비교를 작성해 주세요.”
    2. **Agent → Planner 호출**해서 실행 계획 설계 요청 : 에이전트는 먼저 실행 계획을 설계하기 위해 자신의 **Planner** 컴포넌트를 **호출**합니다.
    3. **Planner → LLM 호출** :질문과 에이전트의 가용 액션들을 바탕으로 **실행 계획** 생성. 이 예시에서는 **3단계 실행 계획**이 나옴:
        
        ```bash
        {
          "plan": ["**query_rag_with_context**", "**generate_analysis**", "**generate_summary"**],
          "reasoning": "First, retrieving relevant contextual information ensures a comprehensive understanding...",
          "estimated_steps": 3
        }
        ```
        
        - ***3단계 sequential plan** : query_rag_with_context, generate_analysis, generate_summary*
            - ***RAG로 context를 찾고, analysis를 만들고, summary를 생성하는 순서***
    4. **ActionExecutor**가 **순서대로 실행(3단계)** 시작 : 먼저 action_query_rag_with_context를 처리.. (*actions.py 참고)*
    5. **1단계 query_rag_with_context**: RAG 시스템이 지식 베이스에서 관련 문서를 찾고(시맨틱 검색), 그 문서를 컨텍스트로 LLM이 답변을 생성. 
        
        ```bash
        Document 1 (Source: A Brief Tutorial on Database Queries, Data Mining, and OLAP ...):
        A Brief Tutorial on Database Queries, Data Mining, and OLAP
        ...
        ```
        
        - *RAG로 가져온 문서는 **analysis prompt에 들어간다***
    6. **2단계 generate_analysis**: 사용자 질문 + 5단계의 결과를 컨텍스트로 LLM에 전달해 더 심층적인 비교 분석 생성
    7. LLM이 질문과 컨텍스트를 바탕으로 심층 비교 분석을 만들어냄
    8. **3단계 generate_summary**: 직전 단계(분석)의 결과를 LLM에게 요약시켜 마무리
    9. Agent가 모든 **액션 출력을 종합해 최종 결과를 사용자에게 반환**
    
- Agent Autonomy 에이전트 자율성
    - **핵심 대비**: 절차적 실행 vs 목표 주도 자율성
        - 전통적 애플리케이션: 고정된 로직 + 사전 정의된 워크플로우로 명령을 절차적으로 실행
        - LLM 기반 에이전트: 사용자 의도를 해석 → 접근 방식을 스스로 선택 → 여러 도구를 통합 → 최소한의 개입으로 결과 전달 (= 목표 주도 자율성)
        
    - **액션(Actions) 정의**
        - 자율성을 가능케 하는 건 **재사용 가능**한 **이산적(discrete) 액션 집합**입니다.
            - *인공지능(AI)과 강화학습에서 선택할 수 있는 **행동(Action)의 가짓수가 딱 떨어지게 셀 수 있는 경우**를 말합니다.*
            - *연속적이지 않고 **서로 명확하게 구분**되는 유한한 개수의 행동들로 구성됩니다.*
                - *예시) 비디오 게임: 조이스틱 조작 (예: '점프', '공격', '왼쪽 이동', '오른쪽 이동')*
        - 이 샘플 에이전트에서는 각 액션이 "**LLM 호출 + 특화된 프롬프트 템플릿**"으로 구현됩니다.
        - 예시로 든 **create_analysis_prompt**는 우리가 이미 actions.py의 generate_analysis()에서 확인한 그 **프롬프트 생성 로직**입니다.
            
            ```python
            # 사용자의 쿼리와 이를 뒷받침하는 맥락을 바탕으로 구조화된 분석을 생성합니다:
            # Analysis prompt는 질문과 context를 결합해 LLM이 근거 기반 분석을 하도록 만든다.
            
            def **create_analysis_prompt**(self, query: str, context: str) -> str:
                prompt = f"""
            You are an expert analyst. Please provide a detailed
            analysis of the following question based on the
            provided context.
            
            Question: {query}
            
            Context:
            {context}
            
            Please provide:
            1. A comprehensive analysis
            2. Key insights and findings
            3. Relevant examples from the context
            4. Any limitations or gaps in the available information
            
            Analysis:
            """
                return prompt
            ```
            
            - *이 샘플 에이전트는 의도적으로 LLM 호출만 쓰는 단순한 구조를 유지합니다*
        - 에이전트의 액션이 **LLM 프롬프트에 국한되지 않는다!**
            - 도구 호출: 웹 검색, API 질의, DB 명령 실행
            - 시스템 연산: 파일 관리, 워크플로우 트리거, 외부 서비스 상호작용
            - 추론 단계: 사용자에게 직접 노출되지 않는 중간 계산/계획 서브루틴
    
    - **LLM 기반 플래닝** Planning with LLMs
        - Planner가 사전 정의된 워크플로우 대신 L**LM에게 고수준 지시를 해석**시켜 **서브태스크로 분해하고 최적 실행 경로를 결정**하게 함으로써,
        - **사용자**가 "PDF 파싱, 관련 섹션 찾기, 결과 요약" 같은 걸 **수동으로 안 해도 되게 만듭니**다.
            
            ```python
            # Planning 단계는 available actions를 기반으로 LLM에게 plan을 생성시키고 JSON response를 파싱한다.
            
            def create_plan(self, query: str) -> Dict[str, Any]:
              planning_prompt = self.llm_manager.create_planning_prompt(
                 query,
                 self.available_actions
              )
              plan_response = self.llm_manager.generate_response(
                 planning_prompt,
                 temperature=0.3
              )
              plan = self._parse_plan_response(plan_response)
              return plan
            ```
            
        
    - **Model Context Protocol (MCP**) : 도구 사용을 위한 대표적 표준화 접근법
        - LLM이 환경에서 사용 가능한 도구를 발견(discover)하고,
        - 구조화된 입력으로 호출(call)하고,
        - 결과를 추론 과정에 다시 반영하는 일관된 인터페이스 제공.
        - 도구 정의를 에이전트 핵심 로직과 분리해 개발을 단순화하고,
        - ad-hoc 프롬프트 엔지니어링의 취약성(brittleness) 문제를 완화.
        - *Model Context Protocol(MCP)은 agent가 외부 tool, data source, application과 표준 방식으로 연결되도록 돕는 프로토콜이다.*
        - *Agentic workflow가 늘어날수록 tool interface와 permission boundary가 중요해진다.*
    
- Retrieval-Augmented Generation (RAG) : 질문할 때 필요한 자료를 찾아서 LLM에게 같이 넣어주는 방법
    - RAG가 필요한 이유 : LLM 단독의 3가지 한계:
        1. 지식이 고정적 : 학습 데이터 컷오프 이후 정보 없음
        2. 환각(hallucination) : 그럴듯하지만 사실과 다른 내용 생성 가능
        3. 전문/최신 정보 부족 : 도메인 특화 지식 갭
        - ***RAG는 쿼리 시점에 외부 지식을 검색해 LLM에 주입함으로써 이 문제를 완화**합니다.*
        - *모델 내부 학습 데이터에만 의존하지 않고, **도메인 특화·최신 정보로 응답을 보강**합니다.*
        - *RAG는 L**LM이 자체 parameter memory만 의존하지 않고,** 외부 문서에서 관련 context를 검색해 답변하도록 만드는 방식이다.*
        
    - 그림 4-3은 기본 RAG 시스템의 두 가지 주요 워크플로우를 보여줍니다.
        
        !Figure 4-3. 기본 RAG system은 index building workflow와 query/retrieval workflow로 구성된다
        
        Figure 4-3. 기본 RAG system은 index building workflow와 query/retrieval workflow로 구성된다
        
    - **Index-building workflow 인덱스 빌딩 워크플로우 (오프라인 프로세스) : 먼저 PDF를 미리 처리합니다.**
    - 문서 정제/파싱 → 청킹(보통 ~1000토큰) → 청크별 임베딩 계산(사전에 대량으로, 벌크 배치 추론) → 벡터 DB에 저장.
    - 인덱스 빌딩 워크플로우(그림 4-3의 A 부분)는 **오프라인 데이터 처리 과정**으로, 일반적으로 **주기적으로 실행되도록 예약**됩니다.
        - *우리 샘플 에이전트는 이걸 단순화해서 매번 agent.py 기동 시 인메모리로 다시 빌드합니다.*
        - *우리가 테스트를 4번 돌릴 때마다 42개 청크를 매번 재임베딩했던 바로 그 이유*
    - 먼저 HTML이나 PDF 같은 형식의 **원시 지식 문서를 정제하고 파싱**하여 **일반 텍스트로 변환**합니다. 이 텍스트는 보통 각각 약 1,000 토큰 단위로 나누어진 **덩어리들로 분할**됩니다.
    - **청킹**(큰 콘텐츠를 더 작은 덩어리로 나누는 것)은 **단순한 임베딩 전처리 단계 이상**입니다. 이는 **검색의 세분화 정도**(시스템이 한 번에 고려할 수 있는 텍스트 양)를 **정의**하며, **오프라인 배치 추론**이 이루어지는 **단계 역할**을 합니다.
    - 전체 문서를 필요할 때마다 임베딩하는 대신, **시스템이 각 청크별 임베딩을 대량으로 미리 계산**해 쿼리 시점에 훨씬 빠르고 효율적으로 검색할 수 있게 합니다.
    - 청킹이 완료되면 각 구간은 임베딩 모델을 사용해 조**밀한 벡터 표현으로 인코딩**됩니다. 이렇게 생성된 청크 임베딩은 **벡터 데이터베이스에 인덱싱**되어 저장되며, 온라인 쿼리 과정에서 **효율적인 유사도 검색의 기반**이 됩니다.
        
        
    - **Fine (Small) Versus Coarse (Large) Chunking ‘작은 청킹’과 ‘큰 청킹’ 비교 트레이드오프**
        - 작은 청크: 검색 정밀도 ↑, 단 문맥 손실 위험
        - 큰 청크: 문맥 풍부, 단 관련 없는 내용 섞여 정밀도 ↓ (dilution)
        - *최적값은 도메인과 LLM 컨텍스트 윈도우에 따라 달라진다*
        
    - **Query/retrieval workflow 질의/검색 워크플로우(온라인 프로세스)**
    - 쿼리 텍스트 임베딩 → 벡터 DB에서 코사인 유사도로 최근접 벡터 검색 → 관련 청크 반환 → LLM에 쿼리+청크를 함께 전달해 답변 생성.
    - 쿼리 워크플로우(그림 4-3의 파트 B)는 고객이 상담원과 상호작용할 때 **실시간으로 이루어지는 온라인 프로세스**입니다.
    - 에이전트가 쿼리를 받으면, 먼저 임베딩 모델을 사용해 쿼리 텍스트의 임베딩 벡터를 얻습니다.
    - 그다음 코사인 유사도 같은 지표를 사용해 데이터베이스 내에서 쿼리 벡터와 가장 가까운 벡터들을 찾아냅니다.
    - 에이전트는 선택된 벡터를 사용해 벡터 데이터베이스에서 가장 관련성 높은 문서 조각을 찾아내고, 이를 사용자 쿼리와 함께 LLM에 전달해 답변을 생성하도록 합니다.
    - 그림 4-3은 단순하고 '기본적인’ RAG 파이프라인을 보여주지만, 실제 운영 환경의 RAG 시스템은 청킹, 랭킹, 중복 제거, 다중 소스 검색 같은 추가적인 복잡성을 포함합니다.
    - *Online 과정에서는 user query를 embedding하고, vector search로 관련 chunk를 찾고, 그 chunk를 prompt context로 넣어 LLM response를 생성한다.*
    - *텍스트를 작은 chunk로 나누는 이유는 retrieval 단위의 정확도를 높이고, context window를 효율적으로 쓰기 위해서다.*
        
        
    - **Why Split Text into Small Chunks? 왜 텍스트를 작은 덩어리로 나눌까요? 왜 청킹이 필요한가?**
        - *LLM은 입출력 합산 토큰 수에 상한(컨텍스트 윈도우)이 있기 때문에, 문서 전체가 아니라 가장 관련성 높은 청크 몇 개만 골라 보내야 합니다*
        - LLM은 입력과 출력 토큰을 포함해 한 번에 처리할 수 있는 **최대 토큰 수에 제한**이 있습니다. 이는 **맥락을 담을 공간이 제한적**이라는 뜻입니다.
        - 그 공간을 **컨텍스트 윈도우**라고 합니다. 이러한 제약을 고려해 RAG는 추출한 텍스트를 더 작고 이해하기 쉬운 조각들로 나누어, LLM에 전달되는 문맥은 **관련성이 높은 일부 조각만 전송합**니다.
    
- Cache-Augmented Generation (CAG) : 미리 긴 Context를 LLM에 입력해 계산하고, 해당 Knowledge의 Context/KV Cache를 재사용
    - 이전 섹션에서는 RAG가 외부 지식 소스를 통합하여 언어 모델을 어떻게 향상시키는지 살펴보았습니다. 강력한 만큼 RAG는 몇 가지 과제도 함께 가져옵니다.
    - **RAG의 한계**
        - **추가 지연시간**: LLM 호출 전에 검색 단계가 끼어듦, "쿼리 임베딩 → 코사인 유사도 검색" 과정 자체가 지연 요인
        - **선택 오류 위험**: 제한된 토큰 안에서 "가장 관련 있는" 문서만 골라야 하는데, 잘못 고를 수 있음
        - **시스템 복잡도**: 임베딩·인덱스·벡터 DB를 구축/유지해야 하는 부담
            
            
    - **CAG의 등장 배경: 커진 컨텍스트 윈도우**
        - Claude Sonnet 4(2025년 8월 기준) 같은 모델이 100만 토큰 컨텍스트를 지원하게 되면서, "몇 개만 골라 넣는다"는 RAG의 전제 자체가 흔들립니다.
        - 지식 베이스 전체(또는 상당 부분)를 통째로 컨텍스트에 넣을 수 있다면, 굳이 복잡한 검색 시스템 없이도 LLM의 고질적 한계(정적 지식, 환각, 도메인 갭)를 해결할 수 있다는 것.
        
    - **CAG란**
        - *쿼리 시점에 검색하는 대신, 지식을 미리 LLM의 KV 캐시에 프리로드해두고 추론 시 그 캐시된 컨텍스트로 바로 답하는 방식입니다.*
        - *→ 검색 지연 제거 + 시스템 복잡도 감소, 그러면서도 외부 지식 기반 응답은 유지.*
        - 캐시 증강 생성(CAG)은 최신 LLM의 확장된 문맥 창을 활용해 **지식을 모델의 KV 캐시에 직접 미리 로드**합니다.
        - CAG는 질의 시점에 검색을 수행하는 대신, 관련 자원을 미리 로드해 두어 **추론 중에 캐시된 문맥을 활용해 답변을 생성**할 수 있도록 합니다.
        - 이 방법은 검색 지연을 없애고 시스템 복잡성을 줄이면서도, LLM이 외부 지식에 기반해 답변을 생성할 수 있도록 합니다.
        
    - 그림 4-4는 **RAG 시스템과 CAG 시스템을 비교**합니다.
        
        !Figure 4-4. RAG와 CAG 비교
        
        Figure 4-4. RAG와 CAG 비교
        
        - RAG: 쿼리마다 매번 동적으로 검색 프로세스 실행
        - CAG: 지식 문서를 한 번 캐시에 로드해두면, 이후 쿼리는 그 캐시된 컨텍스트로 바로 응답
        
    - RAG에서는 모든 쿼리가 동적으로 문맥을 찾기 위한 검색 과정을 촉발하는 반면, CAG에서는 지식 문서가 LLM 캐시에 로드되면 문맥 내 외부 지식을 활용해 쿼리에 직접 답변합니다.
    - 이러한 개선은 그림 4-5에 나타난 것처럼 에이전트 설계를 크게 단순화합니다.
        
        !Figure 4-5. CAG를 사용하는 agent workflow
        
        Figure 4-5. CAG를 사용하는 agent workflow
        
    - **CAG의 대가**
        - 큰 컨텍스트 윈도우 + 캐시 관리는 메모리/연산 요구량을 늘립니다 → 이걸 최적화하는 기법은 7장에서 다룸
        
    - 결론: RAG vs CAG, 양자택일이 아니다
        - **RAG는 동적 검색과 최신성**에 강하고, **CAG는 반복 query나 고정 knowledge set에서 latency를 줄이는 데 유리**하다.
        - RAG = 검색으로 외부 지식을 프롬프트에 확장 → 답변 품질(지식 접지·최신성) 개선
        - CAG = 이미 계산된 컨텍스트를 재사용해 중복 KV 캐시 연산 감소 → 서빙 효율(지연시간·처리량·비용) 개선
        - *RAG로 입력을 보강하고, CAG로 실행을 최적화하는 조합*
    
- Agent가 Model Serving을 사용하는 방식 How Agents Use Model Serving
    - **autonomous agents 자율 에이전트**는 작업을 완료하기 위해 여러 모델과 도구를 조율해야 하는 경우가 많습니다. 대표적인 구성 요소는 다음과 같습니다:
        - LLM : 추론, 계획, 대화 생성
        - 임베딩 모델 : 검색, 유사도 매칭, 시맨틱 매칭
        - 비전/음성 모델 : 멀티모달 인식
        - 태스크 전용 모델 : 코드 생성, 분류, 요약
        - 외부 도구 : API, DB, 서비스 (필요시 호출)
        
    - 현대 에이전트의 핵심 기능 중 하나는 **도구 호출**입니다. 툴 호출은 네 단계로 이루어진 과정입니다:
        
        ```mermaid
        sequenceDiagram
        
            participant U as User
            participant A as Agent
            participant L as LLM
            participant T as Tool
        
            U->>A: 질문
        
            A->>L: 질문 + 사용 가능한 Tool
        
            L-->>A: Tool 선택 + JSON arguments
        
            A->>T: Tool 실행
        
            T-->>A: Result
        
            A->>L: Tool Result 전달
        
            L-->>A: 다음 판단 / 최종 답변
        
            A-->>U: Answer
        ```
        
        1. LLM이 사용자 요청과 사용 가능한 도구들을 놓고 추론
        2. 선택한 도구에 필요한 입력을 인코딩한 구조화된 출력(보통 JSON)을 생성
        3. 에이전트가 그 도구 호출을 실제로 실행
        4. 결과를 다시 LLM에 넣고, 작업이 끝날 때까지 반복(iterate)
        - *이 패턴 덕분에 에이전트는 순수 텍스트 생성을 넘어 정밀한 도구 실행과 추론을 결합할 수 있습니다.*
        
    - 이 모든 모델·도구는 결국 **HTTP/gRPC API 같은 모델 서빙 서비스**를 통해 **온디맨드로 호출**됩니다.
    - 에이전트가 상호작용형·실시간으로 동작하기 때문에, 고**성능·저지연·비용 효율적인 서빙**이 **에이전트 애플리케이션 성공의 핵심 조건**입니다.
    - 이제 에이전트 기반 애플리케이션에서 모델과 도구가 어떻게 사용되는지 살펴보았으니, 이 장의 나머지 부분에서는 실제 서비스에 사용되는 프로덕션급 모델 구축을 위한 다양한 접근법을 탐구할 것입니다.