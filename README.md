# SoloMate Graph (LangGraph)

기타 솔로·보컬 연습을 돕는 **Education Agent** 예제입니다. LangGraph로 노드·조건부 분기·도구·체크포인트(메모리)를 묶었습니다.


## 무엇을 하는 프로젝트인가

1. **첫 노드 `analyze_progression`**  
   사용자가 말한 노래 제목(첫 줄)을 기준으로 시스템 프롬프트를 구성하고, LLM이 코드·분위기를 **도구로 웹 검색**한 뒤 연습 제안을 하도록 유도합니다.

2. **규칙 기반 분기 (`graph`)**  
   import만 하면 쓸 수 있는 소형 그래프로, 분석 직후 **보컬 키워드**가 있으면 보컬용 연습 문구 노드로, 없으면 기타(애드립)용 노드로 갈라집니다.

3. **LLM + 도구 루프 (`graph_llm`)**  
   스크립트 기본 진입점입니다. `chatbot`이 도구를 호출하면 `ToolNode`가 실행되고, `get_human_feedback` 호출 시 **interrupt**로 멈춘 뒤 CLI에서 `--resume` 또는 위치 인자로 피드백을 넣어 **재개**할 수 있습니다.

## 실행 방법

- **Python**: `>=3.13` (`pyproject.toml` 기준)
- **의존성 설치** (예: [uv](https://github.com/astral-sh/uv)): 프로젝트 루트에서 `uv sync` 후 가상환경을 활성화하세요.  
  `ModuleNotFoundError: langchain` 이 나오면 가상환경이 켜져 있는지, `uv sync`로 패키지가 설치됐는지 확인하세요.
- **환경 변수**: OpenAI 사용을 위해 `.env`에 `OPENAI_API_KEY` 등을 설정합니다.
- **CLI 예시** (LLM 그래프):

```bash
python main.py --new-thread --title "나는행복한사람"
```

interrupt 대기 중이면 같은 `thread_id`로 피드백을 넘깁니다.

```bash
python main.py "피드백 내용" --thread-id "<위에서 쓴 thread_id>"
```

## 주요 파일

| 파일 | 역할 |
|------|------|
| `main.py` | `SoloMateState`, 노드/도구 정의, `graph` / `graph_llm`, CLI (`run_llm_cli`) |
| `solomate_memory.db` | 실행 시 생성되는 SQLite 체크포인트(스레드별 상태) |

## 참고

규칙 기반 그래프는 `from main import graph` 로, LLM 그래프는 `graph_llm` 으로 가져와 다른 스크립트·노트북에서 확장할 수 있습니다.
