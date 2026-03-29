"""
SoloMate AI — LangGraph.

`python main.py` → LLM 그래프: analyze_progression → chatbot(bind_tools) + ToolNode
+ 웹검색(search_song_chords, search_song_mood) + get_human_feedback(interrupt) + SqliteSaver

규칙 기반 그래프 `graph`는 모듈에서 그대로 import 가능(스크립트 진입점은 LLM만 사용).

interrupt 재개 예::

    from langgraph.types import Command
    graph_llm.invoke(
        Command(resume={"feedback": "이대로 좋아"}),
        config={"configurable": {"thread_id": "..."}},
    )
"""

from __future__ import annotations

import sqlite3

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from typing_extensions import NotRequired

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------


class SoloMateState(MessagesState):
    """대화와 연습 도메인 상태.

    상속 필드 (MessagesState):
        messages — HumanMessage / AIMessage 등 대화 목록 (add_messages 리듀서)

    추가 필드:
        custom_stuff — 예시 MessagesState 확장과 동일한 보조 문자열(여기서는 노래 제목 등)
    """

    custom_stuff: NotRequired[str]
    song_title: NotRequired[str]
    chord_progression: NotRequired[str]
    mood: NotRequired[str]
    analysis_summary: NotRequired[str]
    practice_output: NotRequired[str]


def _last_user_text(state: SoloMateState) -> str:
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return ""


# ---------------------------------------------------------------------------
# 웹 검색 (analyze 및 @tool 도구에서 공통 사용)
# ---------------------------------------------------------------------------


def _ddg_text(query: str, *, max_results: int = 5) -> str:
    """DuckDuckGo 텍스트 검색 결과를 요약 문자열로 반환한다."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "[오류] duckduckgo-search 패키지가 필요합니다."

    try:
        with DDGS(timeout=25) as ddgs:
            rows = ddgs.text(query.strip(), max_results=max_results)
    except Exception as e:
        return f"[웹 검색 실패] {e!s}"

    if not rows:
        return "(검색 결과 없음)"

    parts: list[str] = []
    for i, r in enumerate(rows, 1):
        title = r.get("title", "")
        body = (r.get("body") or "")[:450]
        href = r.get("href", "")
        parts.append(f"{i}. {title}\n   {body}\n   {href}")
    return "\n\n".join(parts)


def fetch_song_chords_web(song_title: str) -> str:
    q = f"{song_title.strip()} 기타 코드 진행"
    return _ddg_text(q)


def fetch_song_mood_web(song_title: str) -> str:
    q = f"{song_title.strip()} 노래 분위기 장르"
    return _ddg_text(q)


# ---------------------------------------------------------------------------
# 규칙 기반 노드
# ---------------------------------------------------------------------------


def analyze_progression(state: SoloMateState) -> dict:
    """사용자 메시지 첫 줄을 노래 제목으로 보고, LLM이 도구로 코드·분위기를 검색하도록 SystemMessage를 messages 앞에 둔다."""
    text = _last_user_text(state).strip()
    title = text.splitlines()[0].strip() if text else "(제목 없음)"

    summary = (
        f"노래 제목: {title}\n"
        "코드 진행·분위기는 `search_song_chords`·`search_song_mood` 도구 호출 결과로만 채운다."
    )

    system = SystemMessage(
        content=(
            "너는 SoloMate AI다. 기타 솔로·보컬 연습을 돕는다.\n"
            "반드시 먼저 `search_song_chords`와 `search_song_mood`를 이 노래 제목으로 호출해 "
            "코드 진행과 분위기를 검색하고, 그 두 도구의 반환 문자열을 근거로 제안해라.\n"
            "두 도구의 검색 결과를 바탕으로 짧고 실천 가능한 연습 팁을 한국어로 제안해라.\n"
            "연습 제안을 마친 뒤 사용자 의견을 받으려면 `get_human_feedback`을 호출한다. "
            "인자 `practice_suggestion`에는 제안 전체를 한 문자열로 넣어라.\n"
            "사용자 피드백(ToolMessage)을 이미 받은 뒤에는 도구 없이 짧게 마무리해도 된다.\n\n"
            f"[노래 제목] {title}\n"
            f"[분석 요약]\n{summary}"
        )
    )

    raw_msgs = list(state.get("messages") or [])

    return {
        "messages": [
            RemoveMessage(id="__remove_all__"),
            system,
            *raw_msgs,
        ],
        "song_title": title,
        "chord_progression": "",
        "mood": "",
        "analysis_summary": summary,
        "custom_stuff": title,
    }


def route_after_analyze(state: SoloMateState) -> str:
    """사용자 문맥에 보컬 키워드가 있으면 보컬 노드, 아니면 기타(애드립) 노드."""
    blob = (_last_user_text(state) or "") + " " + (state.get("song_title") or "")
    vocal_kw = ("보컬", "발성", "호흡", "가창", "허밍", "성대", "믹스보이스")
    if any(k in blob for k in vocal_kw):
        return "vocal_recommend_practice"
    return "guitar_recommend_practice"


def guitar_recommend_practice(state: SoloMateState) -> dict:
    """기타 애드립·솔로 연습 힌트."""
    title = state.get("song_title") or ""
    chords = state.get("chord_progression") or ""
    mood = state.get("mood") or ""

    scale_hint = (
        "다이아토닉/펜타토닉을 1차 후보로 두고, "
        "IV·V 구간에서는 코드톤(루트·3도·5도)과 근접 반음 앱로치를 섞어 보세요."
    )
    phrase_hint = (
        "프레이즈: 상행 3~4음 스케일 런 후 타겟 코드의 루트 또는 3도로 착지(느린 템포로 반복).\n"
        "팁: 코드 체인지 직전 한 박은 이전 화성의 패싱톤으로 정리하면 자연스럽습니다."
    )
    practice = (
        f"[SoloMate 연습 제안 · 기타]\n"
        f"노래 제목: {title}\n"
        f"코드 진행: {chords}\n"
        f"분위기: {mood}\n\n"
        f"스케일·애드립: {scale_hint}\n\n"
        f"{phrase_hint}"
    )
    return {
        "practice_output": practice,
        "messages": [AIMessage(content=practice)],
    }


def vocal_recommend_practice(state: SoloMateState) -> dict:
    """보컬 연습 힌트."""
    title = state.get("song_title") or ""
    chords = state.get("chord_progression") or ""
    mood = state.get("mood") or ""

    breath_tone = (
        "멜로디 한 구절(또는 2~4마디) 단위로 호흡을 미리 나누고, "
        "코드가 바뀌는 박 앞에서는 숨을 모아 두었다가 모음에서 부드럽게 켜 보세요. "
        "긴 음은 흉·두성 비율을 유지한 채 볼륨만 미세 조절합니다."
    )
    melody_hint = (
        "선율: 화음의 3도·5도 방향을 귀로 확인하며 피치를 맞추고, "
        "멜리스마(꾸밈음)는 짧게 끊어 반복 연습한 뒤 문장 단위로 이어 보세요.\n"
        "표현: 약한 구간은 자음만 또렷하게, 모음에서 감정·비브라토를 얹고, "
        "코드 체인지 직전 음은 짧게 정리한 뒤 다음 화성으로 넘어가면 자연스럽습니다."
    )
    practice = (
        f"[SoloMate 연습 제안 · 보컬]\n"
        f"노래 제목: {title}\n"
        f"코드 진행: {chords}\n"
        f"분위기: {mood}\n\n"
        f"호흡·발성: {breath_tone}\n\n"
        f"{melody_hint}"
    )
    return {
        "practice_output": practice,
        "messages": [AIMessage(content=practice)],
    }


def build_solomate_graph():
    graph = StateGraph[SoloMateState, None, SoloMateState, SoloMateState](SoloMateState)
    graph.add_node("analyze_progression", analyze_progression)
    graph.add_node("guitar_recommend_practice", guitar_recommend_practice)
    graph.add_node("vocal_recommend_practice", vocal_recommend_practice)
    graph.add_edge(START, "analyze_progression")
    graph.add_conditional_edges(
        "analyze_progression",
        route_after_analyze,
        {
            "guitar_recommend_practice": "guitar_recommend_practice",
            "vocal_recommend_practice": "vocal_recommend_practice",
        },
    )
    graph.add_edge("guitar_recommend_practice", END)
    graph.add_edge("vocal_recommend_practice", END)
    return graph.compile()


graph = build_solomate_graph()


# ---------------------------------------------------------------------------
# LLM + 도구 + 체크포인트
# ---------------------------------------------------------------------------


@tool
def search_song_chords(song_title: str) -> str:
    """노래 제목으로 기타 코드·코드 진행·TAB·악보 관련 웹 검색 결과를 가져온다."""
    return fetch_song_chords_web(song_title)


@tool
def search_song_mood(song_title: str) -> str:
    """노래 제목으로 분위기·장르·감성·템포 등 음악적 특징 웹 검색 결과를 가져온다."""
    return fetch_song_mood_web(song_title)


@tool
def get_human_feedback(practice_suggestion: str) -> str:
    """
    연습 제안에 대한 사용자 피드백을 받는다.
    사용자가 만족하면 그렇게 말하고, 수정이 필요하면 구체적으로 요청한다.
    """
    payload = interrupt({"practice_suggestion": practice_suggestion})
    if isinstance(payload, dict):
        return str(payload.get("feedback", payload))
    return str(payload)


tools = [search_song_chords, search_song_mood, get_human_feedback]

llm = init_chat_model("openai:gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools=tools)


def chatbot(state: SoloMateState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools=tools)


def build_solomate_llm_graph(checkpointer: SqliteSaver | None = None):
    g = StateGraph[SoloMateState, None, SoloMateState, SoloMateState](SoloMateState)
    g.add_node("analyze_progression", analyze_progression)
    g.add_node("chatbot", chatbot)
    g.add_node("tools", tool_node)

    g.add_edge(START, "analyze_progression")
    g.add_edge("analyze_progression", "chatbot")
    g.add_conditional_edges("chatbot", tools_condition)
    g.add_edge("tools", "chatbot")

    return g.compile(checkpointer=checkpointer)


def _default_checkpointer() -> SqliteSaver:
    conn = sqlite3.connect("solomate_memory.db", check_same_thread=False)
    return SqliteSaver(conn)


graph_llm = build_solomate_llm_graph(checkpointer=_default_checkpointer())


def _print_messages(values: dict | None) -> None:
    if not isinstance(values, dict):
        print(values)
        return
    if ints := values.get("__interrupt__"):
        print("[interrupt]", ints)
    for m in values.get("messages") or []:
        if hasattr(m, "pretty_print"):
            m.pretty_print()
        else:
            print(m)


def run_llm_cli() -> None:
    """LLM 그래프 + interrupt CLI (`python main.py`)."""
    import argparse
    import sys
    import uuid

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description=(
            "SoloMate LLM 그래프. 첫 실행: --title 또는 위치 인자로 노래 제목. "
            "interrupt 대기 중: --resume 또는 위치 인자로 피드백."
        )
    )
    parser.add_argument(
        "--resume",
        "-r",
        default=None,
        metavar="TEXT",
        help='재개 피드백. 예: --resume "ㅇㅋ"',
    )
    parser.add_argument(
        "resume_positional",
        nargs="?",
        default=None,
        metavar="FEEDBACK",
        help="interrupt 대기 중이면 피드백, 아니면 노래 제목(첫 실행)",
    )
    parser.add_argument(
        "--thread-id",
        default="demo-thread-1",
        help="SqliteSaver 스레드 ID.",
    )
    parser.add_argument(
        "--new-thread",
        action="store_true",
        help="새 대화 스레드(UUID).",
    )
    parser.add_argument(
        "--title",
        default="가을 안부",
        help="노래 제목 한 줄 (첫 실행 시 사용자 메시지로 전달)",
    )
    args = parser.parse_args()

    thread_id = str(uuid.uuid4()) if args.new_thread else args.thread_id
    cfg = {"configurable": {"thread_id": thread_id}}

    resume_text = None
    title = args.title.strip()

    if args.resume is not None:
        resume_text = args.resume
    elif args.resume_positional is not None:
        snap0 = graph_llm.get_state(cfg)
        if snap0.interrupts:
            resume_text = args.resume_positional
        else:
            title = args.resume_positional.strip()

    if resume_text is not None:
        snap = graph_llm.get_state(cfg)
        if not snap.interrupts:
            print(
                "[오류] 재개할 interrupt가 없습니다.\n"
                "  • 새로 시도: python main.py --new-thread\n"
                "  • 또는 solomate_memory.db 를 지우고 다시 첫 실행부터 하세요.\n",
                file=sys.stderr,
            )
            sys.exit(1)
        out = graph_llm.invoke(
            Command(resume={"feedback": resume_text}),
            config=cfg,
        )
    else:
        out = graph_llm.invoke(
            {
                "messages": [
                    HumanMessage(title),
                ],
            },
            config=cfg,
        )

    _print_messages(out)

    snap = graph_llm.get_state(cfg)
    if snap.interrupts or snap.next:
        print(
            "\n[interrupt 대기] 예:\n"
            f'  python main.py --resume "ㅇㅋ" --thread-id {thread_id!r}\n'
            f"  python main.py ㅇㅋ --thread-id {thread_id!r}\n",
            flush=True,
        )


if __name__ == "__main__":
    run_llm_cli()
