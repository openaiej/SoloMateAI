"""
SoloMate AI — LangGraph 기초 그래프.
코드 진행 입력 → 분석 → 연습용 제안(스케일·프레이즈 힌트) 흐름의 최소 뼈대.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from typing_extensions import NotRequired


class SoloMateState(MessagesState):
    """대화(messages)와 기타 연습용 도메인 상태."""

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


def analyze_progression(state: SoloMateState) -> dict:
    """사용자 메시지에서 코드 진행·분위기를 추출하고 분석 요약을 만든다."""
    text = _last_user_text(state).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chords = lines[0] if lines else "(입력 없음)"
    mood = " / ".join(lines[1:]) if len(lines) > 1 else "미지정"

    summary = (
        f"입력된 코드 진행(추정): {chords}\n"
        f"분위기/메모: {mood}\n"
        "다음 노드에서 스케일·애드립·연습 프레이즈 방향을 제안합니다."
    )
    return {
        "chord_progression": chords,
        "mood": mood,
        "analysis_summary": summary,
    }


def recommend_practice(state: SoloMateState) -> dict:
    """분석 결과를 바탕으로 연습용 힌트와 AI 메시지를 생성한다."""
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
        f"[SoloMate 연습 제안]\n"
        f"코드 진행: {chords}\n"
        f"분위기: {mood}\n\n"
        f"스케일·애드립: {scale_hint}\n\n"
        f"{phrase_hint}"
    )
    return {
        "practice_output": practice,
        "messages": [AIMessage(content=practice)],
    }


def build_solomate_graph():
    graph = StateGraph[SoloMateState, None, SoloMateState, SoloMateState](SoloMateState)
    graph.add_node("analyze_progression", analyze_progression)
    graph.add_node("recommend_practice", recommend_practice)
    graph.add_edge(START, "analyze_progression")
    graph.add_edge("analyze_progression", "recommend_practice")
    graph.add_edge("recommend_practice", END)
    return graph.compile()


graph = build_solomate_graph()


if __name__ == "__main__":
    demo = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    "C | Am | F | G\n"
                    "발라드, 첫 솔로라 부담 없이 가고 싶어."
                )
            ],
        }
    )
    print(demo.get("analysis_summary", ""))
    print("---")
    print(demo.get("practice_output", ""))
