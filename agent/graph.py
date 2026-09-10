"""
agent/graph.py
BookMind LangGraph - Book Agent + Summary Agent

구조:
    Book Agent   : Tool 선택 담당 (temperature=0, 결정적 판단을 위해 낮게 설정)
    Tools Node   : Tool 실행
    Summary Agent: Tool 결과 정리 + 답변 생성 (temperature=0.3, 자연스러운 문장을 위해 약간 높게 설정)

흐름:
    book_agent → (tool 있음) → tools → summary_agent → END
    book_agent → (tool 없음) → END

프롬프트 설계 원칙:
    - Book Agent: 도구 선택 기준을 명시하고, 시사/정치/경제 등 실시간 정보가
      필요한 질문에는 사실을 단정하지 않고 책 추천으로 자연스럽게 유도
    - Summary Agent: 장르 표기, 저자 표기, 가격/날짜 형식 등 답변 스타일을
      규칙화. Tool 결과가 없을 때는 절대 자체 지식으로 보완하지 않고
      정해진 안내 문구로만 응답 (hallucination 방지)

외부 인터페이스:
    invoke(user_message, chat_history) → {"answer": str, "tool_used": str}
    - 최근 대화 10개까지 히스토리로 반영
    - 마지막 AIMessage를 answer로, 실제 호출된 tool 이름을 tool_used로 반환
"""

import os
from datetime import datetime
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent.tools import TOOLS

load_dotenv()


# =========================================================
# 프롬프트
# =========================================================
def get_book_agent_prompt() -> str:
    today = datetime.now().strftime("%Y년 %m월 %d일")
    return f"""당신은 개인화 도서 추천 AI, BookMind입니다.
따뜻하고 친근한 말투로 사용자와 자유롭게 대화하세요.
데이터로 가지고 있는 것만 사용해서 답변을 하세요.

대화 방식:
- 대화 중 자연스럽게 책으로 연결해주세요.
- 억지로 책을 끼워넣지 말고, 자연스러운 흐름에서만 연결하세요.

사용 가능한 도구:
- recommend_books   : 주제/감성/상황 기반 책 추천 (풀 RAG)
- get_book_detail   : 특정 책의 가격, 저자, 출판사 등 서지정보 조회
- save_reading_record: 독서 기록과 독후감 저장
- get_user_profile  : 사용자 독서 취향 프로필 조회
- search_user_memos : 사용자가 작성한 독후감/메모 검색

도구 선택 기준:
- "책 추천해줘", "~한 책 있어?" → recommend_books
- "~책 가격", "~책 저자", "~책 장르" → get_book_detail
- "읽었어", "독후감", "메모 저장" → save_reading_record
- "내 취향", "내가 읽은 책" → get_user_profile
- "예전에 읽은 책", "비슷한 메모" → search_user_memos
- 일상 대화, 감정 표현 → 도구 없이 직접 답변

중요 — 책 제목 직접 언급 금지:
- 도구를 호출하지 않고는 어떤 책 제목도 답변에 직접 언급하지 마세요.
  (당신이 알고 있는 유명한 책이라도 마찬가지입니다.)
- 대화 흐름상 책을 추천/언급하고 싶어지면, 지식으로 답하지 말고
  반드시 recommend_books 또는 get_book_detail을 호출하세요.

중요 — 사실 단정 금지:
- 시사, 경제, 정치, 부동산, 주식 등 실시간 정보가 필요한 질문에
  사실인 것처럼 단정해서 답변하지 마세요.
- 이런 질문이 오면 반드시 아래 형식으로 답하세요:
  "저는 도서 추천 AI라 정확한 정보를 드리기 어렵습니다.
   다만 이 주제와 관련된 책을 찾아드릴 수 있어요!"
- 그 후 자연스럽게 관련 책 추천으로 연결하세요.

현재 날짜는 {today}입니다."""


SUMMARY_AGENT_PROMPT = """당신은 BookMind의 Summary Agent입니다.
Book Agent가 도구를 실행한 결과를 받아 사용자에게 친근하고 자연스럽게 전달하세요.

답변 규칙:
- 데이터에 들어있는 내용인지 확인하세요.
- 도구 결과의 핵심만 추려서 자연스러운 문장으로 정리하세요.
- 책을 추천할 때는 제목을 먼저 말하고, 추천 이유를 설명하세요.
- 장르는 카테고리 경로의 마지막 항목을 사용하세요. "시에세이" 금지.
  예: "한국에세이" → "한국 에세이"
- 저자는 "지음" 앞의 이름만 사용하세요. 번역가는 언급하지 마세요.
- 가격/날짜는 "16,650원", "2026년 5월 15일" 형식으로 표현하세요.
- 따뜻하고 친근한 말투를 유지하세요.
- 이전 대화 맥락을 반드시 참고하세요.

정보 조회 질문 답변 규칙 (중요):
- 가격/저자/출판사/발행일/쪽수/평점 등 단순 정보를 묻는 질문은 반드시 간결하게 답하세요.
- 서두 없이 바로 핵심 정보만 말하세요.
- 예시:
  질문: "일론 머스크 출판사가 어디야?"
  좋은 답변: "'일론 머스크'의 출판사는 21세기북스입니다."
  나쁜 답변: "일론 머스크는 정말 혁신적인 기업가인데요... 출판사는 21세기북스입니다."
- 추가 설명이 필요 없으면 한 문장으로 끝내세요.

중요 — 반드시 Tool 결과에 있는 책만 언급 (최우선 규칙):
- 책 제목, 저자, 가격, 출판사, 발행일, 평점 등 모든 정보는 반드시
  Tool 결과에 실제로 적힌 내용만 사용하세요.
- Tool 결과에 없는 책은 당신이 아무리 유명하고 잘 안다고 해도
  추천/언급하지 마세요. (예: 세계적으로 유명한 고전이라도 Tool 결과에
  없으면 절대 언급 금지)
- Tool이 반환한 책 후보가 질문 의도(장르/주제)와 잘 안 맞더라도,
  다른 책으로 대체하거나 지어내지 말고 Tool 결과 안에서만 고르세요.
- Tool 결과 중 정말 적합한 책이 하나도 없다면, 억지로 추천하지 말고
  "죄송해요, 지금 데이터에서는 딱 맞는 책을 찾지 못했어요"라고
  솔직하게 답하세요.

중요 — Tool 결과가 없거나 찾을 수 없는 경우:
- Tool 결과에 "찾을 수 없어요" 또는 "정보가 없습니다"가 포함된 경우
  절대로 GPT 자체 지식으로 보완하거나 추측해서 답변하지 마세요.
- 반드시 이렇게 안내하세요:
  "해당 책 정보가 데이터베이스에 없습니다.
   서점 베스트셀러 기준으로 수집된 책만 조회 가능해요."
- 저자, 가격, 출판사 등 어떤 정보도 지어내지 마세요."""


# =========================================================
# 상태 정의
# =========================================================
class BookMindState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# =========================================================
# LLM 초기화
# =========================================================
_book_agent_llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_summary_agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
_llm_with_tools    = _book_agent_llm.bind_tools(TOOLS)


# =========================================================
# 노드 1: Book Agent (Tool 선택)
# =========================================================
def book_agent_node(state: BookMindState) -> BookMindState:
    messages = state["messages"]

    # 매 호출마다 최신 날짜 반영
    prompt   = get_book_agent_prompt()
    messages = [SystemMessage(content=prompt)] + [
        m for m in messages if not isinstance(m, SystemMessage)
    ]

    response   = _llm_with_tools.invoke(messages)
    tool_names = [tc["name"] for tc in response.tool_calls] if response.tool_calls else []
    print(f"\n[Book Agent] tool_calls: {tool_names if tool_names else '없음 (직접 답변)'}")

    return {"messages": [response]}


# =========================================================
# 노드 2: Summary Agent (Tool 결과 정리 + 답변 생성)
# =========================================================
def summary_agent_node(state: BookMindState) -> BookMindState:
    messages = state["messages"]

    tool_results = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_results:
        return state

    print(f"[Summary Agent] Tool 결과 {len(tool_results)}개 정리 중...")

    # Tool 결과 상세 로그
    for i, tr in enumerate(tool_results, 1):
        content = tr.content if isinstance(tr.content, str) else str(tr.content)
        preview = content[:300].replace("\n", " ")
        print(f"  [Tool {i}] {preview}{'...' if len(content) > 300 else ''}")

    summary_messages = [SystemMessage(content=SUMMARY_AGENT_PROMPT)] + [
        m for m in messages if not isinstance(m, SystemMessage)
    ]
    response = _summary_agent_llm.invoke(summary_messages)

    print(f"[Summary Agent] 답변 생성 완료")
    return {"messages": [response]}


# =========================================================
# 그래프 빌드
# =========================================================
def build_graph():
    graph = StateGraph(BookMindState)

    graph.add_node("book_agent",    book_agent_node)
    graph.add_node("tools",         ToolNode(TOOLS))
    graph.add_node("summary_agent", summary_agent_node)

    graph.set_entry_point("book_agent")

    # book_agent → tool 있으면 tools, 없으면 END
    graph.add_conditional_edges(
        "book_agent",
        tools_condition,
        {
            "tools": "tools",
            END:     END,
        }
    )

    # tools → summary_agent → END
    graph.add_edge("tools",         "summary_agent")
    graph.add_edge("summary_agent", END)

    return graph.compile()


bookmind_graph = build_graph()


# =========================================================
# 외부 호출 함수 (app.py용)
# =========================================================
def invoke(user_message: str, chat_history: list = None) -> dict:
    """
    user_message : 사용자 입력
    chat_history : [{"role": "user"/"assistant", "content": str}, ...]
    반환         : {"answer": str, "tool_used": str}
    """
    messages = []

    # 최근 10개 대화 이력 추가
    if chat_history:
        for msg in chat_history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))

    result = bookmind_graph.invoke({"messages": messages})

    final_messages = result["messages"]
    answer    = ""
    tool_used = ""

    # 마지막 AI 답변 추출
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage) and msg.content:
            answer = msg.content
            break

    # 사용된 Tool 이름 추출
    for msg in final_messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_used = msg.tool_calls[0]["name"]
            break

    return {"answer": answer, "tool_used": tool_used}


class StreamResult:
    """invoke_stream()의 제너레이터를 다 소비한 뒤 tool_used를 꺼내 쓰기 위한 컨테이너"""
    def __init__(self):
        self.tool_used = ""


def invoke_stream(user_message: str, chat_history: list = None):
    """
    invoke()와 동일한 흐름(Book Agent → Tools → Summary Agent)이되,
    Summary Agent의 최종 답변만 실제 토큰 단위로 스트리밍한다.
    (컴파일된 LangGraph는 노드 단위로만 결과를 반환해 토큰 스트리밍이
     안 되므로, 마지막 노드만 스트리밍하려고 별도 경로로 둔다.)

    반환: (chunk_generator, StreamResult)
        - chunk_generator를 다 소비하면 완성된 답변 텍스트가 됨
        - 소비 후 result.tool_used로 사용된 tool 이름 조회
    """
    messages = []
    if chat_history:
        for msg in chat_history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))

    result = StreamResult()

    book_messages = [SystemMessage(content=get_book_agent_prompt())] + messages
    response = _llm_with_tools.invoke(book_messages)

    if not response.tool_calls:
        def _direct():
            yield response.content
        return _direct(), result

    result.tool_used = response.tool_calls[0]["name"]

    tool_state    = ToolNode(TOOLS).invoke({"messages": messages + [response]})
    tool_messages = tool_state["messages"]

    summary_messages = (
        [SystemMessage(content=SUMMARY_AGENT_PROMPT)] + messages + [response] + tool_messages
    )

    def _stream():
        for chunk in _summary_agent_llm.stream(summary_messages):
            if chunk.content:
                yield chunk.content

    return _stream(), result