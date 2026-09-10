"""
app.py
BookMind Streamlit UI

기능:
    - 챗봇 대화 (Book Agent + Summary Agent, agent/graph.py의 invoke() 호출)
    - 스트리밍 답변 (한 글자씩 순차 표시)
    - TTS (소리내서 읽기, OpenAI tts-1 모델)
    - 사이드바: 책 검색, 독후감 작성/목록, 독서 통계
    - 채팅: 책 상세 보기 expander

실행: streamlit run streamlit/app.py

주요 처리 내용:
    - data/faiss/chunks.json을 캐시(@st.cache_resource)하여 앱 실행 중
      반복 로드 방지
    - 책 상세 정보는 intro_info(서지정보) 즉시 표시 + intro_desc/intro_bio/
      review를 탭으로 분리해 표시
    - AI 답변에 recommend_books/get_book_detail이 쓰였을 경우, 정규식으로
      답변 텍스트에서 책 제목(《》, "", ** 등 패턴)을 추출해 해당 책의
      상세정보 expander를 자동으로 함께 표시
    - 독후감/메모는 agent/tools.py의 _load_records, _save_records를
      공유하여 사이드바와 Tool 3(save_reading_record)이 같은 데이터에 접근
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from agent.graph import invoke_stream
from agent.tools import _load_records, _save_records

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_DATA_DIR = os.path.join(BASE_DIR, "data", "user")
FAISS_CHUNKS  = os.path.join(BASE_DIR, "data", "faiss", "chunks.json")
os.makedirs(USER_DATA_DIR, exist_ok=True)

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="BookMind",
    page_icon="🔖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #F8F9FA; }
    .book-card {
        background: white; border-radius: 10px;
        padding: 12px; margin: 6px 0;
        border-left: 4px solid #3498DB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .tool-badge {
        background: #EBF5FB; color: #2980B9;
        padding: 2px 8px; border-radius: 10px;
        font-size: 0.75rem;
    }
    .review-item {
        background: #FAFAFA; border-radius: 8px;
        padding: 8px; margin: 4px 0;
        border-left: 3px solid #E8DAEF;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 유틸: 청크 검색
# =========================================================
@st.cache_resource
def load_chunks():
    if not os.path.exists(FAISS_CHUNKS):
        return []
    with open(FAISS_CHUNKS, encoding="utf-8") as f:
        return json.load(f)

def search_book_chunks(title: str, chunks: list):
    """제목으로 청크 검색"""
    title_lower = title.lower()
    matched = [c for c in chunks if title_lower in c.get("title", "").lower()]
    by_type = {}
    for c in matched:
        ct = c.get("chunk_type", "")
        if ct not in by_type:
            by_type[ct] = c
    return by_type


# =========================================================
# 세션 초기화
# =========================================================
if "messages"     not in st.session_state: st.session_state.messages     = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "tts_enabled"  not in st.session_state: st.session_state.tts_enabled  = False


# =========================================================
# TTS
# =========================================================
def text_to_speech(text: str):
    try:
        response    = client.audio.speech.create(model="tts-1", voice="nova", input=text[:500])
        audio_bytes = response.content
        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    except Exception as e:
        st.warning(f"TTS 오류: {e}")


# =========================================================
# 책 상세 정보 표시 함수
# =========================================================
def show_book_detail(title: str, chunks: list, use_tabs: bool = True):
    by_type = search_book_chunks(title, chunks)
    if not by_type:
        st.warning(f"'{title}' 정보를 찾을 수 없어요.")
        return

    # 서지 정보 (항상 표시)
    if "intro_info" in by_type:
        chunk = by_type["intro_info"]
        meta  = chunk.get("metadata", {})
        col1, col2 = st.columns([1, 2])
        with col1:
            img = meta.get("image", "")
            if img:
                st.image(img, width=120)
            else:
                st.markdown("📚")
        with col2:
            st.markdown(f"**{chunk.get('title', '')}**")
            st.caption(f"저자: {meta.get('author', '')} | 출판사: {meta.get('publisher', '')}")
            st.caption(f"발행일: {meta.get('release_date', '')} | 가격: {int(meta.get('price', 0))}원")
            st.caption(f"쪽수: {meta.get('pages', '')}쪽 | 평점: ⭐{meta.get('rating_value', '')}")
            category = meta.get("category_paths", "")
            if category:
                parts = [p.strip() for p in category.replace("|", ">").split(">")]
                st.caption(f"장르: {parts[-1] if parts else ''}")

    # 탭으로 표시 (expander 중첩 방지)
    # intro_bio 청크 합치기
    bio_chunks  = [c for c in chunks
                   if title.lower() in c.get("title", "").lower()
                   and c.get("chunk_type") == "intro_bio"]
    bio_content = "\n\n".join(c.get("content", "") for c in bio_chunks) if bio_chunks else ""

    review_chunks = [c for c in chunks
                     if title.lower() in c.get("title", "").lower()
                     and c.get("chunk_type") == "review"]

    tab_labels = []
    if "intro_desc" in by_type: tab_labels.append("📖 책소개")
    if bio_content:             tab_labels.append("👤 작가 소개")
    if review_chunks:           tab_labels.append(f"💬 리뷰({len(review_chunks)})")

    if tab_labels:
        tabs = st.tabs(tab_labels)
        idx  = 0
        if "intro_desc" in by_type:
            with tabs[idx]:
                st.write(by_type["intro_desc"].get("content", "")[:600])
            idx += 1
        if bio_content:
            with tabs[idx]:
                st.write(bio_content[:500])
            idx += 1
        if review_chunks:
            with tabs[idx]:
                for rc in review_chunks[:3]:
                    content = rc.get("content", "")
                    lines   = [l for l in content.split("\n") if l.strip().startswith("-")]
                    for line in lines[:5]:
                        st.markdown(f'<div class="review-item">{line[1:].strip()}</div>',
                                    unsafe_allow_html=True)


# =========================================================
# 사이드바
# =========================================================
chunks = load_chunks()

with st.sidebar:
    st.markdown("## 🔖 BookMind")
    st.markdown("---")

    # TTS 토글
    st.session_state.tts_enabled = st.toggle(
        "🔊 소리내서 읽기 (TTS)",
        value=st.session_state.tts_enabled
    )
    st.markdown("---")

    # ── 책 검색
    st.markdown("### 🔍 책 검색")
    search_query = st.text_input("책 제목 검색", placeholder="예: 행복의 기원")
    if search_query:
        show_book_detail(search_query, chunks)

    st.markdown("---")

    # ── 독후감 작성
    st.markdown("### 📝 독후감 / 메모")
    with st.form("memo_form", clear_on_submit=True):
        memo_title  = st.text_input("책 제목")
        memo_status = st.selectbox("독서 상태", ["읽음", "읽는중", "읽고싶음"])
        memo_rating = st.slider("평점", 1, 5, 3)
        memo_text   = st.text_area("독후감 / 메모", height=80)
        submitted   = st.form_submit_button("💾 저장")

        if submitted and memo_title:
            from datetime import datetime
            records  = _load_records()
            existing = next((r for r in records if r["title"] == memo_title), None)
            if existing:
                existing.update({"status": memo_status, "rating": memo_rating, "memo": memo_text,
                                 "updated_at": datetime.now().isoformat()})
            else:
                records.append({"title": memo_title, "status": memo_status,
                                 "rating": memo_rating, "memo": memo_text,
                                 "created_at": datetime.now().isoformat(),
                                 "updated_at": datetime.now().isoformat()})
            _save_records(records)
            st.success(f"✅ '{memo_title}' 저장 완료!")

    st.markdown("---")

    # ── 저장된 독후감 목록
    st.markdown("### 📂 내 독서 기록")
    records = _load_records()
    if records:
        status_filter = st.selectbox("필터", ["전체", "읽음", "읽는중", "읽고싶음"])
        filtered = records if status_filter == "전체" else [r for r in records if r["status"] == status_filter]

        for r in reversed(filtered[-10:]):
            rating_str = f" ⭐{r['rating']}" if r.get("rating") else ""
            status_icon = {"읽음": "✅", "읽는중": "📖", "읽고싶음": "🔖"}.get(r["status"], "")
            with st.expander(f"{status_icon} {r['title']}{rating_str}"):
                st.caption(f"상태: {r['status']}")
                if r.get("memo"):
                    st.write(r["memo"])
                else:
                    st.caption("메모 없음")
    else:
        st.caption("아직 독서 기록이 없어요.")

    st.markdown("---")

    # ── 독서 통계
    st.markdown("### 📊 독서 통계")
    records = _load_records()
    if records:
        read    = len([r for r in records if r["status"] == "읽음"])
        reading = len([r for r in records if r["status"] == "읽는중"])
        want    = len([r for r in records if r["status"] == "읽고싶음"])
        rated   = [r for r in records if r.get("rating")]
        avg     = sum(r["rating"] for r in rated) / len(rated) if rated else 0
        col1, col2 = st.columns(2)
        col1.metric("읽은 책", f"{read}권")
        col2.metric("읽는 중", f"{reading}권")
        col1.metric("위시리스트", f"{want}권")
        col2.metric("평균 평점", f"{avg:.1f}⭐")
    else:
        st.caption("아직 기록이 없어요.")

    st.markdown("---")
    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages     = []
        st.session_state.chat_history = []
        st.rerun()


# =========================================================
# 메인 채팅 영역
# =========================================================
st.markdown("# 🔖 BookMind")
st.caption("나만의 취향을 기억하는 개인화 독서 추천 AI")

# 이전 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("tool_used"):
            st.markdown(f'<span class="tool-badge">🔧 {msg["tool_used"]}</span>',
                        unsafe_allow_html=True)
        # 채팅 내 책 상세 expander
        if msg.get("book_title"):
            with st.expander(f"📖 '{msg['book_title']}' 상세 보기"):
                show_book_detail(msg["book_title"], chunks)


# =========================================================
# 사용자 입력
# =========================================================
if prompt := st.chat_input("읽고 싶은 책이나 기분을 말씀해주세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("생각하는 중..."):
            stream, stream_result = invoke_stream(prompt, chat_history=st.session_state.chat_history)

        # 실제 토큰 스트리밍 출력
        answer    = st.write_stream(stream)
        tool_used = stream_result.tool_used

        # Tool 뱃지
        if tool_used:
            st.markdown(f'<span class="tool-badge">🔧 {tool_used}</span>',
                        unsafe_allow_html=True)

        # 책 추천이면 상세 보기 expander 자동 표시
        book_title = None
        if tool_used in ("recommend_books", "get_book_detail"):
            import re

            # 다양한 패턴으로 제목 추출
            patterns = [
                r'《(.+?)》',                      # 《오만과 편견》
                r'["""](.+?)["""]',               # "오만과 편견"
                r'\*{1,3}["""]?(.+?)["""]?\*{1,3}', # **오만과 편견**
            ]
            book_title = None
            for pattern in patterns:
                matches = re.findall(pattern, answer)
                if matches:
                    # 특수문자 제거
                    raw = matches[0]
                    raw = re.sub(r'[*"""\'"《》「」]', '', raw).strip()
                    if raw and len(raw) > 1:
                        book_title = raw
                        break

            if book_title:
                with st.expander(f"📖 '{book_title}' 상세 보기"):
                    show_book_detail(book_title, chunks)

        # TTS
        if st.session_state.tts_enabled and answer:
            text_to_speech(answer)

    # 히스토리 업데이트
    st.session_state.chat_history.append({"role": "user",      "content": prompt})
    st.session_state.chat_history.append({"role": "assistant",  "content": answer})
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    answer,
        "tool_used":  tool_used,
        "book_title": book_title,
    })