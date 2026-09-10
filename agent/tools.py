"""
agent/tools.py
BookMind Book Agent 도구 5종

이 파일은 LangGraph 에이전트가 호출할 수 있는 5개의 도구(@tool)를 정의한다.
각 도구는 rag/retriever.py의 BookMindRetriever를 통해 검색을 수행하거나,
사용자의 독서 기록(data/user/reading_records.json)을 읽고 쓴다.

Tool 1: recommend_books(query)
    - 감성/상황/주제 기반 책 추천 (예: "위로받고 싶어", "철학 책 추천해줘")
    - 사용자가 좋아한 책, 취향 메모를 쿼리에 자동으로 덧붙여 검색 정확도를 높임
    - MultiQuery 기반 general 검색 수행 후, 정확한 서지정보(가격/저자 등)를
      보장하기 위해 검색된 청크를 같은 책의 intro_info 청크로 교체
    - 추천 결과 중 이미 읽은 책은 "⚠️ 이미 읽은 책"으로 표시만 하고 제외하지는 않음
      (Tool 3이 저장한 기록을 조회, 이 도구는 저장하지 않음)

Tool 2: get_book_detail(title)
    - 특정 책의 저자/출판사/가격/쪽수/평점 등 서지정보 조회
    - info 의도로 검색해 가장 유사한 intro_info 청크 하나를 반환
    - category_paths에서 실제 세부 장르(마지막 경로)를 별도로 추출

Tool 3: save_reading_record(title, status, rating, memo)
    - 독서 상태(읽음/읽는중/읽고싶음), 평점, 메모를 저장
    - 제목이 이미 기록에 있으면 업데이트, 없으면 새로 추가

Tool 4: get_user_profile()
    - 읽음/읽는중/읽고싶음 권수, 평균 평점, 최근 읽은 책 5권 요약 반환
    - 기록이 없으면 안내 메시지 반환

Tool 5: search_user_memos(query)
    - 저장된 메모/독후감을 키워드로 검색
    - 매칭되는 게 없으면 가장 최근 메모 3건을 대신 반환 (fallback)

공통 유틸:
    - _embed: 텍스트를 OpenAI 임베딩 후 L2 정규화 (retriever와 동일한 방식)
    - _get_intro_info: parent_doc_id로 해당 책의 intro_info 청크를 조회
    - _load_records / _save_records: 사용자 독서 기록 JSON 읽기/쓰기
"""

import json
import os
import sys
import numpy as np
import faiss
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_DATA_DIR = os.path.join(BASE_DIR, "data", "user")
RECORDS_PATH  = os.path.join(USER_DATA_DIR, "reading_records.json")

sys.path.insert(0, BASE_DIR)
from rag.retriever import BookMindRetriever

_retriever = BookMindRetriever()
_client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

os.makedirs(USER_DATA_DIR, exist_ok=True)


# =========================================================
# 유틸: 임베딩 생성
# =========================================================
def _embed(text: str) -> np.ndarray:
    response = _client.embeddings.create(
        model="text-embedding-3-small",
        input=[text[:8000]]
    )
    vec = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


# =========================================================
# 유틸: 독서 기록 로드
# =========================================================
def _load_records() -> list:
    if not os.path.exists(RECORDS_PATH):
        return []
    with open(RECORDS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_records(records: list):
    with open(RECORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# =========================================================
# 유틸: parent_doc_id로 intro_info 청크 가져오기
# =========================================================
def _get_intro_info(parent_doc_id: str) -> dict:
    """검색된 청크의 parent_doc_id로 intro_info 청크 반환"""
    for chunk in _retriever.gold_chunks:
        if (chunk.get("parent_doc_id") == parent_doc_id
                and chunk.get("chunk_type") == "intro_info"):
            return chunk
    return None


# =========================================================
# Tool 1: 책 추천 (풀 RAG + 사용자 프로필 자동 참고)
# =========================================================
@tool
def recommend_books(query: str) -> str:
    """
    사용자 질문을 기반으로 관련 책을 추천합니다.
    사용자의 독서 기록과 취향을 자동으로 참고합니다.
    감성/상황/주제 기반 추천에 사용하세요.
    예: "위로받고 싶어", "주식 공부하고 싶어", "철학 책 추천해줘"
    """
    # 사용자 독서 기록 자동 로드
    records      = _load_records()
    read_titles  = [r["title"] for r in records if r["status"] == "읽음"]
    liked_titles = [r["title"] for r in records if r.get("rating") and r["rating"] >= 4]
    memo_texts   = [r["memo"] for r in records if r.get("memo")]

    # 검색 쿼리에 취향 반영
    profile_context = ""
    if liked_titles:
        profile_context += f" 사용자가 좋아한 책: {', '.join(liked_titles[:3])}"
    if memo_texts:
        profile_context += f" 취향 메모: {' '.join(memo_texts[:2])[:100]}"

    enriched_query = f"{query}{profile_context}".strip()

    print(f"\n[recommend_books] 원본 쿼리: {query}")
    print(f"[recommend_books] 보강된 쿼리: {enriched_query[:100]}")

    # MultiQuery 경로(top_k=5 + original_query 지정)에서는 재작성된 쿼리들만
    # 임베딩해 검색하므로, 여기서 별도로 enriched_query를 임베딩할 필요가 없다.
    chunks, scores  = _retriever.search(
        intent="general",
        top_k=5,
        use_multi_query=True,
        original_query=query,
    )

    if not chunks:
        return "관련 책을 찾지 못했습니다."

    # 검색된 청크 상세 로그
    print(f"[recommend_books] 검색 결과 {len(chunks)}개 (intro_info 교체 전):")
    for i, (c, s) in enumerate(zip(chunks, scores), 1):
        print(f"  [{i}] {c.get('title','')} | {c.get('chunk_type','')} | score: {round(s,4)}")
        print(f"       내용: {c.get('content','')[:80].replace(chr(10),' ')}")

    # ── intro_info 청크로 교체 (정확한 가격/날짜/저자 보장)
    enriched_chunks = []
    seen_pids = set()
    for c, s in zip(chunks, scores):
        pid = c.get("parent_doc_id", "")
        if pid in seen_pids:
            continue
        seen_pids.add(pid)

        intro = _get_intro_info(pid)
        if intro:
            enriched_chunks.append((intro, s))
            if c.get("chunk_type") != "intro_info":
                print(f"  → '{c.get('title','')}' {c.get('chunk_type','')} → intro_info 교체")
        else:
            enriched_chunks.append((c, s))

    print(f"[recommend_books] 최종 {len(enriched_chunks)}개 (intro_info 교체 후)")

    results = []

    # 사용자 프로필 컨텍스트 추가
    if records:
        profile_info = "[사용자 독서 프로필]\n"
        if read_titles:
            profile_info += f"- 읽은 책: {', '.join(read_titles[:5])}\n"
        if liked_titles:
            profile_info += f"- 좋아한 책(4점+): {', '.join(liked_titles[:3])}\n"
        if memo_texts:
            profile_info += f"- 취향 메모: {memo_texts[0][:100]}\n"
        results.append(profile_info)

    # 이미 읽은 책 제외 표시
    for c, s in enriched_chunks:
        meta    = c.get("metadata", {})
        title   = c.get("title", "")
        already = "⚠️ 이미 읽은 책" if title in read_titles else ""
        results.append(
            f"📚 {title} {already}\n"
            f"   장르: {meta.get('genre', '')} | 저자: {meta.get('author', '')}\n"
            f"   유사도: {round(s, 3)}\n"
            f"   내용: {c.get('content', '')[:300]}"
        )

    return "\n\n".join(results)


# =========================================================
# Tool 2: 특정 책 서지정보 조회
# =========================================================
@tool
def get_book_detail(title: str) -> str:
    """
    특정 책의 상세 정보(가격, 저자, 출판사, 장르, 쪽수, 평점 등)를 조회합니다.
    사용자가 특정 책 이름을 언급하며 정보를 물어볼 때 사용하세요.
    예: "주식투자를 잘한다는 것 가격이 얼마야?", "이 책 저자가 누구야?"
    """
    query_embedding = _embed(title)
    chunks, scores  = _retriever.search(
        query_embedding,
        intent="info",
        top_k=3,
    )

    if not chunks:
        return f"'{title}' 책 정보를 찾지 못했습니다."

    # 가장 유사한 intro_info 청크
    best = chunks[0]
    meta = best.get("metadata", {})

    # 장르 표현: category_paths 우선, genre는 보조
    category_paths = meta.get("category_paths", "")
    genre_raw      = meta.get("genre", "")

    # category_paths에서 실제 장르 추출 (마지막 항목)
    if category_paths:
        parts        = [p.strip() for p in category_paths.replace("|", ">").split(">")]
        actual_genre = parts[-1] if parts else genre_raw
    else:
        actual_genre = genre_raw

    info = (
        f"📖 **{best.get('title', '')}**\n"
        f"저자: {meta.get('author', '정보 없음')}\n"
        f"출판사: {meta.get('publisher', '정보 없음')}\n"
        f"장르(실제): {actual_genre}\n"
        f"장르(분류): {genre_raw}\n"
        f"카테고리 경로: {category_paths}\n"
        f"발행일: {meta.get('release_date', '정보 없음')}\n"
        f"가격: {int(meta.get('price', 0))}원\n"
        f"쪽수: {meta.get('pages', '정보 없음')}쪽\n"
        f"평점: {meta.get('rating_value', '정보 없음')}\n"
        f"키워드: {meta.get('keywords', '정보 없음')}\n"
        f"수상/추천: {meta.get('awards', '없음')}"
    )
    return info


# =========================================================
# Tool 3: 독서 기록/독후감 저장
# =========================================================
@tool
def save_reading_record(
    title: str,
    status: str,
    rating: Optional[int] = None,
    memo: Optional[str] = None,
) -> str:
    """
    독서 기록과 독후감을 저장합니다.
    사용자가 책을 읽었거나, 읽고 싶다고 하거나, 감상을 남길 때 사용하세요.

    Args:
        title: 책 제목
        status: 독서 상태 ("읽음" | "읽는중" | "읽고싶음")
        rating: 평점 (1~5, 선택)
        memo: 독후감/메모 (선택)
    """
    records = _load_records()

    # 기존 기록 업데이트 or 신규 추가
    existing = next((r for r in records if r["title"] == title), None)
    if existing:
        existing["status"]     = status
        existing["updated_at"] = datetime.now().isoformat()
        if rating is not None:
            existing["rating"] = rating
        if memo:
            existing["memo"] = memo
        msg = f"✅ '{title}' 기록을 업데이트했습니다."
    else:
        record = {
            "title":      title,
            "status":     status,
            "rating":     rating,
            "memo":       memo,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        records.append(record)
        msg = f"✅ '{title}' 독서 기록을 저장했습니다."

    _save_records(records)
    return msg


# =========================================================
# Tool 4: 취향 프로필 조회
# =========================================================
@tool
def get_user_profile() -> str:
    """
    사용자의 독서 취향 프로필을 조회합니다.
    사용자가 자신의 취향이나 읽은 책 목록을 물어볼 때 사용하세요.
    예: "내 취향이 뭐야?", "내가 읽은 책 뭐가 있어?", "내 독서 기록 보여줘"
    """
    records = _load_records()

    if not records:
        return "아직 독서 기록이 없습니다. 책을 읽으셨다면 기록해드릴게요!"

    # 통계 계산
    read     = [r for r in records if r["status"] == "읽음"]
    reading  = [r for r in records if r["status"] == "읽는중"]
    want     = [r for r in records if r["status"] == "읽고싶음"]
    rated    = [r for r in records if r.get("rating") is not None]
    avg_rate = sum(r["rating"] for r in rated) / len(rated) if rated else 0

    # 메모 있는 책들
    memos = [r for r in records if r.get("memo")]

    profile = (
        f"📊 **독서 프로필**\n\n"
        f"✅ 읽은 책: {len(read)}권\n"
        f"📖 읽는 중: {len(reading)}권\n"
        f"🔖 읽고 싶은 책: {len(want)}권\n"
        f"⭐ 평균 평점: {avg_rate:.1f}/5\n\n"
    )

    if read:
        profile += "**읽은 책 목록:**\n"
        for r in read[-5:]:  # 최근 5권
            rating_str = f" ⭐{r['rating']}/5" if r.get("rating") else ""
            profile   += f"  - {r['title']}{rating_str}\n"

    if memos:
        profile += f"\n**독후감 작성한 책:** {len(memos)}권"

    return profile


# =========================================================
# Tool 5: 독후감/메모 검색
# =========================================================
@tool
def search_user_memos(query: str) -> str:
    """
    사용자가 작성한 독후감이나 메모를 검색합니다.
    사용자가 예전에 읽은 책이나 메모를 찾을 때 사용하세요.
    예: "비슷한 책 읽은 적 있어?", "행복에 관해 메모한 거 있어?"
    """
    records = _load_records()
    memos   = [r for r in records if r.get("memo")]

    if not memos:
        return "아직 작성된 독후감이나 메모가 없습니다."

    # 간단한 키워드 매칭 검색
    query_lower = query.lower()
    matched = []
    for r in memos:
        memo_lower  = r["memo"].lower()
        title_lower = r["title"].lower()
        if (query_lower in memo_lower or
            query_lower in title_lower or
            any(word in memo_lower for word in query_lower.split())):
            matched.append(r)

    if not matched:
        # 키워드 매칭 실패 시 전체 반환
        matched = memos[-3:]

    results = []
    for r in matched[:3]:
        results.append(
            f"📝 **{r['title']}**\n"
            f"   상태: {r['status']} | 평점: {r.get('rating', '없음')}/5\n"
            f"   메모: {r['memo'][:200]}"
        )

    return "\n\n".join(results) if results else "관련 메모를 찾지 못했습니다."


# =========================================================
# Tool 목록
# =========================================================
TOOLS = [
    recommend_books,
    get_book_detail,
    save_reading_record,
    get_user_profile,
    search_user_memos,
]