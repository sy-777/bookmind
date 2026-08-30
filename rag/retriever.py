"""
rag/retriever.py
BookMind FAISS 기반 리트리버 (MultiQuery + Reranker 포함)

기능:
    - FAISS 인덱스 + 타입별 서브 인덱스 로드
    - 질문 타입별 검색 (info / general)
    - Title 필터링: info 질문에서 책 제목 추출 → 직접 필터링
    - MultiQuery: 질문 → 3개 관점 쿼리 생성 → 각각 검색 → 결과 합치기
    - Reranker: FAISS 후보 → CrossEncoder 재정렬 (2-stage retrieval)
    - 중복 제거 (parent_doc_id 기준)
"""

import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from rag.reranker import BookMindReranker

load_dotenv()

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_DIR   = os.path.join(BASE_DIR, "data", "faiss")
INDEX_PATH  = os.path.join(FAISS_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(FAISS_DIR, "chunks.json")

INFO_KEYWORDS = [
    "저자", "작가", "가격", "얼마", "쪽수", "페이지",
    "출판사", "발행일", "출판일", "평점", "별점", "누가 썼",
    "언제 나왔", "몇 페이지", "몇 쪽", "카테고리", "키워드", "장르",
]

TYPE_QUOTA = {
    "intro_desc": 2,
    "toc":        1,
    "intro_info": 1,
    "extra":      1,
}

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class BookMindRetriever:
    def __init__(self, use_reranker: bool = True):
        print("📚 FAISS 인덱스 로드 중...")
        self.index            = faiss.read_index(INDEX_PATH)
        self.gold_chunks      = self._load_chunks()
        self.type_sub_indices = self._build_sub_indices()
        self.client           = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.reranker         = BookMindReranker() if use_reranker else None
        print(f"✅ {self.index.ntotal}개 벡터 로드 완료!")

    def _load_chunks(self):
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            return json.load(f)

    def _build_sub_indices(self):
        type_indices = {}
        for i, c in enumerate(self.gold_chunks):
            ct = c["chunk_type"]
            type_indices.setdefault(ct, []).append(i)

        dim = self.index.d
        type_sub_indices = {}
        for ct, indices in type_indices.items():
            vectors = np.array(
                [self.index.reconstruct(i) for i in indices],
                dtype=np.float32
            )
            faiss.normalize_L2(vectors)
            sub = faiss.IndexFlatIP(dim)
            sub.add(vectors)
            type_sub_indices[ct] = (sub, indices)

        return type_sub_indices

    def classify_intent(self, query: str) -> str:
        for kw in INFO_KEYWORDS:
            if kw in query:
                return "info"
        return "general"

    # =========================================================
    # Reranker 호출 (reranker.py 위임)
    # =========================================================
    def _rerank(self, query: str, chunks: list, scores: list, top_k: int) -> tuple:
        if self.reranker:
            return self.reranker.rerank(query, chunks, scores, top_k)
        return chunks[:top_k], scores[:top_k]

    # =========================================================
    # MultiQuery: 쿼리 3개 생성
    # =========================================================
    def generate_multi_queries(self, query: str, n: int = 3) -> list:
        """
        원래 질문에서 n개의 다른 관점 쿼리 생성
        예: "행복한 책 추천해줘"
            → ["행복과 긍정 심리학 관련 도서",
               "마음의 위안을 주는 에세이",
               "삶의 의미와 행복을 다룬 책"]
        """
        if not query or not query.strip():
            return []

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"사용자 질문을 {n}가지 다른 관점으로 재작성하세요. "
                            "각 쿼리는 원래 의도를 유지하되 다른 키워드와 표현을 사용하세요. "
                            "줄바꿈으로 구분하여 쿼리만 출력하세요."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=200,
            )
            queries = [q.strip() for q in
                       response.choices[0].message.content.strip().split("\n")
                       if q.strip()]
            return ([query] + queries)[:n + 1]
        except Exception:
            return [query]

    def embed_query(self, query: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[query[:8000]]
        )
        vec = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        return vec

    # =========================================================
    # 제목 추출 (info 쿼리용)
    # =========================================================
    def extract_title_from_query(self, query: str) -> str:
        """
        "일론 머스크 책 저자가 누구야?" → "일론 머스크"
        "오만과 편견 가격이 얼마야?" → "오만과 편견"
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "사용자 질문에서 책 제목만 추출하세요. "
                            "책 제목이 없으면 빈 문자열을 반환하세요. "
                            "책 제목만 출력하고 다른 텍스트는 출력하지 마세요."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0,
                max_tokens=50,
            )
            title = response.choices[0].message.content.strip()
            title = title.strip('"\'「」《》')
            return title
        except Exception:
            return ""

    def _search_by_title(self, title: str, top_k: int = 5):
        """제목으로 직접 필터링해서 intro_info 청크 반환"""
        title_lower    = title.lower()
        matched_chunks = []
        matched_scores = []

        for chunk in self.gold_chunks:
            chunk_title = chunk.get("title", "").lower()
            if (chunk.get("chunk_type") == "intro_info" and
                    (title_lower in chunk_title or chunk_title in title_lower)):
                matched_chunks.append(chunk)
                matched_scores.append(1.0)

        print(f"   [Title Filter] '{title}' → {len(matched_chunks)}개 매칭")
        return matched_chunks[:top_k], matched_scores[:top_k]

    # =========================================================
    # 단일 쿼리 검색
    # =========================================================
    def _search_single(self, query_embedding: np.ndarray, intent: str,
                       top_k: int = 10, title_hint: str = ""):
        seen          = set()
        unique_chunks = []
        unique_scores = []

        if intent == "info":
            if title_hint:
                title_chunks, title_scores = self._search_by_title(title_hint, top_k)
                if title_chunks:
                    return title_chunks, title_scores

            sub, indices = self.type_sub_indices.get("intro_info", (None, []))
            if sub is not None:
                sc, ix = sub.search(query_embedding, min(20, len(indices)))
                for j, score in zip(ix[0], sc[0]):
                    if j == -1:
                        continue
                    idx = indices[j]
                    pid = self.gold_chunks[idx].get("parent_doc_id", "")
                    if pid not in seen:
                        seen.add(pid)
                        unique_chunks.append(self.gold_chunks[idx])
                        unique_scores.append(float(score))
            return unique_chunks[:top_k], unique_scores[:top_k]

        else:
            for ct, quota in TYPE_QUOTA.items():
                sub, indices = self.type_sub_indices.get(ct, (None, []))
                if sub is None or not indices:
                    continue
                k      = min(quota * 5, len(indices))
                sc, ix = sub.search(query_embedding, k)
                count  = 0
                for j, score in zip(ix[0], sc[0]):
                    if j == -1 or count >= quota:
                        continue
                    idx = indices[j]
                    pid = self.gold_chunks[idx].get("parent_doc_id", "")
                    if pid not in seen:
                        seen.add(pid)
                        unique_chunks.append(self.gold_chunks[idx])
                        unique_scores.append(float(score))
                        count += 1

            if len(unique_chunks) < top_k:
                sub, indices = self.type_sub_indices.get("review", (None, []))
                if sub is not None:
                    need   = (top_k - len(unique_chunks)) * 5
                    sc, ix = sub.search(query_embedding, min(need, len(indices)))
                    for j, score in zip(ix[0], sc[0]):
                        if len(unique_chunks) >= top_k or j == -1:
                            break
                        idx = indices[j]
                        pid = self.gold_chunks[idx].get("parent_doc_id", "")
                        if pid not in seen:
                            seen.add(pid)
                            unique_chunks.append(self.gold_chunks[idx])
                            unique_scores.append(float(score))

            return unique_chunks, unique_scores

    # =========================================================
    # 메인 검색 (FAISS → [MultiQuery] → Reranker)
    # =========================================================
    def search(self, query_embedding: np.ndarray, intent: str,
               top_k: int = 5, use_multi_query: bool = False,
               original_query: str = "") -> tuple:

        rerank_k = top_k * 3  # Reranker용 후보 수 (top_k의 3배)

        # ── info 질문: title 필터링 + Reranker
        if intent == "info":
            title_hint = ""
            if original_query:
                title_hint = self.extract_title_from_query(original_query)
                if title_hint:
                    print(f"   [Info] 제목 추출: '{title_hint}'")
            chunks, scores = self._search_single(
                query_embedding, intent, rerank_k, title_hint=title_hint
            )
            return self._rerank(original_query, chunks, scores, top_k)

        # ── general 질문: 단일 또는 MultiQuery + Reranker
        if use_multi_query and original_query:
            queries = self.generate_multi_queries(original_query, n=3)
            print(f"   MultiQuery: {len(queries)}개 쿼리 생성")
            for i, q in enumerate(queries):
                print(f"     [{i}] {q}")

            pid_scores = {}
            pid_chunks = {}
            for q in queries:
                if not q.strip():
                    continue
                emb = self.embed_query(q)
                cks, scs = self._search_single(emb, intent, top_k=rerank_k)
                for chunk, score in zip(cks, scs):
                    pid = chunk.get("parent_doc_id", "")
                    if pid:
                        pid_scores[pid] = pid_scores.get(pid, 0) + score
                        pid_chunks[pid] = chunk

            sorted_pids = sorted(pid_scores, key=lambda p: pid_scores[p], reverse=True)
            chunks = [pid_chunks[p] for p in sorted_pids[:rerank_k]]
            scores = [pid_scores[p] for p in sorted_pids[:rerank_k]]

        else:
            chunks, scores = self._search_single(
                query_embedding, intent, top_k=rerank_k
            )

        return self._rerank(original_query, chunks, scores, top_k)

    def log_results(self, query: str, intent: str, chunks: list, scores: list):
        print(f"\n🔍 질문: {query}")
        print(f"   질문 타입: {intent}")
        print(f"   결과: {len(chunks)}개")
        print("-" * 70)
        for i, (c, s) in enumerate(zip(chunks, scores)):
            print(f"  Rank {i+1} | Score: {round(s,4)} | Type: {c['chunk_type']:12} | Title: {c.get('title','')}")
        print("-" * 70)