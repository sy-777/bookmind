"""
rag/reranker.py
BookMind CrossEncoder 기반 Reranker

모델: BAAI/bge-reranker-v2-m3 (한국어 지원)
역할: FAISS 후보 청크 → 질문과의 관련도 재평가 → 상위 top_k 반환

사용 방법:
    from rag.reranker import BookMindReranker
    reranker = BookMindReranker()
    chunks, scores = reranker.rerank(query, chunks, scores, top_k=5)

주요 처리 내용:
    - 각 후보 청크의 title + content 앞 400자를 결합해 query-document
      쌍 생성 (CrossEncoder의 max_length=512에 맞춰 컨텍스트 확보)
    - CrossEncoder로 query와 각 후보를 1:1 정밀 비교해 관련도 점수 산출
    - 점수 내림차순 정렬 후 상위 top_k만 반환
    - 모델 로드 실패 또는 예측 중 오류 발생 시, FAISS 원본 순위/점수를
      그대로 반환하여 서비스 중단 방지 (fail-safe)
"""

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 한국어 지원, bge-reranker-base보다 성능 좋음


class BookMindReranker:
    def __init__(self):
        print(f"   🔧 Reranker 로드 중: {RERANKER_MODEL} (최초 1회만)")
        try:
            self.model = CrossEncoder(RERANKER_MODEL, max_length=512)
            print(f"   ✅ Reranker 로드 완료")
        except Exception as e:
            print(f"   ⚠️  Reranker 로드 실패: {e}")
            self.model = None

    def rerank(self, query: str, chunks: list, scores: list,
               top_k: int = 5) -> tuple:
        """
        FAISS 후보 청크를 CrossEncoder로 재정렬

        Args:
            query  : 사용자 질문
            chunks : FAISS가 반환한 청크 리스트
            scores : FAISS 유사도 점수 리스트
            top_k  : 최종 반환 개수

        Returns:
            (reranked_chunks, reranked_scores)
        """
        if not self.model or not chunks:
            return chunks[:top_k], scores[:top_k]

        # 청크 content로 query-document 쌍 생성
        # title + content 앞부분을 같이 넣어서 컨텍스트 강화
        pairs = []
        for c in chunks:
            title   = c.get("title", "")
            content = c.get("content", "")[:400]
            text    = f"{title}\n{content}".strip()
            pairs.append((query, text))

        try:
            rerank_scores = self.model.predict(pairs)

            # 점수 기준 내림차순 정렬
            ranked = sorted(
                zip(chunks, rerank_scores.tolist()),
                key=lambda x: x[1],
                reverse=True
            )

            top_chunks = [r[0] for r in ranked[:top_k]]
            top_scores = [float(r[1]) for r in ranked[:top_k]]

            print(f"   [Reranker] {len(chunks)}개 후보 → 상위 {top_k}개 재정렬")
            return top_chunks, top_scores

        except Exception as e:
            print(f"   ⚠️  Reranker 실패 ({e}) → FAISS 결과 그대로 사용")
            return chunks[:top_k], scores[:top_k]