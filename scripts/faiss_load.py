"""
faiss_load.py  —  Gold → FAISS 인덱스 저장 (1회 실행)

사용법:
    python faiss_load.py

입력: data/gold/gold_*.json 파일들
출력:
    data/faiss/index.faiss   ← FAISS 벡터 인덱스
    data/faiss/chunks.json   ← 청크 메타데이터

주요 처리 내용:
    - data/gold/의 모든 gold_*.json을 읽어 청크를 하나로 합침
    - 각 청크의 content를 OpenAI text-embedding-3-small 모델로 임베딩
      (100개씩 배치 처리, 8000자 초과 텍스트는 앞부분만 사용)
    - 임베딩 벡터를 L2 정규화한 뒤 IndexFlatIP(내적 기반)로 인덱스 생성
      → 정규화된 벡터의 내적 = 코사인 유사도가 되므로, 검색 시 코사인
        유사도 기반 검색이 됨
    - 인덱스와 별개로, 검색 결과 표시에 필요한 청크 메타데이터를
      chunks.json에 순서를 맞춰 저장 (인덱스의 벡터 순번 = chunks.json의
      리스트 순번)

Score 범위: 0 ~ 1 (높을수록 유사)
"""

import json
import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GOLD_DIR   = "data/gold"
FAISS_DIR  = "data/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(FAISS_DIR, "chunks.json")


def get_embeddings(client, texts, model="text-embedding-3-small", batch_size=100):
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = [t[:8000] for t in texts[i:i + batch_size]]
        response = client.embeddings.create(model=model, input=batch)
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)
        print(f"  [{min(i + batch_size, total)}/{total}] 임베딩 완료")

    return all_embeddings


if __name__ == "__main__":
    os.makedirs(FAISS_DIR, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY가 없습니다.")
        exit(1)

    client = OpenAI(api_key=api_key)

    # ── 골드 파일 전체 로드
    gold_files = sorted([f for f in os.listdir(GOLD_DIR) if f.endswith(".json")])
    if not gold_files:
        print("❌ data/gold/ 폴더에 JSON 파일이 없습니다.")
        exit(1)

    gold_chunks = []
    for fname in gold_files:
        with open(os.path.join(GOLD_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)
        gold_chunks.extend(data)
        print(f"✅ {fname}: {len(data)}개 청크 로드")

    print(f"\n총 {len(gold_chunks)}개 청크")

    # ── 임베딩 계산
    print("\n🔧 임베딩 계산 중...")
    texts = [c["content"] for c in gold_chunks]
    embeddings = get_embeddings(client, texts)

    # ── FAISS 인덱스 생성 (코사인 유사도)
    print("\n💾 FAISS 인덱스 생성 중...")
    dim = len(embeddings[0])
    emb_array = np.array(embeddings, dtype=np.float32)

    # L2 정규화 → Inner Product = 코사인 유사도
    faiss.normalize_L2(emb_array)
    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)

    # ── 저장
    faiss.write_index(index, INDEX_PATH)
    print(f"✅ FAISS 인덱스 저장: {INDEX_PATH}")
    print(f"   총 {index.ntotal}개 벡터")

    # ── 청크 메타데이터 저장
    chunks_data = [
        {
            "chunk_id":      c["chunk_id"],
            "parent_doc_id": c["parent_doc_id"],
            "title":         c["title"],
            "chunk_type":    c["chunk_type"],
            "content":       c["content"],
            "metadata":      c.get("metadata", {}),
        }
        for c in gold_chunks
    ]
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False)
    print(f"✅ 청크 메타데이터 저장: {CHUNKS_PATH}")
    print(f"\n이제 streamlit run app.py 실행하세요!")