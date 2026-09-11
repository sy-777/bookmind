# 🔖 BookMind

나만의 취향을 기억하는 개인화 독서 추천 AI

## 소개

BookMind는 사용자의 독서 취향과 기록을 기억하고, 이를 바탕으로 책을 추천하는 멀티에이전트 기반 RAG 챗봇입니다. LangGraph로 구성된 Book Agent와 Summary Agent가 협업하여 책 검색, 상세 정보 제공, 독후감 관리, 사용자 프로필 기반 추천 기능을 제공합니다.

## 참고 사항

코드 및 데이터 필드 내 실제 서점명은 저작권 및 보안을 위해 명시하지 않았습니다.

## 주요 기능

- 📚 자연어 기반 책 검색 및 추천
- 📖 책 상세 정보 조회 (서지 정보, 작가 소개, 독자 리뷰)
- 📝 독후감 / 메모 작성 및 관리
- 📊 독서 통계 (읽은 책, 평균 평점 등)
- 🔊 TTS(음성 답변) 지원
- 💬 스트리밍 답변 UI (Streamlit)

## 아키텍처

```
서점사이트 베스트 도서 크롤링 (crawling/book_data_crawling.ipynb)
    ↓
Bronze (원본 데이터)
    ↓ bronze_to_silver.py
Silver (전처리 데이터)
    ↓ silver_to_gold_chunking.py
Gold (청킹 완료 데이터)
    ↓ faiss_load.py
FAISS 벡터 인덱스 구축
    ↓
EnsembleRetriever (BM25 + FAISS) → CrossEncoder Reranker
    ↓
LangGraph 멀티에이전트 (Book Agent + Summary Agent)
    ↓
Streamlit UI
```


## 기술 스택

- **LLM / Orchestration**: LangGraph, LangChain
- **임베딩**: BAAI/bge-m3
- **리랭커**: BAAI/bge-reranker-base
- **벡터 검색**: FAISS
- **UI**: Streamlit
- **데이터 소스**: 서점사이트 베스트 도서 크롤링 (경제/경영, 시/에세이, 자기계발, 소설, 인문 5개 장르, 장르당 100권 총 500권)

## 폴더 구조

```
bookmind/
├── crawling/                           # 서점사이트 베스트 도서 크롤링 코드
├── agent/
│   ├── graph.py                        # LangGraph 워크플로우 정의
│   └── tools.py                        # Book Agent 도구 5종
├── rag/
│   ├── retriever.py                    # EnsembleRetriever (BM25 + FAISS)
│   └── reranker.py                     # CrossEncoder 리랭커
├── scripts/
│   ├── bronze_to_silver.py             # Bronze → Silver 전처리
│   ├── silver_to_gold_chunking.py      # Silver → Gold 청킹
│   └── faiss_load.py                   # FAISS 인덱스 로드
├── streamlit/
│   └── app.py                          # Streamlit UI                            
└── README.md
```

## Book Agent 도구 (Tools)

| 도구 | 설명 |
|---|---|
| `recommend_books` | 사용자 질문과 독서 기록(좋아한 책, 메모)을 함께 반영해 책을 추천. Multi-Query 검색 후 정확한 서지정보 보장을 위해 intro_info 청크로 교체하는 후처리 포함 |
| `get_book_detail` | 특정 책의 저자, 출판사, 장르, 가격, 평점 등 서지정보를 조회 |
| `save_reading_record` | 독서 상태(읽음/읽는중/읽고싶음), 평점, 독후감을 저장하거나 기존 기록을 업데이트 |
| `get_user_profile` | 읽은 책 수, 평균 평점 등 사용자의 독서 통계와 최근 읽은 책 목록을 조회 |
| `search_user_memos` | 과거 작성한 독후감/메모를 키워드 기반으로 검색 |


## 설치 및 실행

```bash
# 1. 저장소 클론
git clone <repo-url>
cd bookmind

# 2. 패키지 설치
# (GPU 없이 실행 시, 무거운 CUDA 패키지 다운로드를 피하려면 먼저 CPU 버전 torch 설치)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. .env 파일 생성 후 OpenAI API 키 등록
OPENAI_API_KEY=your_key_here

# 4. Streamlit 앱 실행
streamlit run streamlit/app.py
```

## 데이터 안내

- 저작권 문제로 서점 원본 데이터(`data_original/`, 500권 전체)는 저장소에 포함되어 있지 않습니다.
- 대신 `data/` 폴더에 장르당 5권씩 추린 데모용 샘플(gold/faiss)만 포함돼 있어 바로 실행해볼 수 있습니다.
- 크롤링 출처, 책 상세 URL 등 서점을 특정할 수 있는 정보는 익명화 처리했습니다.
- 책 표지 이미지는 실제 표지 대신, 저작권 문제를 피하기 위해 흰 배경에 "Book Cover" 텍스트를 넣은 CSS 플레이스홀더로 표시됩니다.

