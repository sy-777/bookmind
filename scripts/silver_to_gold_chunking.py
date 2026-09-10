"""
silver_to_gold_chunking.py  —  Silver → Gold 청킹 (RecursiveCharacterTextSplitter 적용)

사용법:
    python silver_to_gold_chunking.py

입력: data/silver/silver_{장르}.json 파일들
출력: data/gold/gold_{장르}.json

chunk_id 규칙:
    BookStore_{isbn}_intro_info         : 기본 서지 정보
    BookStore_{isbn}_intro_bio          : 저자 소개
    BookStore_{isbn}_intro_desc         : 책소개
    BookStore_{isbn}_toc                : 목차
    BookStore_{isbn}_review_{n:02d}     : 리뷰 5개씩 묶음
    BookStore_{isbn}_extra              : 작가의 말 + 책 속 발췌

content 원칙:
    - 순수 원본 데이터만
    - 제목 접두어 없음
    - 마크다운 형식
    - AI 요약은 metadata로만 보존

주요 처리 내용:
    - 책 1권을 위 6가지 chunk_type으로 분해
    - intro_info / intro_bio는 분할 없이 통째로 1개 청크로 유지
      (서지정보·저자소개는 짧아서 굳이 나눌 필요 없음)
    - intro_desc(책소개) / extra(작가의말+발췌): chunk_size=800, overlap=100
    - toc(목차): chunk_size=600, overlap=50
    - review(리뷰): chunk_size=500, overlap=0, 리뷰 5개씩 배치로 묶어 청킹
    - 분할된 청크가 여러 개면 chunk_id에 _01, _02... 순번 부여
    - 모든 청크에 동일한 metadata(제목/저자/가격/평점/키워드/카테고리 등
      25개 필드)를 공통으로 부여 (base_meta)

## 참고 사항

코드 및 데이터 필드 내 실제 서점명은 저작권 및 보안을 위해 익명화했습니다.
(영문: BookStore / 국문: 북스토어)

"""

import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

SILVER_DIR = "data/silver"
GOLD_DIR   = "data/gold"

SPLITTERS = {
    "intro_desc": RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    ),
    "toc": RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=50,
        separators=["\n", " ", ""],
    ),
    "review": RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0,
        separators=["\n- ", "\n", " ", ""],
    ),
    "extra": RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    ),
}

NO_SPLIT_TYPES = {"intro_info", "intro_bio"}


def base_meta(silver):
    si   = silver.get("source_info", {})
    ai   = (silver.get("ai_processed") or {}).get("ai_review_summary") or {}
    meta = silver.get("metadata", {})

    category_paths = meta.get("category_paths") or []
    category_str   = " | ".join(category_paths) if category_paths else ""

    awards     = meta.get("awards") or []
    awards_str = " | ".join(awards) if awards else ""

    detail_images    = meta.get("detail_image_urls") or []
    detail_image_str = detail_images[0] if detail_images else ""

    return {
        # ── 식별자
        "parent_doc_id":  silver.get("doc_id", ""),      # book_id → parent_doc_id
        "isbn":           si.get("isbn", ""),
        # ── 출처
        "source_type":    silver.get("source_type", ""),  # ← 추가
        "source_name":    silver.get("source_name", ""),  # ← 추가
        "source_file":    silver.get("source_file", ""),  # ← 추가
        # ── 기본 서지
        "title":          si.get("title", ""),
        "author":         si.get("author", ""),
        "genre":          si.get("genre", ""),
        "publisher":      si.get("publisher", "") or "",
        "release_date":   si.get("release_date") or "",
        "price":          si.get("price") or 0.0,
        "pages":          si.get("pages") or 0,
        "rating_value":   si.get("rating_value") or 0.0,
        "rating_count":   si.get("rating_count") or 0,
        # ── 분류/검색
        "keywords":       ", ".join(meta.get("keywords") or []),
        "category_paths": category_str,
        "awards":         awards_str,
        # ── AI 요약
        "tags":           ", ".join(ai.get("tags") or []),
        "ai_lead":        ai.get("lead") or "",
        # ── 이미지/URL
        "url":            meta.get("url") or "",
        "image":          meta.get("image") or "",
        "detail_image":   detail_image_str,
        # ── 기타
        "rank":           meta.get("rank") or 0,
        "review_count":   meta.get("review_count_raw") or 0,
    }


def make_chunks(base_chunk_id, chunk_type, content, silver):
    meta   = base_meta(silver)
    chunks = []

    if chunk_type in NO_SPLIT_TYPES:
        chunks.append({
            "chunk_id":      base_chunk_id,
            "parent_doc_id": meta["parent_doc_id"],
            "title":         meta["title"],
            "chunk_type":    chunk_type,
            "content":       content,
            "metadata":      meta,
        })
    else:
        splitter = SPLITTERS.get(chunk_type)
        if splitter is None:
            chunks.append({
                "chunk_id":      base_chunk_id,
                "parent_doc_id": meta["parent_doc_id"],
                "title":         meta["title"],
                "chunk_type":    chunk_type,
                "content":       content,
                "metadata":      meta,
            })
        else:
            splits = splitter.split_text(content)
            if len(splits) == 1:
                chunks.append({
                    "chunk_id":      base_chunk_id,
                    "parent_doc_id": meta["parent_doc_id"],
                    "title":         meta["title"],
                    "chunk_type":    chunk_type,
                    "content":       splits[0],
                    "metadata":      meta,
                })
            else:
                for idx, split in enumerate(splits, start=1):
                    chunks.append({
                        "chunk_id":      f"{base_chunk_id}_{idx:02d}",
                        "parent_doc_id": meta["parent_doc_id"],
                        "title":         meta["title"],
                        "chunk_type":    chunk_type,
                        "content":       split,
                        "metadata":      meta,
                    })

    return chunks


def chunk_book(silver):
    chunks  = []
    doc_id  = silver.get("doc_id", "")
    si      = silver.get("source_info", {})
    cb      = silver.get("content_blocks", {})
    reviews = silver.get("reviews") or []
    meta    = silver.get("metadata", {})
    toc     = cb.get("table_of_contents")

    # ── 1. intro_info
    info_lines = []
    if si.get("title"):
        info_lines.append(f"# {si['title']}")
        info_lines.append("")
    if si.get("author"):
        info_lines.append(f"**저자**: {si['author']}")
    if si.get("publisher"):
        info_lines.append(f"**출판사**: {si['publisher']}")
    if si.get("genre"):
        info_lines.append(f"**장르**: {si['genre']}")

    category_paths = meta.get("category_paths") or []
    if category_paths:
        info_lines.append(f"**카테고리**: {category_paths[0]}")

    keywords = meta.get("keywords") or []
    if keywords:
        info_lines.append(f"**키워드**: {', '.join(keywords)}")

    if si.get("release_date"):
        info_lines.append(f"**발행일**: {si['release_date']}")
    if si.get("price"):
        info_lines.append(f"**가격**: {int(si['price'])}원")
    if si.get("pages"):
        info_lines.append(f"**쪽수**: {si['pages']}쪽")
    if si.get("rating_value"):
        info_lines.append(f"**평점**: {si['rating_value']} ({si.get('rating_count', 0)}명)")

    awards = meta.get("awards") or []
    if awards:
        info_lines.append(f"**수상/추천**: {' | '.join(awards)}")

    if info_lines:
        chunks.extend(make_chunks(f"{doc_id}_intro_info", "intro_info", "\n".join(info_lines), silver))

    # ── 2. intro_bio
    bio_lines = []
    for a in (si.get("authors") or []):
        if a.get("bio"):
            if bio_lines:
                bio_lines.append("")
            bio_lines.append(f"## 저자 소개 - {a['name']} ({a.get('role', '')})")
            bio_lines.append("")
            bio_lines.append(a["bio"])

    if bio_lines:
        chunks.extend(make_chunks(f"{doc_id}_intro_bio", "intro_bio", "\n".join(bio_lines), silver))

    # ── 3. intro_desc
    if cb.get("description_full"):
        chunks.extend(make_chunks(f"{doc_id}_intro_desc", "intro_desc", cb["description_full"], silver))

    # ── 4. toc
    if toc:
        chunks.extend(make_chunks(f"{doc_id}_toc", "toc", toc, silver))

    # ── 5. review
    if reviews:
        chunk_size = 5
        for batch_idx, start in enumerate(range(0, len(reviews), chunk_size), start=1):
            batch = reviews[start:start + chunk_size]
            review_lines = []
            for r in batch:
                content = r.get("content") or ""
                if content:
                    rating  = r.get("rating", "")
                    emotion = r.get("emotion_tag", "")
                    line    = f"- [{rating}점]"
                    if emotion:
                        line += f" ({emotion})"
                    line += f" {content[:300]}"
                    review_lines.append(line)

            if review_lines:
                chunks.extend(make_chunks(
                    f"{doc_id}_review_{batch_idx:02d}", "review",
                    "\n".join(review_lines), silver
                ))

    # ── 6. extra
    extra_parts = []
    if cb.get("writer_words"):
        extra_parts.append(cb["writer_words"])
    if cb.get("book_excerpt"):
        if extra_parts:
            extra_parts.append("")
        extra_parts.append(cb["book_excerpt"])

    if extra_parts:
        chunks.extend(make_chunks(f"{doc_id}_extra", "extra", "\n\n".join(extra_parts), silver))

    return chunks


if __name__ == "__main__":
    os.makedirs(GOLD_DIR, exist_ok=True)

    silver_files = sorted([
        f for f in os.listdir(SILVER_DIR)
        if f.startswith("silver_") and f.endswith(".json")
    ])

    if not silver_files:
        print(f"❌ '{SILVER_DIR}' 폴더에 silver_*.json 파일이 없습니다.")
        exit(1)

    print(f"총 {len(silver_files)}개 실버 파일 발견\n")

    total_chunks = 0
    for fname in silver_files:
        genre = fname.replace("silver_", "").replace(".json", "")
        fpath = os.path.join(SILVER_DIR, fname)

        with open(fpath, encoding="utf-8") as f:
            silver_books = json.load(f)

        gold_chunks = []
        for book in silver_books:
            gold_chunks.extend(chunk_book(book))

        output_fname = f"gold_{genre}.json"
        output_path  = os.path.join(GOLD_DIR, output_fname)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(gold_chunks, f, ensure_ascii=False, indent=2)

        print(f"✅ {fname} ({len(silver_books)}권)")
        print(f"   → {output_fname}: {len(gold_chunks)}개 청크")
        total_chunks += len(gold_chunks)

    print(f"\n✅ 전체 완료: 총 {total_chunks}개 청크")
    print(f"   저장 위치: {GOLD_DIR}/")