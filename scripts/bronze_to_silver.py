"""
transform.py  —  Bronze → Silver 변환 (장르별 개별 파일 생성)

사용법:
    python transform.py

입력: data/bronze/ 폴더 안의 JSON 파일들
출력: data/silver/silver_{장르}.json (장르별 개별 파일)

예시:
    silver_소설.json
    silver_인문.json
    silver_경제경영.json
    silver_자기계발.json
    silver_시에세이.json

## 참고 사항

코드 및 데이터 필드 내 실제 서점명은 저작권 및 보안을 위해 익명화했습니다.
(영문: BookStore / 국문: 북스토어)
    
"""

import json
import re
import os

BRONZE_DIR = "data/bronze"
SILVER_DIR = "data/silver"


def clean_text(text):
    if not text:
        return None
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip() or None


def parse_pages(spec):
    raw = (spec or {}).get("쪽수")
    if not raw:
        return None
    m = re.search(r"\d+", str(raw))
    return int(m.group()) if m else None


def parse_date(raw):
    if not raw:
        return None
    raw = str(raw).strip()
    if re.fullmatch(r"\d{8}", raw):
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def parse_keywords(raw):
    if not raw:
        return []
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    return [t for t in tokens if not t.isdigit()]


def parse_tags(ai_summary):
    if not ai_summary or not ai_summary.get("tags"):
        return []
    return [t.lstrip("#") for t in ai_summary["tags"]]


def extract_genre_from_filename(fname):
    """파일명에서 장르 추출: BookStore_xxx(소설_raw_data).json → 소설"""
    m = re.search(r'\((.+?)_raw_data\)', fname)
    if m:
        return m.group(1)
    # 괄호 패턴이 없으면 파일명 그대로 사용
    return fname.replace(".json", "")


def get_source_name(genre):
    return f"북스토어 베스트셀러 목록({genre})"


def get_source_file_name(genre):
    return f"북스토어_베스트셀러_{genre}.json"


def transform_book(b, source_file="", genre_override=None):
    isbn  = b.get("isbn", "")
    genre = genre_override or b.get("genre_name") or b.get("genre", "")
    ai    = b.get("ai_review_summary")

    return {
        "doc_id":      f"BookStore_{isbn}",
        "source_type": "북스토어",
        "source_name": get_source_name(genre),
        "source_file": get_source_file_name(genre),

        "source_info": {
            "isbn":         isbn,
            "title":        clean_text(b.get("title")),
            "author":       b.get("author"),
            "authors": [
                {
                    "role": a.get("role"),
                    "name": a.get("name"),
                    "bio":  clean_text(a.get("bio"))
                }
                for a in (b.get("authors") or []) if a.get("name")
            ],
            "publisher":    clean_text(b.get("publisher")),
            "genre":        genre,
            "release_date": parse_date(b.get("release_date")),
            "price":        float(b["price"]) if b.get("price") not in (None, "") else None,
            "pages":        parse_pages(b.get("spec")),
            "spec": {
                "크기":     (b.get("spec") or {}).get("크기"),
                "총권수":   (b.get("spec") or {}).get("총권수"),
                "시리즈명": (b.get("spec") or {}).get("시리즈명"),
            },
            "rating_value": float(b["rating_value"]) if b.get("rating_value") not in (None, "") else None,
            "rating_count": int(b["rating_count"]) if b.get("rating_count") not in (None, "") else None,
        },

        "content_blocks": {
            "description_full":  clean_text(b.get("description_full")),
            "table_of_contents": clean_text(b.get("table_of_contents")),
            "book_excerpt":      clean_text(b.get("book_excerpt")),
            "writer_words":      clean_text(b.get("writer_words")),
        },

        "reviews": [
            {
                "rating":      r.get("rating"),
                "content":     clean_text(r.get("content")),
                "created_at":  r.get("created_at"),
                "emotion_tag": r.get("emotion_tag"),
            }
            for r in (b.get("reviews") or [])
            if clean_text(r.get("content"))
        ],

        "ai_processed": {
            "ai_review_summary": {
                "lead":   (ai or {}).get("lead"),
                "detail": clean_text((ai or {}).get("detail")),
                "tags":   parse_tags(ai),
            } if ai else None
        },

        "metadata": {
            "saleCmdtid":        b.get("saleCmdtid"),
            "rank":              b.get("rank"),
            "url":               b.get("url"),
            "image":             b.get("image"),
            "detail_image_urls": b.get("detail_image_urls") or [],
            "keywords":          parse_keywords(b.get("keywords")),
            "category_paths":    b.get("category_paths") or [],
            "awards":            b.get("awards") or [],
            "recommendations":   b.get("recommendations") or [],
            "date_published":    [],
            "review_count_raw":  len(b.get("reviews") or []),
        }
    }


def bronze_to_silver(bronze_books, genre_override=None, min_description_len=50):
    merged = {}
    for b in bronze_books:
        isbn = b.get("isbn")
        if not isbn:
            continue
        description = clean_text(b.get("description_full"))
        if not description or len(description) < min_description_len:
            continue
        silver = transform_book(b, genre_override=genre_override)
        if isbn not in merged:
            merged[isbn] = silver
    return list(merged.values())


if __name__ == "__main__":
    os.makedirs(SILVER_DIR, exist_ok=True)

    bronze_files = sorted([f for f in os.listdir(BRONZE_DIR) if f.endswith(".json")])
    if not bronze_files:
        print(f"❌ '{BRONZE_DIR}' 폴더에 JSON 파일이 없습니다.")
        exit(1)

    print(f"총 {len(bronze_files)}개 파일 발견\n")

    total_silver = 0
    for fname in bronze_files:
        genre = extract_genre_from_filename(fname)
        fpath = os.path.join(BRONZE_DIR, fname)

        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        silver_books = bronze_to_silver(data, genre_override=genre)

        output_fname = f"silver_{genre}.json"
        output_path  = os.path.join(SILVER_DIR, output_fname)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(silver_books, f, ensure_ascii=False, indent=2)

        print(f"✅ {fname}")
        print(f"   → {output_fname}: {len(silver_books)}권")
        total_silver += len(silver_books)

    print(f"\n✅ 전체 완료: 총 {total_silver}권")
    print(f"   저장 위치: {SILVER_DIR}/")