#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Korea Herald RSS → 본문 추출 → OpenAI 요약/어휘 → Markdown & Anki CSV
"""
import os, argparse, csv, datetime, json, sys, time
from dataclasses import dataclass, asdict
from typing import List, Optional
from urllib.parse import urlparse
import re, xml.etree.ElementTree as ET
import ssl, httpx, feedparser
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from bs4 import BeautifulSoup
import re

from urllib.parse import quote_plus

# --- Telegram message writer ---------------------------------
def _is_simple_english_word(w: str) -> bool:
    if not w:
        return False
    # 영문자/공백/하이픈/아포스트로피만 허용 (고유명사 필터는 LLM 단계에서 수행)
    for ch in w:
        if ch.isalpha() or ch in " -'":
            continue
        return False
    return True

def write_telegram_messages(outdir: str, date_ymd: str, digests: List[ArticleDigest], topn: int = 8):
    """
    메시지 포맷:
      m/d

      word1 뜻1
      word2 뜻2
      ...

      기사 링크
    """
    ensure_dir(outdir)
    tg_dir = os.path.join(outdir, "tg_msgs")
    ensure_dir(tg_dir)

    # 날짜 m/d
    m = int(date_ymd[5:7]); d = int(date_ymd[8:10])
    md_str = f"{m}/{d}"

    created = []
    for idx, dgt in enumerate(digests, start=1):
        lines = []
        count = 0
        for v in dgt.vocab:
            w = (v.word or "").strip()
            mean = (v.meaning_ko or "").strip()
            if not w or not mean:
                continue
            lines.append(f"{w} {mean}")  # ← 단어 다음에 뜻을 공백으로 연결
            count += 1
            if count >= topn:
                break

        vocab_block = "\n".join(lines) if lines else "(no vocab)"
        text = f"{md_str}\n\n{vocab_block}\n\n{dgt.link}"

        path = os.path.join(tg_dir, f"{date_ymd}_msg_{idx}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        created.append(path)
    return created




def resolve_final_url(url: str, client) -> str:
    # Google News 중계 URL → 원문으로 리다이렉트 추적
    try:
        r = client.get(url, follow_redirects=True, timeout=15.0)
        return str(r.url)
    except Exception:
        return url


def google_news_feed_for_domain(domain: str, days: int = 3):
    # 최근 N일간 KH 기사만: when:N d, 영어 UI 고정
    query = f"site:{domain} when:{days}d"
    # 주: 구글 뉴스 RSS 검색
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"

def collect_from_google_news(client, domain: str, limit: int):
    import feedparser
    url = google_news_feed_for_domain(domain)
    r = client.get(url, follow_redirects=True)
    print(f"[INFO] Google News GET {url} -> {r.status_code}")
    if r.status_code != 200:
        return []
    parsed = feedparser.parse(r.content)
    print(f"[INFO] Google News parsed entries: {len(parsed.entries)}")
    items = []
    seen = set()
    for e in parsed.entries[: max(limit, 20)]:
        link = getattr(e, "link", None)
        title = (getattr(e, "title", "") or "").strip()
        if not link or not title:
            continue
        # 구글뉴스 링크는 중간 리디렉션이 있을 수 있지만, 최종 페이지(언론사)로 잘 넘어감.
        if link in seen:
            continue
        seen.add(link)
        items.append({"title": title, "link": link, "summary": getattr(e, "summary", "").strip(), "pub": None})
    return items[:limit]


HTML_SOURCES = [
    "https://www.koreaherald.com/",
    "https://www.koreaherald.com/National",
    "https://www.koreaherald.com/Business",
    "https://www.koreaherald.com/Culture",
]

ARTICLE_PAT = re.compile(r"https?://(m\.)?koreaherald\.com/(article/\d+|view\.php\?ud=\d+)", re.I)

def collect_article_links_from_html(client, max_items=30):
    links, seen = [], set()
    for url in HTML_SOURCES:
        try:
            r = client.get(url, follow_redirects=True)
            print(f"[INFO] HTML GET {url} -> {r.status_code}")
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/"):
                    href = "https://www.koreaherald.com" + href
                if ARTICLE_PAT.match(href) and href not in seen:
                    seen.add(href)
                    title = (a.get_text() or "").strip()
                    links.append({"title": title or href, "link": href, "summary": "", "pub": None})
            if len(links) >= max_items:
                break
        except Exception as e:
            print(f"[WARN] HTML fetch failed: {url} ({e})")
            continue
    return links[:max_items]

SITEMAP_CANDIDATES = [
    "https://www.koreaherald.com/sitemap.xml",
    "https://www.koreaherald.com/sitemap-index.xml",
    "https://m.koreaherald.com/sitemap.xml",
]

ARTICLE_URL_RE = re.compile(r"https?://(m\.)?koreaherald\.com/(view\.php\?ud=\d+|article/\d+)", re.I)

def fetch_text(client, url):
    r = client.get(url, follow_redirects=True)
    if r.status_code == 200:
        return r.text
    return None

def parse_sitemap_urls(xml_text):
    # urlset or sitemapindex 모두 지원
    urls = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return urls
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    # 1) urlset → <url><loc>
    for loc in root.findall(".//sm:url/sm:loc", ns):
        if loc.text:
            urls.append(loc.text.strip())
    # 2) sitemapindex → 하위 sitemap 재귀
    for sm_loc in root.findall(".//sm:sitemap/sm:loc", ns):
        if sm_loc.text:
            urls.append(sm_loc.text.strip())
    return urls

def find_recent_article_links_via_sitemaps(client, max_items=20):
    # 1차: sitemap 인덱스/메인에서 url/sitemap 링크 수집
    queue = []
    seen = set()
    for s in SITEMAP_CANDIDATES:
        txt = fetch_text(client, s)
        if not txt:
            continue
        for u in parse_sitemap_urls(txt):
            if u not in seen:
                seen.add(u)
                queue.append(u)

    # sitemap 안의 sitemap(섹션별)도 열기
    collected = []
    for u in queue[:50]:  # 과도한 호출 방지
        txt = fetch_text(client, u)
        if not txt:
            continue
        # 이 파일 안의 실제 기사 URL들 추출
        for v in parse_sitemap_urls(txt):
            if ARTICLE_URL_RE.match(v):
                collected.append(v)
        # 혹시 urlset이 아니라 일반 텍스트면 정규식으로도 긁자
        for v in re.findall(r"https?://[^<>\s\"]+", txt):
            if ARTICLE_URL_RE.match(v):
                collected.append(v)

        if len(collected) >= max_items * 3:
            break

    # 중복 제거 + 최신 링크가 앞쪽에 오도록 단순 정렬(숫자 id 기준)
    uniq = []
    seen2 = set()
    for u in collected:
        if u in seen2:
            continue
        seen2.add(u)
        uniq.append(u)

    # 숫자 ID(=기사번호)가 크면 최신일 가능성 높음
    def sort_key(x):
        m = re.search(r"/(\d+)$", x) or re.search(r"ud=(\d+)", x)
        return int(m.group(1)) if m else 0

    uniq.sort(key=sort_key, reverse=True)
    return uniq[:max_items]


REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36 NewsDigestBot/1.0",
    "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
}

def make_ssl_context(insecure=False, ca_bundle=None, seclevel1=False):
    if insecure:
        return ssl._create_unverified_context()
    ctx = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else ssl.create_default_context()
    if seclevel1:
        # 일부 기업망 루트CA가 약한 키로 서명된 경우 우회
        try:
            ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
        except Exception:
            pass
    return ctx

try:
    from trafilatura import fetch_url, extract as trafi_extract
except Exception as e:
    print("ERROR: trafilatura import failed. Did you install requirements?", file=sys.stderr)
    raise

# ---- Models ----

@dataclass
class VocabItem:
    word: str
    meaning_ko: str
    pos: Optional[str] = None
    example_en: Optional[str] = None
    cefr: Optional[str] = None

@dataclass
class ArticleDigest:
    title: str
    link: str
    ko_summary: str
    vocab: List[VocabItem]

# ---- Utils ----

def today_ymd(tz="Asia/Seoul") -> str:
    # naive local date is fine for filenames; customize if needed
    return datetime.date.today().isoformat()

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def read_env():
    load_dotenv()
    feeds = os.getenv("KOREAHERALD_FEEDS", "https://www.koreaherald.com/rss")
    feeds = [f.strip() for f in feeds.split(",") if f.strip()]
    max_items = int(os.getenv("MAX_ITEMS", "8"))
    outdir = os.getenv("OUTPUT_DIR", "out")
    tz = os.getenv("TZ", "Asia/Seoul")
    return feeds, max_items, outdir, tz

def pick_entries(feeds: List[str], limit: int):
    import feedparser, time, httpx
    seen_links, entries = set(), []
    alt_map = {"https://www.koreaherald.com/rss": ["https://www.koreaherald.com/rss/newsAll"]}

    # 입력 피드 + 보조 피드 병합
    to_fetch = []
    for u in feeds:
        to_fetch.append(u)
        if u in alt_map:
            to_fetch.extend(alt_map[u])

    ctx = make_ssl_context(insecure=True, seclevel1=True)  # 임시: 인증 우회(회사망 SSL 검사 회피)
    # ↑ 루트 ②(회사 루트 CA 신뢰)로 전환되면 insecure=False, ca_bundle=... 로 바꾸세요.

    # 1) RSS 시도
    with httpx.Client(timeout=15.0, headers=REQUEST_HEADERS, verify=ctx) as client:
        for url in to_fetch:
            try:
                r = client.get(url)
                print(f"[INFO] GET {url} -> {r.status_code}")
                if r.status_code != 200:
                    continue
                parsed = feedparser.parse(r.content)
                print(f"[INFO] RSS parsed entries: {len(parsed.entries)}")
            except Exception as e:
                print(f"[WARN] RSS fetch failed: {url} ({e})")
                continue

            for e in parsed.entries:
                link = getattr(e, "link", None)
                title = getattr(e, "title", "").strip()
                if not link or not title or link in seen_links:
                    continue
                seen_links.add(link)
                pub = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
                entries.append({
                    "title": title,
                    "link": link,
                    "summary": getattr(e, "summary", "").strip(),
                    "pub": pub
                })

        if entries:
            # 최신순 정렬 후 제한
            entries.sort(key=lambda x: (x["pub"] is not None, x["pub"], x["title"]), reverse=True)
            return entries[:limit] if limit else entries

        # 2) RSS가 0개면 → 사이트맵에서 최신 기사 링크 수집
        print("[INFO] RSS empty. Trying sitemaps fallback...")
        urls = find_recent_article_links_via_sitemaps(client, max_items=max(20, limit or 20))
        print(f"[INFO] Sitemaps collected URLs: {len(urls)}")
        for link in urls:
            if link in seen_links:
                continue
            seen_links.add(link)
            # 제목은 나중에 본문에서 추출해도 되지만 우선 링크만
            entries.append({"title": link, "link": link, "summary": "", "pub": None})

        # (여기서 바로 return 하지 말고) ──────▼

        if not entries:
            print("[INFO] Sitemaps empty. Trying HTML fallback (homepage/category)...")
            html_entries = collect_article_links_from_html(client, max_items=max(20, limit or 20))
            print(f"[INFO] HTML fallback collected: {len(html_entries)}")
            entries.extend(html_entries)

        # ★ 여기부터 추가: HTML도 실패하면 Google News로
        if not entries:
            print("[INFO] HTML fallback empty. Trying Google News domain feed...")
            gn_entries = collect_from_google_news(client, "koreaherald.com", limit or 10)
            print(f"[INFO] Google News fallback collected: {len(gn_entries)}")
            entries.extend(gn_entries)

        if not entries:
            print("[WARN] No entries from all fallbacks.")
            return []

        return entries[:limit] if limit else entries

from trafilatura import extract as trafi_extract

def extract_article_text(url: str, client=None) -> Optional[str]:
    """
    httpx로 HTML을 가져와 trafilatura.extract로 본문 추출
    (회사망 SSL 검사 우회를 위해 상위에서 생성한 verify=ctx 클라이언트를 재사용)
    """
    try:
        if client is None:
            # 최후 수단: 기본 클라이언트(가능하면 상단 client를 넘겨 사용)
            import httpx
            ctx = make_ssl_context(insecure=True, seclevel1=True)
            client = httpx.Client(timeout=15.0, headers=REQUEST_HEADERS, verify=ctx, follow_redirects=True)

        final = resolve_final_url(url, client)
        r = client.get(final)  # follow_redirects=True 설정되어 있음
        if r.status_code != 200:
            return None
        html = r.text
        text = trafi_extract(html, include_comments=False, favor_precision=True)
        return text
    except Exception:
        return None

# ---- LLM ----

def have_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def llm_analyze(title: str, text: str) -> ArticleDigest:
    """
    OpenAI Chat Completions(JSON mode) 호출 → 한국어 요약 + 학습용 어휘 10~15개
    """
    from openai import OpenAI
    client = OpenAI()

    system = (
        "You are a bilingual news study assistant. "
        "Return compact Korean summaries and study-ready English vocabulary lists. "
        "IMPORTANT: Output strictly in the JSON schema requested."
    )
    user = {
        "title": title,
        "text": text[:8000]  # 길이 제한 보호
    }

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Extract: 1) a Korean summary of 3-5 sentences capturing facts, names, numbers; "
                "2) a vocabulary list of 10-15 EN words good for B1-C1 learners. "
                "For each vocab item include fields: word, meaning_ko, pos, example_en, cefr(A1~C2). "
                "Avoid proper nouns/rare technical terms. "
                "Return JSON with keys: ko_summary (string), vocab (array). "
                "Title: {title}\n\nText:\n{text}"
            ).format(**user),
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    data = json.loads(content)

    ko_summary = data.get("ko_summary", "").strip()
    vocab_list = []
    for v in data.get("vocab", []):
        vocab_list.append(VocabItem(
            word=v.get("word","").strip(),
            meaning_ko=v.get("meaning_ko","").strip(),
            pos=v.get("pos"),
            example_en=v.get("example_en"),
            cefr=v.get("cefr")
        ))
    return ArticleDigest(title=title, link="", ko_summary=ko_summary, vocab=vocab_list)

# ---- Writers ----

def write_markdown(outdir: str, date_ymd: str, digests: List[ArticleDigest]):
    ensure_dir(outdir)
    path = os.path.join(outdir, f"{date_ymd}_digest.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Korea Herald Digest — {date_ymd}\n\n")
        for d in digests:
            f.write(f"## {d.title}\n")
            if d.link:
                f.write(f"[원문 보기]({d.link})\n\n")
            f.write(d.ko_summary.strip() + "\n\n")
            if d.vocab:
                f.write("**Vocab**\n\n")
                for v in d.vocab:
                    f.write(f"- {v.word} — {v.meaning_ko} ({v.pos or 'pos'}) • {v.cefr or 'CEFR?'}\n")
                    if v.example_en:
                        f.write(f"  - e.g., {v.example_en}\n")
                f.write("\n")
    return path

def write_anki_csv(outdir: str, date_ymd: str, digests: List[ArticleDigest]):
    ensure_dir(outdir)
    path = os.path.join(outdir, f"{date_ymd}_anki_vocab.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Front", "Back"])
        for d in digests:
            for v in d.vocab:
                front = f"{v.word} — {v.meaning_ko}"
                back = f"{v.pos or ''} | {v.cefr or ''}\\n{v.example_en or ''}\\n(from: {d.title})"
                w.writerow([front, back])
    return path

# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feeds", type=str, default=None, help="Comma-separated RSS URLs (optional)")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, help="Output directory (default from env)")
    parser.add_argument("--dry-run", action="store_true", help="LLM 호출 없이 제목/링크만 출력")
    args = parser.parse_args()

    feeds_env, max_items_env, outdir_env, tz = read_env()
    feeds = [f.strip() for f in (args.feeds.split(",") if args.feeds else feeds_env)]
    max_items = args.max_items if args.max_items else max_items_env
    outdir = args.out or outdir_env

    # ✅ SSL context 정의 (회사망 환경 대응)
    ctx = make_ssl_context(insecure=True, seclevel1=True)

    with httpx.Client(timeout=15.0, headers=REQUEST_HEADERS, verify=ctx, follow_redirects=True) as client:
        entries = pick_entries(feeds, max_items)

        if args.dry_run:
            print("[DRY RUN] Latest entries:")
            for e in entries:
                print("-", e["title"], "=>", e["link"])
            return

        if not have_key():
            print("ERROR: OPENAI_API_KEY not set; create .env and try again.", file=sys.stderr)
            sys.exit(1)

        digests: List[ArticleDigest] = []
        for e in entries:
            title = e["title"]
            link = e["link"]
            text = extract_article_text(link, client) or e.get("summary") or title
            analyzed = llm_analyze(title, text)
            analyzed.link = resolve_final_url(link, client)  # 요약 파일의 링크도 원문으로 정리
            digests.append(analyzed)

    ymd = today_ymd(tz)
    md_path = write_markdown(outdir, ymd, digests)
    csv_path = write_anki_csv(outdir, ymd, digests)

    # ✅ 텔레그램 메시지 파일 생성(기사별 1건)
    tg_paths = write_telegram_messages(outdir, ymd, digests, topn=8)
    print("Telegram message files:")
    for p in tg_paths:
        print(" -", p)
    print("Wrote:", md_path)
    print("Wrote:", csv_path)

if __name__ == "__main__":
    main()
