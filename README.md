# KoreaHerald Digest (Python + Cron/Serverless)

자동으로 Korea Herald RSS를 수집하고, 기사 본문을 추출한 뒤 OpenAI로 한국어 요약과 학습용 영단어 리스트(CEFR 추정 포함)를 생성하여
- 일간 마크다운 다이제스트 (`out/YYYY-MM-DD_digest.md`)
- Anki 단어장 CSV (`out/YYYY-MM-DD_anki_vocab.csv`)

를 만듭니다.

## 빠른 시작

### 0) 사전 준비
- Python 3.10+
- OpenAI API Key

### 1) 설치
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY 값을 입력하세요.
```

### 2) 드라이런(LLM 호출 없이 제목만 확인)
```bash
python digest.py --dry-run --max-items 5
```

### 3) 실제 실행(요약/단어 생성 + 파일 저장)
```bash
python digest.py --max-items 8 --out out
```

### 4) 스케줄링(둘 중 택1)
- **로컬/서버 cron**: `crontab -e`에 다음 예시(매일 07:30 KST)
  ```
  30 7 * * * /path/to/venv/python /path/to/project/digest.py --max-items 8 --out /path/to/project/out >> /path/to/project/log.txt 2>&1
  ```
- **GitHub Actions**: `.github/workflows/daily.yml`를 참고해 주세요.

## 출력물 구조
- `out/2025-09-23_digest.md` — 기사별 한국어 요약 + 원문 링크
- `out/2025-09-23_anki_vocab.csv` — Front/Back 두 열(Anki 표준 임포트)

## 참고
- Korea Herald RSS: https://www.koreaherald.com/rss
- 본문 추출: trafilatura
- OpenAI Chat Completions(JSON mode) 사용
