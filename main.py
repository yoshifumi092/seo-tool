import asyncio
import json
import os
import re
import time
import uuid
import unicodedata
from urllib.parse import urlparse

import fitz  # PyMuPDF
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from playwright.async_api import async_playwright
from pydantic import BaseModel

app = FastAPI(title="SEO削除申請ツール")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

sessions: dict = {}
SESSION_TTL = 3600

# URL解析結果キャッシュ（同じURLを再解析してAPIレート制限を回避）
_analysis_cache: dict = {}  # key: (url, trademark) → {"result": dict, "ts": float}
CACHE_TTL = 300  # 5分（同一URLの連続リクエスト対策。本番で安定したら延長可）



# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────

def clean_filename(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", text)
    text = text.strip().replace(" ", "_").replace("\u3000", "_")
    return text[:50] or "document"


def validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    if not parsed.netloc:
        return False
    blocked = ("localhost", "127.", "192.168.", "10.", "172.16.", "::1")
    for b in blocked:
        if parsed.netloc.startswith(b) or parsed.netloc == b.rstrip("."):
            return False
    return True


def cleanup_session(session_id: str) -> None:
    session = sessions.pop(session_id, None)
    if not session:
        return
    for key in ("pdf_path", "output_path"):
        path = session.get(key)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


async def _schedule_cleanup(session_id: str, delay: int) -> None:
    await asyncio.sleep(delay)
    cleanup_session(session_id)


# ─────────────────────────────────────────────
# URL → PDF
# ─────────────────────────────────────────────

async def url_to_pdf(url: str, output_path: str) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--disable-accelerated-2d-canvas",
                "--no-first-run",
                "--no-zygote",
                "--disable-gpu",
            ]
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
            extra_http_headers={
                "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
            },
        )

        # navigator.webdriver を隠すスクリプトを注入
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['ja-JP', 'ja', 'en-US', 'en'] });
            window.chrome = { runtime: {} };
        """)

        page = await context.new_page()

        # まず domcontentloaded で取得
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        except Exception:
            await page.goto(url, wait_until="load", timeout=45000)

        # JS・画像の読み込みを待つ（少し長めに）
        await page.wait_for_timeout(5000)

        # アクセスブロック検出
        page_text = await page.inner_text("body")
        block_keywords = ["アクセスできません", "403", "Access Denied", "Forbidden", "blocked", "Bot detection"]
        if any(kw.lower() in page_text.lower() for kw in block_keywords) and len(page_text.strip()) < 500:
            await browser.close()
            raise HTTPException(400, "このサイトはBot対策によりアクセスがブロックされました。別のURLをお試しください。")

        await page.pdf(
            path=output_path,
            format="A4",
            print_background=True,
            margin={"top": "15px", "bottom": "15px", "left": "15px", "right": "15px"},
        )
        await browser.close()


# ─────────────────────────────────────────────
# テキスト抽出
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()
    return "\n".join(pages_text)


# ─────────────────────────────────────────────
# AI解析
# ─────────────────────────────────────────────

def _build_analysis_prompt(
    text_chunk: str,
    url: str,
    trademark_hint: str,
    section_label: str = "",
) -> str:
    section_note = f"（記事テキスト {section_label}）" if section_label else "（記事テキスト）"
    return f"""あなたは、記事削除申請準備のための権利侵害論点整理AIです。

【役割】
対象記事の中から、削除申請・修正申請・社内法務確認の材料となる具体表現を抽出し、問題性を整理する。
感情的批判や一般論ではなく、対象記事の具体文言に基づいて冷静に出力する。

【対象URL】
{url}
{trademark_hint}

【分析手順】
第1段階: 記事全体の構造・目的・収益動線を把握する
  - 記事タイトル・URLに対象の商標名・人名を使って検索流入を奪っていないか
  - 「中立な調査・口コミ記事」を装いながら実質的に別サービスへ誘導していないか
  - LINEへの誘導・アフィリエイトリンク・比較ランキングなど収益構造があるか
  - 記事全体が読者に与える最終印象は何か

第2段階: 各文・見出しを4種類に分類する
  - 事実主張: 料金・所在地・実績・会社情報等の断定
  - 意見論評: 怪しい・信頼しづらい・危険等の評価
  - 推測: おそらく・可能性がある・かもしれない等
  - 印象操作: 直接断定せず読者がネガティブに受け取るよう設計された表現

第3段階: 以下の問題類型に分類する（複数該当可）
  A: 名誉毀損リスク（詐欺・危険・悪質等の断定またはこれに近い表現）
  B: 信用毀損リスク（サービス品質・安全性・誠実性への信頼を低下させる表現）
  C: 営業妨害（登録するな・やめろ等の行動阻害、購入・参加を妨げる警告）
  D: 商標・不正競争（対象の商標名・人名でSEO集客しながら別サービスへ誘導）
  E: 事実誤認（会社情報・料金・実績・人物情報が事実と異なる可能性）
  F: 根拠不明な断定（出典・証拠なしの強い評価・断定）
  G: 印象操作・ミスリード（見出しの強断定、疑問形による実質断定、引用選別）
  H: 競合誘導・アフィリエイトバイアス（対象を下げて自社・他社へ誘導）

第4段階: 記事を先頭から末尾まで段落ごとに必ずスキャンすること
  - 各段落・各見出しを1つずつ確認し、問題のある表現を見落とさない
  - 「問題なさそう」と判断した段落でも必ず全文を読んでから次へ進む

第5段階: 特に見落としやすい以下のパターンを必ず確認する
  ① タイトル・見出しが本文より強断定になっていないか（「詐欺なのか？」等の疑問形も含む）
  ② 商標名・人名をタイトル/URLに使いながら別のLINE・サービスへ誘導していないか（D類型・最重要）
  ③ 口コミ・体験談を選別引用してネガティブ印象を形成していないか
  ④ 「稼げない」「怪しい」「信頼性が低い」等を証拠なく事実のように書いていないか
  ⑤ 「安全なのか」「本当に大丈夫か」等の不安を煽る表現で購入・参加を妨げていないか
  ⑥ 「〜と言われている」「〜という声がある」等の責任回避表現で実質断定していないか
  ⑦ 比較表で対象だけ評価を低く設定していないか（恣意的な評価軸・数値）
  ⑧ 記事末尾に「おすすめ」「安全な代替」等を誘導していないか

【絶対禁止事項】
- 記事に書かれていない内容を推測・補完しない
- 法律上の最終結論を断定しない（「該当し得る」「おそれがある」「問題となり得る」を使う）
- 感情的・糾弾的な表現を使わない
- テキスト内に存在しない文字列を引用しない
- 同じ趣旨の指摘を言い換えで重複させない

【指摘しないもの】
- ナビゲーション・メニュー・パンくず・タグ・カテゴリ・コピーライト
- 著者名・日付・SNSボタン単体
- 完全に中立な事実説明・導入文

【抽出ルール】
- 問題性のある箇所はすべて抽出する。見落とすことは削除申請の材料を失うことと同義であるため厳禁
- 同趣旨の指摘は統合する。ただし論点が独立しているならば必ず別指摘にする
- 件数は問わない。記事の内容に応じて正直に出力する（多くても少なくても水増し・省略しない）
- 「もう十分だろう」と途中で打ち切ることは禁止。最後の段落まで必ずスキャンを完了してから出力する
- D類型（商標名を使った不正誘導）は必ず最初に指摘する

【指摘品質の基準】
各指摘は必ず以下の4層構造で説明すること：
  文言 → 読者印象 → 問題の本質 → 侵害観点

悪い例: 「この表現は名誉毀損です」
良い例: 「当該表現は、対象が著しく不誠実または危険であるとの印象を一般読者に与え得る一方、その評価を基礎づける具体的事実や客観的根拠の提示が十分ではなく、対象の社会的信用を不当に低下させるおそれがあります」

【最重要：商標名利用・不正誘導パターンの検出】
このパターンは削除申請において最も強力な論点のひとつである。必ず確認すること。
- 記事タイトル・見出し・URLに対象の商標名や人名を使って検索上位を狙っていないか
- そうして集めた読者を、対象とは無関係の自社LINEや他サービスへ誘導していないか
- 対象のブランド名で検索した読者が、当該記事を経由して別サービスへ流れる構造になっていないか
これらが確認できた場合、severity=5でD類型として必ず指摘すること。

以下のJSON形式のみで返してください（前置き・補足・コードブロック一切不要）：

{{
  "article_title": "記事タイトル（原文のまま）",
  "trademark": "この記事が攻撃している商標・サービス名または人名",
  "overall_tone": "記事全体の論調・読者に与える全体印象（1〜2文）",
  "violations": [
    {{
      "text": "問題のある記述（テキストから15〜60文字を一字一句そのまま引用）",
      "statement_type": "事実主張／意見論評／推測／印象操作　のいずれか",
      "type": "問題類型の名称のみ（記号不要。例：根拠不明な断定、商標権侵害・不正競争、名誉毀損リスク）",
      "severity": 問題の深刻度（1〜5の整数。5が最重要）,
      "confidence": "高／中／低（本文だけで問題性を説明できるか）",
      "reader_impression": "一般読者がこの表現から受ける印象（1文）",
      "explanation": "なぜ問題か（文言→読者印象→問題本質→侵害観点の順で2〜3文）",
      "deletion_comment": "削除申請書に転用しやすい指摘文案（法的表現で1〜2文）"
    }}
  ],
  "primary_claims": ["削除申請の主論点候補（severity4〜5の要約）"],
  "human_review_notes": ["人間が別途確認すべきポイント"]
}}

{section_note}
{text_chunk}"""


def _parse_ai_response(response_text: str) -> dict:
    # コードブロックマーカーを先に除去してからパース
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```\s*$', '', text)
        text = text.strip()

    parsers = [
        lambda t: json.loads(t),
        lambda t: json.loads(t[t.find("{") : t.rfind("}") + 1]),
    ]
    for parser in parsers:
        try:
            result = parser(text)
            if "violations" in result:
                return result
        except Exception:
            continue
    return {"article_title": "記事", "trademark": "", "violations": []}


async def _call_gemini_once(
    text_chunk: str,
    url: str,
    trademark_hint: str,
    section_label: str = "",
) -> dict:
    import urllib.request
    import json as _json

    api_key = os.environ["GEMINI_API_KEY"]
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    prompt = _build_analysis_prompt(text_chunk, url, trademark_hint, section_label)
    payload = _json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 16000,
            "temperature": 0.2,
            "thinkingConfig": {"thinkingBudget": 12000},
        },
    }).encode()

    def _call():
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return _json.loads(resp.read())

    try:
        result = await asyncio.to_thread(_call)
    except Exception as e:
        err = str(e)
        import sys
        body = ""
        if hasattr(e, "read"):
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
        print(f"[Gemini Error] {err} | body={body}", flush=True, file=sys.stderr)
        if "429" in err:
            raise HTTPException(429, f"Gemini APIレート制限: {body or err[:200]}")
        if "400" in err or "API_KEY_INVALID" in err:
            raise HTTPException(400, f"Gemini APIキーエラー: {body or err[:200]}")
        raise HTTPException(500, f"Gemini APIエラー: {err[:200]}")
    # Gemini 2.5系レスポンス解析: 複数のpartから有効なJSONを探す
    import sys as _sys
    parts = result["candidates"][0]["content"]["parts"]
    print(f"[Gemini] parts count={len(parts)}, keys={[list(p.keys()) for p in parts]}", flush=True, file=_sys.stderr)

    text = ""
    # 優先1: thought=Trueでなく "violations" を含むpart
    for part in parts:
        if part.get("thought"):
            continue
        t = part.get("text", "")
        if '"violations"' in t:
            text = t
            break
    # 優先2: thought=Trueでない最初のpart
    if not text:
        for part in parts:
            if part.get("thought"):
                continue
            t = part.get("text", "")
            if t:
                text = t
                break
    # 優先3: "violations" を含む任意のpart
    if not text:
        for part in parts:
            t = part.get("text", "")
            if '"violations"' in t:
                text = t
                break
    # 最終: 最後のpart
    if not text and parts:
        text = parts[-1].get("text", "")

    print(f"[Gemini] selected text (first 500): {text[:500]}", flush=True, file=_sys.stderr)
    parsed = _parse_ai_response(text)
    print(f"[Gemini] parsed: violations={len(parsed.get('violations', []))}, title={parsed.get('article_title','')[:30]}", flush=True, file=_sys.stderr)
    return parsed


async def _analyze_once(
    text_chunk: str,
    url: str,
    trademark_hint: str,
    section_label: str = "",
) -> dict:
    """Geminiで1チャンクを解析する（429の場合は65秒待って1回リトライ）。"""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if not gemini_key:
        raise HTTPException(500, "GEMINI_API_KEY を Railway の環境変数に設定してください。")

    try:
        return await _call_gemini_once(text_chunk, url, trademark_hint, section_label)
    except HTTPException as e:
        if e.status_code == 400:
            raise HTTPException(400, "Gemini APIキーが無効です。RailwayのGEMINI_API_KEYを確認してください。")
        raise


async def analyze_with_claude(text: str, url: str, trademark: str = "") -> dict:
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(500, "GEMINI_API_KEY が設定されていません。")

    trademark_hint = (
        f"なお、ユーザーより対象商標として「{trademark}」が指定されています。"
        if trademark else ""
    )

    # キャッシュチェック（同じURLを再解析してAPIレート制限を回避）
    cache_key = (url, trademark)
    cached = _analysis_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < CACHE_TTL:
        return cached["result"]

    result = await _analyze_once(text[:12000], url, trademark_hint)

    # キャッシュに保存
    _analysis_cache[cache_key] = {"result": result, "ts": time.time()}
    # 古いキャッシュを掃除（最大50件）
    if len(_analysis_cache) > 50:
        oldest = min(_analysis_cache, key=lambda k: _analysis_cache[k]["ts"])
        del _analysis_cache[oldest]

    return result


async def analyze_area_with_ai(text: str, trademark: str = "") -> dict:
    """選択エリアのテキストをGeminiで解析して違反カテゴリと理由を返す。"""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return {"type": "手動追加", "explanation": "ユーザーにより手動で追加された指摘箇所"}

    import urllib.request
    import json as _json

    trademark_hint = f"対象商標：{trademark}" if trademark else ""
    prompt = f"""以下のテキストはネガティブ記事の一部です。{trademark_hint}
削除申請・修正要求に使える客観的な指摘として、違反カテゴリと理由を特定してください。

【分析の視点】
- 事実の摘示か意見・感想かを切り分ける
- 根拠の弱い断定表現かどうかを見る
- 社会的評価の低下・営業信用毀損・業務妨害につながるかを判断する
- 感情的表現ではなく削除申請に転用できる法的表現で指摘する

テキスト：「{text[:500]}」

以下のJSON形式のみで回答してください：
{{
  "type": "違反カテゴリ（名誉毀損/信用毀損・業務妨害/印象操作/虚偽・根拠なし/過大なネガティブ表現/営業妨害/寄生マーケティング/不正競合 のいずれか）",
  "explanation": "この記述が問題である具体的な理由（2〜3文。根拠の弱さ・断定の強さ・読者への印象を指摘し、削除申請に直接転用できる表現で記載してください）"
}}"""

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    payload = _json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 512},
    }).encode()

    def _call():
        req = urllib.request.Request(
            endpoint, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return _json.loads(resp.read())

    try:
        result = await asyncio.to_thread(_call)
        # Gemini 2.5思考モデル対応: thought=TrueのpartをスキップしてJSONを含むpartを探す
        parts = result["candidates"][0]["content"]["parts"]
        response_text = ""
        for part in parts:
            if part.get("thought"):
                continue
            t = part.get("text", "")
            if t:
                response_text = t
                break
        if not response_text and parts:
            response_text = parts[-1].get("text", "")
        parsers = [
            lambda t: json.loads(t),
            lambda t: json.loads(re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", t).group(1)),
            lambda t: json.loads(t[t.find("{") : t.rfind("}") + 1]),
        ]
        for parser in parsers:
            try:
                r = parser(response_text)
                if "type" in r and "explanation" in r:
                    return r
            except Exception:
                continue
    except Exception:
        pass

    return {"type": "手動追加", "explanation": "ユーザーにより手動で追加された指摘箇所"}


# ─────────────────────────────────────────────
# PDF加工
# ─────────────────────────────────────────────

def _normalize_search_text(text: str) -> str:
    """検索テキストをNFKC正規化＋空白正規化する。"""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\s\u3000\u00a0]+', ' ', text)
    return text.strip()


def _best_rect(rects: list, page_height: float) -> fitz.Rect:
    """
    複数のマッチ候補から最適な1件を返す。
    優先順位: ① ページ上部150pt以外かつ下部50pt以外の本文領域
              ② それ以外ならy座標が最も大きい（本文に最も近い）もの
    ページ上部はヘッダー・ナビゲーション・パンくずリストが集中するため除外する。
    """
    body = [r for r in rects if r.y0 > 150 and r.y1 < page_height - 50]
    candidates = body if body else rects
    return min(candidates, key=lambda r: r.y0)


def _search_text_in_page(page: fitz.Page, normalized_text: str) -> fitz.Rect | None:
    """
    ページ内でテキストを検索して座標を返す。
    方法1: search_for（高速・複数候補から本文優先）
    方法2: 文字単位の座標マッチング（Playwright生成PDFの日本語対応）
    """
    ph = page.rect.height

    # ── 方法1: search_for ──
    # 短い文字列ほどナビゲーション等への誤マッチが増えるため、
    # 最短でも12文字までしか短縮しない
    for length in [len(normalized_text), 40, 30, 20, 15, 12]:
        cand = normalized_text[:length].strip()
        if len(cand) < 8:
            break
        rects = page.search_for(cand, quads=False)
        if rects:
            return _best_rect(rects, ph)

    # ── 方法2: rawdict で文字単位の位置マッチング ──
    target_nospace = re.sub(r'\s+', '', normalized_text)[:40]
    if len(target_nospace) < 4:
        return None

    try:
        all_chars = []
        for block in page.get_text("rawdict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for ch in span.get("chars", []):
                        c = unicodedata.normalize("NFKC", ch["c"])
                        if not c.strip():
                            continue
                        all_chars.append({"c": c, "bbox": ch["bbox"]})

        if not all_chars:
            return None

        page_str = "".join(ch["c"] for ch in all_chars)

        # 全出現箇所を収集して最適候補を選ぶ
        # 最短でも12文字まで：短すぎるとナビゲーション等に誤マッチする
        for trim in [len(target_nospace), 40, 30, 20, 15, 12]:
            short = target_nospace[:trim]
            if len(short) < 8:
                break

            occurrences = []
            start = 0
            while True:
                idx = page_str.find(short, start)
                if idx < 0:
                    break
                matched = all_chars[idx: idx + len(short)]
                if matched:
                    x0 = min(c["bbox"][0] for c in matched)
                    y0 = min(c["bbox"][1] for c in matched)
                    x1 = max(c["bbox"][2] for c in matched)
                    y1 = max(c["bbox"][3] for c in matched)
                    occurrences.append(fitz.Rect(x0, y0, x1, y1))
                start = idx + 1

            if occurrences:
                return _best_rect(occurrences, ph)

        return None

    except Exception:
        return None


def find_violation_positions(pdf_path: str, violations: list) -> list:
    """各違反テキストのPDF上の座標を検索する（リスト形式で返す）。
    テキストが見つからない場合でも必ずページ0に配置する。
    """
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    page_widths  = [doc[p].rect.width  for p in range(page_count)]
    page_heights = [doc[p].rect.height for p in range(page_count)]
    positions = []
    not_found_count = 0  # 見つからなかった件数（フォールバック位置計算用）

    for i, v in enumerate(violations):
        raw_text = v.get("text", "")
        normalized = _normalize_search_text(raw_text)
        found = False

        for page_num in range(page_count):
            page = doc[page_num]
            rect = _search_text_in_page(page, normalized)
            if rect:
                positions.append({
                    "page_num": page_num,
                    "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
                    "number": i + 1,
                    "type": v.get("type", ""),
                    "statement_type": v.get("statement_type", ""),
                    "severity": v.get("severity", 3),
                    "confidence": v.get("confidence", "中"),
                    "reader_impression": v.get("reader_impression", ""),
                    "explanation": v.get("explanation", ""),
                    "deletion_comment": v.get("deletion_comment", ""),
                    "text": raw_text,
                })
                found = True
                break

        if not found:
            # テキストが見つからない場合：
            # ページ0の最初のテキストブロック付近に配置（ユーザーがドラッグ移動可能）
            pw = page_widths[0] if page_widths else 595
            ph = page_heights[0] if page_heights else 842
            row = not_found_count % 20
            fy0 = 30.0 + row * 40
            fy1 = fy0 + 16
            positions.append({
                "page_num": 0,
                "rect": [10.0, fy0, min(pw - 10, 400.0), fy1],
                "number": i + 1,
                "type": v.get("type", ""),
                "statement_type": v.get("statement_type", ""),
                "severity": v.get("severity", 3),
                "confidence": v.get("confidence", "中"),
                "reader_impression": v.get("reader_impression", ""),
                "explanation": v.get("explanation", ""),
                "deletion_comment": v.get("deletion_comment", ""),
                "text": raw_text,
                "auto_placed": True,
            })
            not_found_count += 1

    doc.close()
    return positions


def sort_and_renumber(violations: list) -> None:
    """ページ順・縦位置順に並べ替えてナンバリングを振り直す。"""
    def sort_key(v):
        page = v.get("page_num", -1)
        if page < 0 or not v.get("rect"):
            return (999, 999)
        return (page, v["rect"][1])  # ページ番号 → y座標（上から順）

    violations.sort(key=sort_key)
    for i, v in enumerate(violations):
        v["number"] = i + 1


def _draw_violation_on_page(page: fitz.Page, item: dict, font_path) -> None:
    """1件の違反を指定ページに描画する（赤枠＋注釈テキストボックス）。"""
    r = item["rect"]
    rect = fitz.Rect(r[0], r[1], r[2], r[3]) if isinstance(r, (list, tuple)) else r
    # アルファベット記号（例: "D: "）を除去
    vtype = re.sub(r'^[A-H]:\s*', '', item.get("type", ""))
    explanation = item.get("explanation", "")
    pw, ph = page.rect.width, page.rect.height

    # ── 赤枠 ──
    num = item.get("number", "")
    expanded = fitz.Rect(
        max(0, rect.x0 - 3), max(0, rect.y0 - 3),
        min(pw, rect.x1 + 3), min(ph, rect.y1 + 3),
    )
    page.draw_rect(expanded, color=(1, 0, 0), width=2.5)

    # ── 番号バッジ（赤枠の左上角） ──
    if num:
        badge_size = 14.0
        bx = expanded.x0
        by = expanded.y0 - badge_size if expanded.y0 - badge_size >= 0 else expanded.y0
        badge_rect = fitz.Rect(bx, by, bx + badge_size, by + badge_size)
        page.draw_rect(badge_rect, color=(1, 0, 0), fill=(1, 0, 0))
        try:
            page.insert_text(
                fitz.Point(bx + 2, by + badge_size - 3),
                str(num),
                fontsize=8, color=(1, 1, 1),
                fontfile=font_path if font_path else None,
                fontname="Helv" if not font_path else None,
            )
        except Exception:
            page.insert_text(
                fitz.Point(bx + 2, by + badge_size - 3),
                str(num), fontsize=8, color=(1, 1, 1), fontname="Helv",
            )

    # ── 注釈ボックスのサイズ・位置 ──
    box_w, chars = 200.0, 22
    exp_lines = max(2, (len(explanation) + chars - 1) // chars)
    box_h = float(min(max(14 + exp_lines * 12 + 10, 52), 140))

    ann_pos = item.get("annotation_pos")
    if ann_pos and len(ann_pos) == 2:
        ax, ay = float(ann_pos[0]), float(ann_pos[1])
    else:
        ax, ay = auto_place_annotation(expanded, page.rect, box_w, box_h)

    ax = max(0.0, min(ax, pw - box_w))
    ay = max(0.0, min(ay, ph - box_h))
    ann_rect = fitz.Rect(ax, ay, ax + box_w, ay + box_h)
    inner    = fitz.Rect(ax + 4, ay + 4, ax + box_w - 4, ay + box_h - 4)

    # ── 白背景ボックス ──
    page.draw_rect(ann_rect, color=(0.75, 0, 0), fill=(1, 1, 1), width=1.0)

    # ── テキスト挿入（方法1: insert_htmlbox） ──
    label = f"【{num}】{vtype}" if num else vtype
    inserted = False
    if hasattr(page, "insert_htmlbox"):
        html = (
            f'<span style="font-size:9pt;font-weight:bold;color:#cc0000;">{label}</span>'
            f'<br>'
            f'<span style="font-size:8pt;color:#333333;">{explanation}</span>'
        )
        try:
            page.insert_htmlbox(inner, html)
            inserted = True
        except Exception:
            pass

    # ── テキスト挿入（方法2: CJKフォントファイルで insert_text） ──
    if not inserted and font_path:
        try:
            page.insert_text(fitz.Point(ax + 4, ay + 13), label,
                             fontsize=9, color=(0.75, 0, 0), fontfile=font_path)
            ty, text = ay + 26, explanation
            while text and ty < ay + box_h - 5:
                page.insert_text(fitz.Point(ax + 4, ty), text[:chars],
                                 fontsize=8, color=(0.15, 0.15, 0.15), fontfile=font_path)
                text, ty = text[chars:], ty + 12
            inserted = True
        except Exception:
            pass

    # ── テキスト挿入（方法3: ASCII フォールバック） ──
    if not inserted:
        page.insert_text(fitz.Point(ax + 4, ay + 13), label[:30],
                         fontsize=8, color=(0.75, 0, 0), fontname="Helv")
        page.insert_text(fitz.Point(ax + 4, ay + 25), explanation[:55],
                         fontsize=7, color=(0.15, 0.15, 0.15), fontname="Helv")


def build_annotated_pdf(pdf_path: str, violations: list) -> fitz.Document:
    """
    グレースケール変換＋赤枠注釈を行い fitz.Document を返す。
    ポイント: 赤枠を「先に描画」してから画像を overlay=False で背面挿入する。
    これにより画像が赤枠を隠す問題を回避する。
    """
    font_path = _get_cjk_font_path()
    orig_doc = fitz.open(pdf_path)
    new_doc = fitz.open()

    for page_num in range(len(orig_doc)):
        orig_page = orig_doc[page_num]
        pw, ph = orig_page.rect.width, orig_page.rect.height
        new_page = new_doc.new_page(width=pw, height=ph)

        # ① 赤枠＋注釈テキストを先に描画（これが最前面になる）
        for item in violations:
            if item.get("page_num") == page_num and item.get("rect"):
                _draw_violation_on_page(new_page, item, font_path)

        # ② グレースケール画像を背面に挿入（overlay=False = 既存の描画の後ろ）
        mat = fitz.Matrix(2, 2)
        pix = orig_page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        new_page.insert_image(
            fitz.Rect(0, 0, pw, ph),
            pixmap=pix,
            overlay=False,  # ← 背面に挿入。赤枠が隠れない。
        )

    orig_doc.close()
    return new_doc


def auto_place_annotation(expanded: fitz.Rect, page_rect: fitz.Rect, box_w: float, box_h: float, margin: float = 6) -> tuple:
    """赤枠の近くで最もスペースのある位置を返す（下→右→左→上の優先順）。"""
    pw, ph = page_rect.width, page_rect.height
    rx0, ry0, rx1, ry1 = expanded.x0, expanded.y0, expanded.x1, expanded.y1
    # 下
    if ry1 + box_h + margin <= ph:
        return (max(0, min(rx0, pw - box_w)), ry1 + margin)
    # 右
    if rx1 + box_w + margin <= pw:
        return (rx1 + margin, max(0, min(ry0, ph - box_h)))
    # 左
    if rx0 - box_w - margin >= 0:
        return (rx0 - box_w - margin, max(0, min(ry0, ph - box_h)))
    # 上
    if ry0 - box_h - margin >= 0:
        return (max(0, min(rx0, pw - box_w)), ry0 - box_h - margin)
    # フォールバック（ページ下端）
    return (max(0, min(rx0, pw - box_w)), max(0, ph - box_h - 5))


def _find_cjk_font() -> str | None:
    """日本語フォントファイルのパスを返す。見つからなければ None。"""
    import glob as _glob
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    # macOS Hiragino（ファイル名が日本語のためglobで探す）
    for pattern in [
        "/System/Library/Fonts/ヒラギノ角ゴ*.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック*.ttc",
    ]:
        found = _glob.glob(pattern)
        if found:
            candidates.insert(0, found[0])
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


_CJK_FONT_PATH: str | None = None
_CJK_FONT_SEARCHED = False


def _get_cjk_font_path() -> str | None:
    global _CJK_FONT_PATH, _CJK_FONT_SEARCHED
    if not _CJK_FONT_SEARCHED:
        _CJK_FONT_PATH = _find_cjk_font()
        _CJK_FONT_SEARCHED = True
    return _CJK_FONT_PATH


def _wrap_text(text: str, chars_per_line: int) -> list[str]:
    """テキストを指定文字数で折り返す（日本語考慮）。"""
    lines = []
    while text:
        lines.append(text[:chars_per_line])
        text = text[chars_per_line:]
    return lines


def add_red_annotations(doc: fitz.Document, positions: list) -> None:
    font_path = _get_cjk_font_path()

    for item in positions:
        if item["rect"] is None:
            continue
        page_num = item.get("page_num", 0)
        if page_num < 0 or page_num >= len(doc):
            continue
        page = doc[page_num]
        r = item["rect"]
        rect = fitz.Rect(r[0], r[1], r[2], r[3]) if isinstance(r, (list, tuple)) else r
        vtype = item.get("type", "")
        explanation = item.get("explanation", "")

        # ── 赤枠を描画 ──
        expanded = fitz.Rect(
            max(0, rect.x0 - 3),
            max(0, rect.y0 - 3),
            min(page.rect.width, rect.x1 + 3),
            min(page.rect.height, rect.y1 + 3),
        )
        page.draw_rect(expanded, color=(1, 0, 0), width=2.5)

        # ── テキストボックスのサイズを計算 ──
        box_w = 200.0
        chars_per_line = 22
        exp_lines = max(2, (len(explanation) + chars_per_line - 1) // chars_per_line)
        box_h = float(min(max(12 + 14 + exp_lines * 12 + 8, 52), 140))

        # ── 注釈位置の決定 ──
        ann_pos = item.get("annotation_pos")
        if ann_pos and len(ann_pos) == 2:
            ax, ay = float(ann_pos[0]), float(ann_pos[1])
        else:
            ax, ay = auto_place_annotation(expanded, page.rect, box_w, box_h)

        ax = max(0.0, min(ax, page.rect.width - box_w))
        ay = max(0.0, min(ay, page.rect.height - box_h))
        ann_rect = fitz.Rect(ax, ay, ax + box_w, ay + box_h)

        # ── 白背景＋赤枠ボックスを描画 ──
        page.draw_rect(ann_rect, color=(0.75, 0, 0), fill=(1, 1, 1), width=1.0)

        # ── テキスト挿入（方法1: insert_htmlbox）──
        inserted = False
        html = (
            f'<span style="font-size:9pt;font-weight:bold;color:#cc0000;">{vtype}</span>'
            f'<br>'
            f'<span style="font-size:8pt;color:#333333;">{explanation}</span>'
        )
        inner = fitz.Rect(ax + 3, ay + 3, ax + box_w - 3, ay + box_h - 3)
        try:
            page.insert_htmlbox(inner, html)
            inserted = True
        except Exception:
            pass

        if not inserted:
            # ── 方法2: insert_textbox（CJKフォントあり） ──
            if font_path:
                try:
                    font = fitz.Font(fontfile=font_path)
                    # カテゴリ名（1行目）
                    page.insert_text(
                        fitz.Point(ax + 3, ay + 12),
                        vtype,
                        fontsize=9, color=(0.75, 0, 0),
                        fontfile=font_path,
                    )
                    # 理由文（折り返し）
                    ty = ay + 25
                    for line in _wrap_text(explanation, chars_per_line):
                        if ty + 12 > ay + box_h - 3:
                            break
                        page.insert_text(
                            fitz.Point(ax + 3, ty),
                            line,
                            fontsize=8, color=(0.2, 0.2, 0.2),
                            fontfile=font_path,
                        )
                        ty += 12
                    inserted = True
                except Exception:
                    pass

        if not inserted:
            # ── 方法3: ASCII フォントで最低限表示 ──
            page.insert_text(
                fitz.Point(ax + 3, ay + 12),
                vtype[:30],
                fontsize=8, color=(0.75, 0, 0), fontname="Helv",
            )
            page.insert_text(
                fitz.Point(ax + 3, ay + 24),
                explanation[:60],
                fontsize=7, color=(0.2, 0.2, 0.2), fontname="Helv",
            )


async def create_summary_page(positions: list, article_title: str, trademark: str) -> str:
    rows_html = ""
    for item in positions:
        preview = item["text"][:55] + ("…" if len(item["text"]) > 55 else "")
        rows_html += f"""
        <tr>
          <td class="num">【{item["number"]}】</td>
          <td class="type">{item["type"]}</td>
          <td class="excerpt">「{preview}」</td>
          <td class="reason">{item["explanation"]}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Hiragino Kaku Gothic Pro", "Yu Gothic", "Meiryo", "Noto Sans JP", "MS Gothic", sans-serif;
      font-size: 10.5px; padding: 30px 35px; color: #1a1a1a; line-height: 1.55;
    }}
    h1 {{ font-size: 17px; color: #b00000; border-bottom: 3px solid #b00000; padding-bottom: 8px; margin-bottom: 14px; }}
    .meta {{ background: #fff5f5; border-left: 4px solid #b00000; padding: 10px 14px; margin-bottom: 14px; }}
    .meta p {{ margin: 3px 0; font-size: 11.5px; }}
    .meta strong {{ color: #b00000; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 10px; }}
    th {{ background: #b00000; color: #fff; padding: 7px 5px; text-align: left; font-size: 10.5px; }}
    td {{ border: 1px solid #ddd; padding: 6px 5px; vertical-align: top; }}
    tr:nth-child(even) {{ background: #fff8f8; }}
    .num  {{ font-weight: bold; color: #b00000; font-size: 12px; text-align: center; white-space: nowrap; }}
    .type {{ font-weight: bold; color: #b00000; white-space: nowrap; min-width: 110px; }}
    .excerpt {{ font-style: italic; color: #444; min-width: 160px; }}
    .reason {{ min-width: 200px; }}
    .footer {{ margin-top: 18px; font-size: 9px; color: #777; border-top: 1px solid #ddd; padding-top: 8px; }}
  </style>
</head>
<body>
  <h1>📋 削除申請 指摘事項一覧表</h1>
  <div class="meta">
    <p><strong>記事タイトル：</strong>{article_title}</p>
    <p><strong>侵害・攻撃対象の商標・ブランド：</strong>{trademark}</p>
    <p><strong>指摘件数：</strong>{len(positions)} 件</p>
  </div>
  <table>
    <thead>
      <tr>
        <th style="width:55px">No.</th>
        <th style="width:115px">違反カテゴリ</th>
        <th style="width:185px">問題箇所（抜粋）</th>
        <th>指摘内容・削除理由</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  <div class="footer">
    本書類は、上記URLに掲載された記事のホスティングサービスへの削除申請のために作成されました。
    赤枠で囲まれた箇所および上記一覧に記載の問題箇所は、名誉毀損・営業妨害・不正競合その他の
    法令または利用規約に違反する疑いがあるため、速やかな削除を要請いたします。
  </div>
</body>
</html>"""

    summary_path = f"/tmp/{uuid.uuid4()}_summary.pdf"
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox", "--disable-setuid-sandbox"])
        pg = await browser.new_page()
        await pg.set_content(html, wait_until="domcontentloaded")
        await pg.pdf(path=summary_path, format="A4", print_background=True)
        await browser.close()

    return summary_path


# ─────────────────────────────────────────────
# モデル
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    url: str
    trademark: str = ""


class GenerateRequest(BaseModel):
    session_id: str
    trademark: str


class AddAreaRequest(BaseModel):
    page_num: int
    rect: list  # [x0, y0, x1, y1] PDF座標
    trademark: str = ""


class UpdateViolationRequest(BaseModel):
    type: str = ""
    explanation: str = ""
    annotation_pos: list = []
    rect: list = []  # [x0, y0, x1, y1] PDF座標


# ─────────────────────────────────────────────
# エンドポイント
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/status")
async def api_status():
    """APIキーの設定状況を確認する診断エンドポイント。"""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    return {
        "gemini_key_set": bool(gemini_key),
        "gemini_key_preview": (gemini_key[:8] + "...") if gemini_key else "未設定",
    }


@app.get("/api/models")
async def list_models():
    """利用可能なGeminiモデルを確認する。"""
    import urllib.request
    import json as _json
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return {"error": "APIキー未設定"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = _json.loads(resp.read())
        names = [m["name"] for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
        return {"models": names}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug/{session_id}")
async def debug_session(session_id: str):
    """デバッグ用：セッションの違反データを確認する。"""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "セッションが見つかりません。")
    violations = session.get("violations", [])
    return {
        "session_id": session_id,
        "page_count": session.get("page_count"),
        "violation_count": len(violations),
        "violations_summary": [
            {
                "number": v.get("number"),
                "page_num": v.get("page_num"),
                "has_rect": v.get("rect") is not None,
                "rect": v.get("rect"),
                "text_preview": (v.get("text") or "")[:40],
            }
            for v in violations
        ],
    }


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    import sys
    print(f"[analyze] URL={request.url}", flush=True, file=sys.stderr)
    if not validate_url(request.url):
        raise HTTPException(400, "無効なURLです。http/https のURLを入力してください。")

    session_id = str(uuid.uuid4())
    pdf_path = f"/tmp/{session_id}_orig.pdf"

    try:
        try:
            await url_to_pdf(request.url, pdf_path)
        except Exception as e:
            err = str(e)
            if "net::" in err or "ERR_" in err:
                raise HTTPException(400, f"URLにアクセスできませんでした。URLが正しいか確認してください。（{err}）")
            if "Timeout" in err or "timeout" in err:
                raise HTTPException(400, "ページの読み込みがタイムアウトしました。時間をおいて再試行してください。")
            raise HTTPException(400, f"ページの読み込みに失敗しました: {err}")

        text = extract_text_from_pdf(pdf_path)
        if len(text.strip()) < 80:
            raise HTTPException(400, "このページはテキストを抽出できませんでした。Cloudflare等のボット対策で保護されている可能性があります。")

        analysis = await analyze_with_claude(text, request.url, request.trademark)

        violations_raw = analysis.get("violations", [])
        positions = find_violation_positions(pdf_path, violations_raw)

        doc = fitz.open(pdf_path)
        page_count = len(doc)
        page_dims = [{"w": p.rect.width, "h": p.rect.height} for p in doc]
        doc.close()

        sessions[session_id] = {
            "pdf_path": pdf_path,
            "url": request.url,
            "article_title": analysis.get("article_title", "記事"),
            "trademark": analysis.get("trademark", request.trademark or "商標"),
            "overall_tone": analysis.get("overall_tone", ""),
            "primary_claims": analysis.get("primary_claims", []),
            "human_review_notes": analysis.get("human_review_notes", []),
            "violations": positions,
            "page_count": page_count,
            "page_dims": page_dims,
            "created_at": time.time(),
        }

        sort_and_renumber(positions)
        background_tasks.add_task(_schedule_cleanup, session_id, SESSION_TTL)

        return {
            "session_id": session_id,
            "trademark": sessions[session_id]["trademark"],
            "article_title": sessions[session_id]["article_title"],
            "overall_tone": sessions[session_id]["overall_tone"],
            "primary_claims": sessions[session_id]["primary_claims"],
            "human_review_notes": sessions[session_id]["human_review_notes"],
            "violations": positions,
            "page_count": page_count,
            "page_dims": page_dims,
        }

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError:
                pass
        raise HTTPException(500, f"処理中にエラーが発生しました: {e}")


@app.get("/api/preview/{session_id}/{page_num}")
async def preview_page(session_id: str, page_num: int):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "セッションが見つかりません。")

    pdf_path = session["pdf_path"]
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "PDFが見つかりません。")

    doc = fitz.open(pdf_path)
    if page_num < 0 or page_num >= len(doc):
        doc.close()
        raise HTTPException(404, "ページが見つかりません。")

    page = doc[page_num]
    pdf_w = page.rect.width
    pdf_h = page.rect.height
    mat = fitz.Matrix(1.5, 1.5)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img_bytes = pix.tobytes("png")
    doc.close()

    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={
            "X-Pdf-Width": str(pdf_w),
            "X-Pdf-Height": str(pdf_h),
            "Cache-Control": "no-cache",
            "Access-Control-Expose-Headers": "X-Pdf-Width, X-Pdf-Height",
        },
    )


@app.delete("/api/session/{session_id}/violation/{idx}")
async def delete_violation(session_id: str, idx: int):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "セッションが見つかりません。")

    violations = session["violations"]
    if idx < 0 or idx >= len(violations):
        raise HTTPException(404, "指摘が見つかりません。")

    violations.pop(idx)
    sort_and_renumber(violations)

    return {"ok": True, "violations": violations}


@app.patch("/api/session/{session_id}/violation/{idx}")
async def update_violation(session_id: str, idx: int, request: UpdateViolationRequest):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "セッションが見つかりません。")
    violations = session["violations"]
    if idx < 0 or idx >= len(violations):
        raise HTTPException(404, "指摘が見つかりません。")
    if request.type:
        violations[idx]["type"] = request.type
    if request.explanation:
        violations[idx]["explanation"] = request.explanation
    if request.annotation_pos and len(request.annotation_pos) == 2:
        violations[idx]["annotation_pos"] = request.annotation_pos
    if request.rect and len(request.rect) == 4:
        violations[idx]["rect"] = request.rect
    return {"ok": True, "violation": violations[idx]}


@app.post("/api/session/{session_id}/add")
async def add_area(session_id: str, request: AddAreaRequest):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "セッションが見つかりません。")

    pdf_path = session["pdf_path"]
    doc = fitz.open(pdf_path)
    if request.page_num < 0 or request.page_num >= len(doc):
        doc.close()
        raise HTTPException(404, "ページが見つかりません。")

    page = doc[request.page_num]
    clip = fitz.Rect(*request.rect)
    area_text = page.get_text("text", clip=clip).strip() or "（テキストなし）"
    doc.close()

    trademark = request.trademark or session.get("trademark", "")
    ai_result = await analyze_area_with_ai(area_text, trademark)

    violations = session["violations"]
    new_v = {
        "page_num": request.page_num,
        "rect": request.rect,
        "number": len(violations) + 1,
        "type": ai_result["type"],
        "explanation": ai_result["explanation"],
        "text": area_text,
        "annotation_pos": None,
    }
    violations.append(new_v)
    sort_and_renumber(violations)

    return {"ok": True, "violation": new_v, "violations": violations}


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, "セッションが見つかりません（1時間で自動削除されます）。再度「解析する」を押してください。")

    pdf_path = session["pdf_path"]
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "元PDFが存在しません。再度「解析する」を押してください。")

    try:
        violations = session["violations"]
        trademark = request.trademark or session.get("trademark", "商標")
        article_title = session.get("article_title", "記事")

        final_doc = build_annotated_pdf(pdf_path, violations)

        final_path = f"/tmp/{request.session_id}_final.pdf"
        final_doc.save(final_path)
        final_doc.close()

        session["output_path"] = final_path

        with open(final_path, "rb") as f:
            pdf_bytes = f.read()

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=\"removal-report.pdf\""},
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"PDF生成中にエラーが発生しました: {e}")


# ─────────────────────────────────────────────
# 起動
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
