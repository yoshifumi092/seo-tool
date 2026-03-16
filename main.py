import asyncio
import json
import os
import re
import time
import uuid
import unicodedata
from urllib.parse import urlparse

from groq import Groq
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
            ]
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="ja-JP",
        )
        page = await context.new_page()

        # まず domcontentloaded で取得（networkidle はタイムアウトしやすい）
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        except Exception:
            # それでも失敗した場合は load イベントで再試行
            await page.goto(url, wait_until="load", timeout=45000)

        # JS・画像の読み込みを少し待つ
        await page.wait_for_timeout(3000)

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
    return f"""あなたは風評被害・誹謗中傷の削除申請を専門とする法務アシスタントです。

【背景と前提】
以下は、第三者（競合業者・アフィリエイター等）が書いたネガティブSEO記事の一部です。
記事の目的は対象の商標・講師・会社の評判を傷つけ、読者を競合サービスへ誘導することです。
{trademark_hint}

【対象URL】
{url}

【指摘すべき問題記述】

① 根拠なき断定（最優先）
   - 「詐欺」「詐欺師」「悪質」「危険」「やばい」「騙された」「被害が出ている」
   - 「稼げない」「効果がない」「返金できない」「連絡が取れない」を証拠なく断定
   - 上記を含む一文・一節をそのまま引用する

② 名誉毀損
   - 人名・商標名を使って「信用できない」「問題がある」「危険人物」と断定
   - 「○○の正体」「○○の裏側」「○○の真実」など人格・経歴を根拠なく否定

③ アフィリエイト誘導
   - 対象を否定した直後に別サービスを「おすすめ」と紹介している文
   - 「○○よりも△△の方が稼げる」など比較で不当に貶めている文

④ 匿名情報の断定
   - 「〜という被害報告がある」「口コミで〜という声が多い」を根拠に悪評を断定
   - 出典のない「体験談」を事実として記載している文

⑤ 営業妨害
   - 「絶対に手を出すな」「買ってはいけない」「登録前にこれを読め」など
   - 購入・参加を妨げる強い警告文

【絶対に指摘しないもの】
- ナビゲーション・メニュー・サイト名・ロゴ・パンくず・タグ・カテゴリ名
- 著者名・日付・URL・SNSボタン・コピーライト
- 「この記事では〜を紹介します」などの中立的な導入・説明文
- 商標名・人名を単体で書いているだけの箇所（問題のある断定を伴わない場合）
- 「〜でしょうか」「〜と思います」など断定していない感想・疑問

以下のJSON形式のみで返してください（前置き・補足・コードブロック一切不要）：

{{
  "article_title": "記事タイトル（原文のまま）",
  "trademark": "この記事が攻撃している商標・サービス名または人名",
  "violations": [
    {{
      "text": "問題のある記述（下記テキストから15〜35文字を一字一句そのまま抜粋）",
      "type": "根拠なき断定／名誉毀損／アフィリエイト誘導／匿名情報断定／営業妨害　のいずれか",
      "explanation": "なぜ名誉毀損・信用毀損・業務妨害にあたるか。削除申請書に使える法的表現で2文以内。"
    }}
  ]
}}

【必ず守るルール】
1. violations は10〜15件出力する
2. "text" は下記テキストから15〜35文字を一字一句そのまま引用（改変・要約・省略・補足 禁止）
3. "text" は問題のある断定表現を含む文節にする。商標名・サービス名だけの引用は禁止
4. "text" は下記テキスト内に実際に存在する文字列のみ（存在しない文字列は作らない）
5. 絶対に指摘しないものリストは厳守する
6. ①を最優先。次に②③④⑤の順で探す

{section_note}
{text_chunk}"""


def _parse_ai_response(response_text: str) -> dict:
    parsers = [
        lambda t: json.loads(t),
        lambda t: json.loads(re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", t).group(1)),
        lambda t: json.loads(t[t.find("{") : t.rfind("}") + 1]),
    ]
    for parser in parsers:
        try:
            result = parser(response_text)
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
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )
    prompt = _build_analysis_prompt(text_chunk, url, trademark_hint, section_label)
    payload = _json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 4096},
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

    result = await asyncio.to_thread(_call)
    text = result["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_ai_response(text)


async def _call_groq_once(
    client: Groq,
    text_chunk: str,
    url: str,
    trademark_hint: str,
    section_label: str = "",
) -> dict:
    prompt = _build_analysis_prompt(text_chunk, url, trademark_hint, section_label)

    def _call():
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )

    try:
        response = await asyncio.to_thread(_call)
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            wait = re.search(r'try again in ([^.]+)', err)
            wait_msg = f"約{wait.group(1)}後に再試行できます。" if wait else "しばらく待ってから再試行してください。"
            raise HTTPException(
                429,
                f"AI解析の1日の無料利用上限に達しました。{wait_msg}"
                " または GEMINI_API_KEY を Railway に設定すると無料で続けられます（https://aistudio.google.com/app/apikey）"
            )
        raise
    return _parse_ai_response(response.choices[0].message.content)


async def _analyze_once(
    text_chunk: str,
    url: str,
    trademark_hint: str,
    section_label: str = "",
) -> dict:
    """Gemini優先・Groqフォールバックで1チャンクを解析する。"""
    if os.environ.get("GEMINI_API_KEY"):
        return await _call_gemini_once(text_chunk, url, trademark_hint, section_label)
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(500, "GEMINI_API_KEY または GROQ_API_KEY を Railway の環境変数に設定してください。")
    return await _call_groq_once(Groq(api_key=groq_key), text_chunk, url, trademark_hint, section_label)


async def analyze_with_claude(text: str, url: str, trademark: str = "") -> dict:
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(500, "GEMINI_API_KEY または GROQ_API_KEY が設定されていません。")

    trademark_hint = (
        f"なお、ユーザーより対象商標として「{trademark}」が指定されています。"
        if trademark else ""
    )

    CHUNK = 10000

    if len(text) <= CHUNK:
        return await _analyze_once(text, url, trademark_hint)

    # 長い記事：前半・後半を並列解析してマージ
    chunk1 = text[:CHUNK]
    chunk2 = text[CHUNK : CHUNK * 2]

    r1, r2 = await asyncio.gather(
        _analyze_once(chunk1, url, trademark_hint, "前半"),
        _analyze_once(chunk2, url, trademark_hint, "後半"),
    )

    violations = r1.get("violations", []) + r2.get("violations", [])
    return {
        "article_title": r1.get("article_title") or r2.get("article_title", "記事"),
        "trademark": r1.get("trademark") or r2.get("trademark", trademark or "商標"),
        "violations": violations,
    }


async def analyze_area_with_ai(text: str, trademark: str = "") -> dict:
    """選択エリアのテキストをAIで解析して違反カテゴリと理由を返す。"""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return {"type": "手動追加", "explanation": "ユーザーにより手動で追加された指摘箇所"}

    client = Groq(api_key=api_key)
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

    def _call():
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )

    try:
        response = await asyncio.to_thread(_call)
        response_text = response.choices[0].message.content
        parsers = [
            lambda t: json.loads(t),
            lambda t: json.loads(re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", t).group(1)),
            lambda t: json.loads(t[t.find("{") : t.rfind("}") + 1]),
        ]
        for parser in parsers:
            try:
                result = parser(response_text)
                if "type" in result and "explanation" in result:
                    return result
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
                    "explanation": v.get("explanation", ""),
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
                "explanation": v.get("explanation", ""),
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
    vtype = item.get("type", "")
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
