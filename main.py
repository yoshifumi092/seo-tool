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
            args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(2000)
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

async def analyze_with_claude(text: str, url: str, trademark: str = "") -> dict:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(500, "GROQ_API_KEY が設定されていません。")

    client = Groq(api_key=api_key)
    trademark_hint = (
        f"なお、ユーザーより対象商標として「{trademark}」が指定されています。"
        if trademark else ""
    )

    prompt = f"""あなたは日本の法務・削除申請の専門家です。
以下のマインドセットで記事を分析し、ホスティングサービスへの削除申請に使える証拠書類を作成してください。
{trademark_hint}

【対象URL】
{url}

【分析マインドセット】
- 気に入らない表現を探すのではなく、削除請求・修正要求に耐えうる客観的で使える指摘を積み上げる
- 事実の摘示なのか、筆者の意見・感想なのか、断定なのか、推測なのかを切り分ける
- 根拠の弱い断定表現（「危険」「怪しい」「稼げない」「信用できない」等の言い切り）を特に重視する
- 記事全体が読者に与える印象（ネガティブ誘導の構造）を見る
- 社会的評価の低下・営業上の信用毀損・業務妨害につながるかを判断する
- 感情的表現ではなく、削除申請・申立てに転用しやすい法的表現で指摘する
- 通りやすい論点（名誉毀損・信用毀損・業務妨害・誤認を招く断定表現）を優先する
- 主張の強弱を整理し、本丸となる強い指摘を優先する

【違反カテゴリ（現実的に主張しやすい順）】
1. 名誉毀損 - 根拠なく個人・企業の社会的評価を低下させる断定的記述
2. 信用毀損・業務妨害 - 営業・集客・採用・販売に悪影響を与える根拠不十分な記述
3. 印象操作 - 中立を装いながらネガティブ印象に誘導する構成・表現
4. 虚偽・根拠なし - 匿名情報・伝聞・出典不明を根拠にした断定表現
5. 過大なネガティブ表現 - 根拠に見合わない強い否定・侮辱的表現
6. 営業妨害 - 検索流入・比較検討段階の見込み客への悪影響が見込まれる記述
7. 寄生マーケティング - 商標・ブランド名を利用して自社サービスへ誘導する記述
8. 不正競合 - 虚偽の比較・格付けによる競合他社の不当な貶め

以下のJSON形式のみで回答してください（前置き・説明文は不要）：

{{
  "article_title": "記事のタイトル（原文）",
  "trademark": "記事で攻撃対象または不正使用されている商標・ブランド・会社名（最重要なもの1つ）",
  "violations": [
    {{
      "text": "問題のある箇所のテキスト（記事の原文から20〜80文字。検索で特定できる一意な文字列）",
      "type": "違反カテゴリ名",
      "explanation": "この記述が問題である具体的な理由（2〜3文。根拠の弱さ・断定の強さ・読者への印象・法的リスクを明確に指摘し、削除申請に直接転用できる表現で記載してください）"
    }}
  ]
}}

重要なルール：
- violationsは必ず15件以上20件以下で返してください（10件以下は絶対に不可）
- "text"は必ず記事中に実在するテキストをそのまま引用してください（変形・要約禁止）
- 強く言える指摘（根拠不十分な断定、社会的評価低下、営業信用毀損）を優先してください
- 補助的な指摘（一方的表現、中立性の欠如）は後半にまとめてください
- 15件未満で止まることは許可されていません。必ず15〜20件出力してください

【記事テキスト（最大12000文字）】
{text[:12000]}"""

    def _call_groq():
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
        )

    response = await asyncio.to_thread(_call_groq)
    response_text = response.choices[0].message.content

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

    return {"article_title": "記事", "trademark": trademark or "商標", "violations": []}


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


def _search_text_in_page(page: fitz.Page, normalized_text: str) -> fitz.Rect | None:
    """
    ページ内でテキストを検索する。
    1) search_for（PyMuPDF組み込み、空白正規化あり）
    2) get_text("blocks") を使ったPythonレベルの部分一致 → ブロックRectを返す
    """
    # ── 方法1: search_for で候補の長さを変えながら試す ──
    for length in [len(normalized_text), 80, 60, 40, 20, 12, 8]:
        cand = normalized_text[:length].strip()
        if len(cand) < 5:
            continue
        rects = page.search_for(cand, quads=False)
        if rects:
            return rects[0]

    # ── 方法2: Pythonレベルのテキストブロック部分一致 ──
    # get_text("blocks") はPDF内のテキストブロックを座標付きで返す
    for length in range(min(len(normalized_text), 40), 4, -2):
        fragment = normalized_text[:length].strip()
        if len(fragment) < 5:
            continue
        for b in page.get_text("blocks"):
            if b[6] != 0:  # テキストブロック以外はスキップ
                continue
            block_text = _normalize_search_text(b[4])
            if fragment in block_text:
                # ブロック内でさらに正確な位置を取得（短い文字列で再試行）
                precise = page.search_for(fragment[:20], quads=False)
                if precise:
                    return precise[0]
                # 正確な位置が取れなければブロックRectを使う
                return fitz.Rect(b[0], b[1], b[2], b[3])

    return None


def find_violation_positions(pdf_path: str, violations: list) -> list:
    """各違反テキストのPDF上の座標を検索する（リスト形式で返す）。"""
    doc = fitz.open(pdf_path)
    positions = []

    for i, v in enumerate(violations):
        raw_text = v.get("text", "")
        normalized = _normalize_search_text(raw_text)
        found = False

        for page_num in range(len(doc)):
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
            positions.append({
                "page_num": -1,
                "rect": None,
                "number": i + 1,
                "type": v.get("type", ""),
                "explanation": v.get("explanation", ""),
                "text": raw_text,
            })

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
        await url_to_pdf(request.url, pdf_path)

        text = extract_text_from_pdf(pdf_path)
        if len(text.strip()) < 80:
            raise HTTPException(400, "PDFからテキストを十分に抽出できませんでした。")

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
