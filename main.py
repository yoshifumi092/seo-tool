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
from playwright.async_api import async_playwright, Browser, Playwright
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

# Playwright ブラウザの使い回し（毎回起動しないことで4〜5秒短縮）
_pw_instance: Playwright | None = None
_browser_instance: Browser | None = None
_browser_lock = asyncio.Lock()


async def _get_browser() -> Browser:
    """グローバルブラウザインスタンスを返す（存在しない/切断済みなら再起動）。"""
    global _pw_instance, _browser_instance
    async with _browser_lock:
        if _browser_instance is None or not _browser_instance.is_connected():
            import sys as _sys
            print("[Playwright] launching new browser instance", flush=True, file=_sys.stderr)
            # 古いブラウザインスタンスを確実にクローズしてからリソース解放
            if _browser_instance is not None:
                try:
                    await _browser_instance.close()
                except Exception:
                    pass
                _browser_instance = None
            if _pw_instance is None:
                _pw_instance = await async_playwright().start()
            _browser_instance = await _pw_instance.chromium.launch(
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
        return _browser_instance



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


def _session_file(session_id: str) -> str:
    return f"/tmp/{session_id}_session.json"


def _save_session(session_id: str, data: dict) -> None:
    """セッションデータをファイルに保存（サーバー再起動後も復元できるように）。"""
    try:
        with open(_session_file(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        import sys as _sys
        print(f"[session] save error: {e}", flush=True, file=_sys.stderr)


def _load_session(session_id: str) -> dict | None:
    """メモリにない場合はファイルから復元する。"""
    path = _session_file(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # PDFファイルが実際に存在する場合のみ復元
        if data.get("pdf_path") and os.path.exists(data["pdf_path"]):
            sessions[session_id] = data
            return data
    except Exception as e:
        import sys as _sys
        print(f"[session] load error: {e}", flush=True, file=_sys.stderr)
    return None


def get_session(session_id: str) -> dict | None:
    """メモリ → ファイルの順でセッションを取得する。"""
    return sessions.get(session_id) or _load_session(session_id)


def cleanup_session(session_id: str) -> None:
    session = sessions.pop(session_id, None)
    if not session:
        session = _load_session(session_id)
    if not session:
        return
    for key in ("pdf_path", "output_path"):
        path = session.get(key)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    # セッションファイルも削除
    sf = _session_file(session_id)
    if os.path.exists(sf):
        try:
            os.remove(sf)
        except OSError:
            pass


async def _schedule_cleanup(session_id: str, delay: int) -> None:
    await asyncio.sleep(delay)
    cleanup_session(session_id)


# ─────────────────────────────────────────────
# URL → PDF
# ─────────────────────────────────────────────

async def url_to_pdf(url: str, output_path: str, text_queue: asyncio.Queue = None) -> None:
    global _browser_instance
    browser = await _get_browser()
    try:
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
    except Exception as e:
        # ブラウザが劣化している場合は破棄して次回リクエストで再起動させる
        import sys as _sys
        print(f"[Playwright] new_context failed, resetting browser: {e}", flush=True, file=_sys.stderr)
        _browser_instance = None
        raise

    try:
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

        # ネットワークアイドルを最大2秒待つ（固定5秒から短縮）
        try:
            await page.wait_for_load_state("networkidle", timeout=2000)
        except Exception:
            pass  # タイムアウトしても続行

        # アクセスブロック検出
        page_text = await page.inner_text("body")
        block_keywords = ["アクセスできません", "403", "Access Denied", "Forbidden", "blocked", "Bot detection"]
        if any(kw.lower() in page_text.lower() for kw in block_keywords) and len(page_text.strip()) < 500:
            err = HTTPException(400, "このサイトはBot対策によりアクセスがブロックされました。別のURLをお試しください。")
            if text_queue:
                await text_queue.put(err)
            raise err

        # テキストが取得できた時点でキューに送信 → Claude解析を並列スタートさせる
        if text_queue:
            await text_queue.put(page_text)

        await page.pdf(
            path=output_path,
            format="A4",
            print_background=True,
            margin={"top": "12mm", "bottom": "12mm", "left": "12mm", "right": "12mm"},
        )
    finally:
        await context.close()  # ページは閉じるがブラウザは維持


# ─────────────────────────────────────────────
# テキスト抽出
# ─────────────────────────────────────────────


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

第3段階: 以下の分類から最も適切なものを選んで分類する
  各分類の定義と典型例を必ず参照し、最も本質的な問題を表す分類を選ぶこと。
  「信用毀損リスク」はあらゆる否定表現に適用できるため安易に選ばない。
  より具体的な分類（事実誤認・根拠不明な断定・構成的ミスリード・競合誘導バイアス等）が当てはまる場合は、そちらを優先する。

  【名誉毀損リスク】
     → 対象を「詐欺師」「悪質」「犯罪的」等と直接断定、または疑問形・仮定形で実質的に断定している
     → 典型例：「〇〇は詐欺なのか」「悪質業者と言わざるを得ない」「被害者が続出」
     → 信用毀損リスクとの違い：「人格・道徳・違法性」を攻撃するなら名誉毀損リスク、「品質・能力・信頼性」なら信用毀損リスク
     → 名誉毀損として指摘する際は以下の4要素を必ず確認し、explanation/deletion_commentに反映する：
        1. 社会的評価の低下：当該表現により一般読者の対象への評価が低下するか
        2. 公益性の欠如：公人・公共の利害に関わらない個人・事業者への攻撃であるか
        3. 真実相当性の否定：執筆者が当該評価を真実と信じる相当な根拠を記事内に提示しているか
        4. 違法性阻却事由の不存在：公益目的・真実性のいずれも認められないか
     → この4要素が揃うほど削除申請の論点として強くなる。確認できた要素は指摘文に明記する

  【信用毀損リスク】
     → 事実誤認・根拠不明な断定・構成的ミスリードに当てはまらない、サービス品質・能力・実績への否定的評価
     → 典型例：「サポートが機能していない」「結果を出せる人がほとんどいない」
     → 注意：根拠なしの断定なら「根拠不明な断定」、事実の誤りなら「事実誤認」、構成による誘導なら「構成的ミスリード」を優先する

  【営業妨害】
     → 読者の購買・登録・参加行動を直接妨げる表現、または受講検討者の不安を直接煽る構成
     → 典型例：「絶対に登録してはいけない」「お金を払う前に要注意」「今すぐ退会すべき」
     → 単一の文言だけでなく、記事全体が「検討者の意思決定を阻害する構成」になっている場合も該当する

  【商標権侵害・ブランド毀損】
     → 「商標権侵害」「信用毀損」「業務妨害」の観点で整理する
     → 以下の3論点をそれぞれ独立して検討し、該当するものを指摘する
     ① 検索誘導構造：対象の商標名・人名をタイトル/URLに使って検索上位を狙い、読者を囲い込む行為
        → タイトルに名称があるだけでは弱い。「名称で検索した読者→記事経由→別サービスへ流れる構造」の有無を確認
        → 商標権者の名称を無断で集客目的に使用しているという観点で指摘する
     ② 比較誹謗型：対象を否定的に描いた上で競合・代替サービスを推奨する比較コンテンツ
        → 「〇〇より△△の方が安全」「本当におすすめはこちら」等の誘導があれば非常に強い論点
        → 「信用毀損＋業務妨害＋競合誘導」の三重構造として指摘する
     ③ ブランド毀損：記事全体の構成が対象ブランドへの信頼・社会的評価を破壊する内容になっているか
        → 記事全体を「ブランドの社会的評価を不当に低下させる構成」として指摘する

  【事実誤認・虚偽記載】
     → 最重要：執筆者が独自調査なしに断定的評価を下している点を問う（調査限界論ではなく断定の問題として指摘）
     → 客観資料（公式サイト・取材・一次情報）なしに事実として記述している表現
     → 典型例：「実績が確認できない」（調査不足を事実として断定）「料金は〇〇円」（確認根拠なし）「〇〇人が被害」（出典なし）
     → 注意：「口コミが存在しない」等の不存在主張は、証拠があれば「事実誤認」として強く指摘できる。証拠がなければ「根拠不明な断定」として指摘する

  【根拠不明な断定】
     → 出典・証拠・調査結果なしに断定的評価を下している
     → 典型例：「効果がないのは明らか」「エビデンスが一切ない」「実態は〇〇だ」（出典なし）
     → 事実誤認との違い：既知の事実と照合して誤りが疑われるなら「事実誤認」、そもそも根拠を提示していないなら「根拠不明な断定」

  【構成的ミスリード】
     → 「印象操作」という抽象語は使わない。媒体側に「これは意見です」と逃げられないよう、以下を明確に分離して指摘する
     ① 事実摘示として書かれているが根拠がない表現
        → 「〇〇は△△だ」という断定形式でありながら根拠・出典を提示していない → 事実摘示として問題である
     ② 評価表現であることを隠して事実のように見せている表現
        → 「危険」「ヤバい」「詐欺的」等の評価が、事実確認をしたかのように記述されている
     ③ 記事全体の選択的構成による誘導
        → ネガティブ情報のみ選別・ポジティブ情報を意図的に不掲載・否定的結論ありきの情報配置
        → これは「記事構成そのものが読者を誤導する設計である」として記事全体の問題として指摘する
     → 典型例：見出しだけ「〇〇は危険？」と断定的、疑問形で不安を煽る、ネガティブな口コミのみ選別引用

  【競合誘導バイアス】
     → 対象を否定した後に競合サービス・自社LINE・アフィリエイトリンクへ誘導する構造
     → 典型例：「〇〇より安全なサービスはこちら」「私が本当におすすめするのは〜」
     → 「評価記事・口コミ記事」を装いながら実質的に特定先への誘導を行っている場合は特に重大

第4段階: 記事を先頭から末尾まで段落ごとに必ずスキャンすること
  - 各段落・各見出しを1つずつ確認し、問題のある表現を見落とさない
  - 「問題なさそう」と判断した段落でも必ず全文を読んでから次へ進む

第5段階: 特に見落としやすい以下のパターンを必ず確認する
  ① タイトル・見出しが本文より強断定になっていないか（「詐欺なのか？」等の疑問形も含む）
  ② 商標名・人名をタイトル/URLに使いながら別のLINE・サービスへ誘導していないか（商標権侵害・最重要）
  ③ 口コミ・体験談を選別引用してネガティブ印象を形成していないか
  ④ 「稼げない」「怪しい」「信頼性が低い」等を証拠なく事実のように書いていないか
  ⑤ 「安全なのか」「本当に大丈夫か」等の不安を煽る表現で購入・参加を妨げていないか
  ⑥ 「〜と言われている」「〜という声がある」等の責任回避表現で実質断定していないか
     → 外部引用・口コミ引用であっても、選択・配置した執筆者に責任が生じ得る不適切な表現である
  ⑦ 比較表で対象だけ評価を低く設定していないか（恣意的な評価軸・数値）
  ⑧ 記事末尾に「おすすめ」「安全な代替」等を誘導していないか
  ⑨ 「口コミがない」「実績が見当たらない」「評判が確認できない」等の不存在断定をしていないか
     → 不存在を証拠なく断定するのは「根拠不明な断定」として指摘する
       対象の実績・口コミが実際に存在する証拠がある場合は、「事実誤認」として強く指摘する
  ⑩ 「エビデンスがない」「根拠が示されていない」「証拠がない」等の断定をしていないか
     → 客観的調査・一次情報の収集なしに断定するのは不適切な表現であり、「根拠不明な断定」として指摘する
  ⑪ 「事実無根」「でたらめ」「嘘をついている」等の直接的な虚偽断定をしていないか
  ⑫ 【構造的問題】記事全体が「否定的結論ありき」で情報を選別していないか
     → ポジティブな情報・反証が存在するにも関わらず意図的に省いている構成は、読者に誤解を与える構成である
       これは単一表現の問題ではなく記事全体の構造問題として「構成的ミスリード」として指摘する
  ⑬ 【中立性欠如】取材・問い合わせ・当事者確認なしに一方的な評価を掲載していないか
     → 対象への取材・確認なしに断定的評価を下している記事は、客観性を欠いた不適切なコンテンツである
       「ポジティブ情報の不掲載」「一方的な情報選択」として指摘することで論点が強くなる
  ⑭ 【誘導記事パターン】「評価記事」「口コミまとめ」を装いながら特定先への誘導を行っていないか
     → 中立な評価記事を装いつつ、実質的に競合サービス・特定LINEへ誘導する構成は
       読者に誤解を与える構成であり、「商標権侵害・ブランド毀損」または「競合誘導バイアス」として強い論点になる

【断定レベルの使い分け】
削除申請資料は「運営側がトラブルになりそうと感じる文章」にすることが最重要。
以下の基準で断定レベルを使い分けること：

- 断定してよい場面：記事内の表現・構成・選択が客観的に確認できる問題（「根拠の記載がない表現である」「読者に誤解を与える構成である」「不適切な表現である」）
- 「〜し得る」「〜おそれがある」を使う場面：法的効果・権利侵害の成否など最終判断が必要な場合のみ
- 避けるべき：「可能性がある」の多用。記事の構成・表現上の問題は断定的に指摘してよい

【条文参照テーブル（このテーブル以外の条文を引用することは絶対禁止・ハルシネーション厳禁）】
legal_basis フィールドは必ず以下のテーブルから1〜2件を選び記述すること。テーブルにない条文の引用・創作は絶対禁止。

【名誉毀損リスク】
  → 刑法第230条第1項：「公然と事実を摘示し、人の名誉を毀損した者は、その事実の有無にかかわらず、三年以下の拘禁刑又は五十万円以下の罰金に処する」
  → 民法第709条：「故意又は過失によって他人の権利又は法律上保護される利益を侵害した者は、これによって生じた損害を賠償する責任を負う」

【信用毀損リスク】
  → 刑法第233条：「虚偽の風説を流布し、又は偽計を用いて、人の信用を毀損し、又はその業務を妨害した者は、三年以下の拘禁刑又は五十万円以下の罰金に処する」
  → 民法第709条（同上）

【営業妨害】
  → 刑法第233条（同上）
  → 刑法第234条：「威力を用いて人の業務を妨害した者も、前条の例による」
  → 民法第709条（同上）

【商標権侵害・ブランド毀損】
  → 商標法第36条第1項：「商標権者又は専用使用権者は、自己の商標権又は専用使用権を侵害する者又は侵害するおそれがある者に対し、その侵害の停止又は予防を請求することができる」
  → 商標法第37条第1号：登録商標と同一・類似の商標を指定商品・役務に使用する行為を商標権侵害とみなす
  → 不正競争防止法第2条第1項第1号：「需要者の間に広く認識されている商品等表示と同一若しくは類似のものを使用し、他人の商品又は営業と混同を生じさせる行為」

【事実誤認・虚偽記載】
  → 景品表示法第5条第1号（優良誤認）：「商品又は役務の品質その他の内容について、実際のものよりも著しく優良であると示す表示」
  → 景品表示法第5条第2号（有利誤認）：「価格その他の取引条件について、実際のものよりも著しく有利であると一般消費者に誤認される表示」
  → 民法第709条（同上）

【根拠不明な断定】
  → 景品表示法第5条第1号（同上）
  → 民法第709条（同上）

【構成的ミスリード】
  → 景品表示法第5条第1号（同上）
  → 民法第709条（同上）

【競合誘導バイアス】
  → 不正競争防止法第2条第1項第21号（営業誹謗行為）：「競争関係にある他人の営業上の信用を害する虚偽の事実を告知し、又は流布する行為」
  → 民法第709条（同上）

【絶対禁止事項】
- 記事に書かれていない内容を推測・補完しない
- 法律上の最終結論（違法・合法の断定）はしない
- 感情的・糾弾的な表現を使わない
- テキスト内に存在しない文字列を引用しない
- 同じ趣旨の指摘を言い換えで重複させない
- 上記条文参照テーブル以外の法令・条番号を引用しない（ハルシネーション厳禁）
- explanation・deletion_comment・legal_basis のいずれにも「A類型」「B類型」「G類型」等の類型記号・類型番号を絶対に書かない
  → これらは社外に提出する文書に転用されるため、内部分類コードの混入は厳禁
  → 問題の性質を説明する場合は「名誉毀損」「信用毀損」「虚偽の事実の流布」等、法令上の言葉か平易な日本語で表現すること

【指摘しないもの・text引用禁止テキスト】
- ナビゲーション・メニュー・パンくず・タグ・カテゴリ・コピーライト
- 著者名・日付・SNSボタン単体
- 完全に中立な事実説明・導入文
- ボタン・バナー・CTAの文言単体（例：「LINEで相談する」「無料登録はこちら」「お問い合わせ」）
  → LINE誘導の問題を指摘する場合、ボタン文言ではなく「誘導先の案内文・導線の説明文」を引用すること
  → 例：「〇〇の代わりにこちらへ」「安全な相談先はこちら」等、誘導を説明している本文テキストを引用する

【抽出ルール】
- 問題性のある箇所はすべて抽出する。見落とすことは削除申請の材料を失うことと同義であるため厳禁
- 同趣旨の指摘は統合する。ただし論点が独立しているならば必ず別指摘にする
- 件数は問わない。記事の内容に応じて正直に出力する（多くても少なくても水増し・省略しない）
- 「もう十分だろう」と途中で打ち切ることは禁止。最後の段落まで必ずスキャンを完了してから出力する
- 「商標権侵害・ブランド毀損」（商標名を使った不正誘導）は必ず最初に指摘する
- legal_basis は出力できる場合のみ記載する。条文参照テーブルに明確に対応する条文がない場合は legal_basis を空文字にしてよい
  → legal_basis が空であることを理由に指摘を省略・弱めることは絶対禁止。法的根拠の有無にかかわらず問題表現はすべて指摘すること

【指摘品質の基準】
各指摘は必ず以下の4層構造で説明すること：
  文言 → 読者印象 → 問題の本質 → 侵害観点
ただし指摘文は長くする必要はない。4層を1文に凝縮してシンプルに伝えること。

【最重要：商標名利用・不正誘導パターンの検出】
このパターンは削除申請において最も強力な論点のひとつである。必ず以下の3点を独立して確認すること。

① 検索誘導構造の確認
   - タイトル・URL・見出しに商標名・人名を使用して検索流入を得ているか
   - タイトルに名称があるだけでは不十分。「名称で検索→記事→別サービスへ流れる導線」の構造を確認する
   - 確認できた場合：severity=5、「商標権侵害・ブランド毀損（検索誘導構造）」として指摘

② 比較誹謗型コンテンツの確認
   - 対象を否定的に評価した上で、競合サービス・代替サービスを推奨しているか
   - 「〇〇より安全」「本当のおすすめ」等の比較誘導があれば非常に強い論点
   - 確認できた場合：severity=5、「商標権侵害・ブランド毀損（比較誹謗型）」または「競合誘導バイアス」として指摘

③ ブランド毀損の確認
   - 記事全体の構成が対象ブランドへの信頼・社会的評価を継続的に破壊する内容になっていないか
   - 「評価記事・口コミ記事を装った組織的ブランド攻撃」に該当する場合は記事全体の問題として指摘

【deletion_comment の出力フォーマット（厳守）】
deletion_comment は必ず以下の2文構成で出力すること。条文のフルテキスト引用は絶対禁止。
条文テーブルに該当する法的根拠がある場合は必ず1文目に記載すること（積極的に書く）。該当なしの場合のみ省略可。

1文目：これは〔法令名〕第〔条番号〕（〔条名称〕）に該当する可能性がある。（法的根拠がある場合のみ。なければ省略）
2文目（法的根拠あり）：〔問題となる行為・表現・構成〕は、〔問題の本質〕にあたる。
2文目（法的根拠なし）：〔問題となる行為・表現・構成〕は、〔問題の本質〕にあたる。（この1文のみ）

出力例：
「これは不正競争防止法第2条第1項第21号（営業誹謗行為）に該当する可能性がある。対象の人名で流入した読者を記事冒頭から競合サービスへ誘導する構成は、商標名を利用した不正な集客誘導にあたる。」
「これは刑法第230条第1項（名誉毀損）に該当する可能性がある。根拠を示さずに「詐欺まがい」と断定する記述は、対象の社会的評価を不当に低下させる表現にあたる。」
「条文テーブルに該当なしの場合の例：一方的なネガティブ情報のみを選別掲載する構成は、読者に誤解を与える不適切なコンテンツにあたる。」

【出力の絶対ルール】
- 純粋なJSONのみ返すこと。```json や ``` などのコードブロックは絶対に使わない
- explanation は文言→読者印象→問題本質→侵害観点を1文に凝縮する
- deletion_comment は上記フォーマットに従い1〜2文
- reader_impression は1文
- overall_tone は1文
- 問題性のない箇所は絶対に指摘しない。水増し禁止

{{
  "article_title": "記事タイトル（原文のまま）",
  "trademark": "この記事が攻撃している商標・サービス名または人名",
  "overall_tone": "記事全体の論調（1文）",
  "violations": [
    {{
      "text": "問題のある記述（テキストから15〜60文字を一字一句そのまま引用）",
      "statement_type": "事実主張／意見論評／推測／印象操作　のいずれか",
      "type": "以下8種類から1つだけ選ぶ（記号・類型番号は絶対に使わない）：名誉毀損リスク／信用毀損リスク／営業妨害／商標権侵害・ブランド毀損／事実誤認／根拠不明な断定／構成的ミスリード／競合誘導バイアス",
      "severity": 問題の深刻度（1〜5の整数。5が最重要）,
      "confidence": "高／中／低（本文だけで問題性を説明できるか）",
      "reader_impression": "一般読者がこの表現から受ける印象（1文）",
      "explanation": "問題の核心を35文字以内の1文で。冗長な前置き不要",
      "legal_basis": "○○法第○条第○項（問題の性質）",
      "deletion_comment": "これは〔法令名〕第〔条番号〕（〔条名称〕）に該当する可能性がある。〔問題となる行為・構成〕は、〔問題の本質〕にあたる。"
    }}
  ],
  "primary_claims": ["削除申請の主論点候補（severity4〜5の要約）"],
  "top3_deep_dive": [
    {{
      "rank": 1,
      "claim_title": "最強論点のタイトル（一言で）",
      "legal_basis": "上記フォーマットに従い2文で出力（テーブル外の条文引用は絶対禁止）",
      "key_evidence": "記事内の具体的な根拠表現（原文引用）",
      "logic": "なぜこの表現が問題か。社会的評価の低下・公益性の欠如・真実相当性の否定・違法性阻却事由の不存在　の観点から因果関係を記述",
      "deletion_argument": "削除申請書に使える主張文（2〜3文で因果関係まで書く）"
    }}
  ],
  "human_review_notes": ["人間が別途確認すべきポイント"]
}}

{section_note}
{text_chunk}"""


def _fix_json_text(text: str) -> str:
    """LLMが生成しがちなJSON文法エラーを修正する。"""
    # トレーリングカンマ（}, または ,] の直前のカンマ）を除去
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text


def _try_partial_json(text: str) -> dict | None:
    """途中で切れたJSONから完結した違反オブジェクトだけ抽出して返す。"""
    result: dict = {"violations": [], "primary_claims": [], "human_review_notes": []}
    for key in ("article_title", "trademark", "overall_tone"):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            result[key] = m.group(1)

    va = text.find('"violations"')
    if va >= 0:
        arr_start = text.find('[', va)
        if arr_start >= 0:
            depth = 0
            obj_start = -1
            for i in range(arr_start, len(text)):
                c = text[i]
                if c == '{':
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0 and obj_start >= 0:
                        obj_str = _fix_json_text(text[obj_start:i + 1])
                        try:
                            obj = json.loads(obj_str)
                            if "text" in obj or "explanation" in obj:
                                result["violations"].append(obj)
                        except Exception:
                            pass
                        obj_start = -1

    if result.get("article_title") or result["violations"]:
        return result
    return None


def _parse_ai_response(response_text: str) -> dict:
    import sys as _sys
    text = response_text.strip()

    # 閉じる``` がある場合はその中を使う
    code_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if code_block:
        text = code_block.group(1).strip()
    else:
        # レスポンスが途中で切れて閉じる``` がない場合：開き``` の直後から使う
        open_block = re.search(r'```(?:json)?\s*', text)
        if open_block:
            text = text[open_block.end():].strip()

    # { から最後の } までを切り出す
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    # JSON修正を適用してからパース試行
    cleaned = _fix_json_text(text)

    for t in [cleaned, text]:
        try:
            result = json.loads(t)
            if "violations" in result:
                return result
        except Exception:
            continue

    # 標準パース失敗 → 途中切れと判断して完結した違反オブジェクトだけ救出
    partial = _try_partial_json(response_text)
    if partial and (partial.get("violations") or partial.get("article_title")):
        import sys as _sys
        print(f"[Claude] partial parse: {len(partial['violations'])} violations salvaged", flush=True, file=_sys.stderr)
        return partial

    print(f"[Claude] PARSE FAILED. Raw response (first 1000):\n{response_text[:1000]}", flush=True, file=_sys.stderr)
    raise ValueError(f"Claude response could not be parsed as JSON. Response starts with: {response_text[:200]}")


async def _call_claude_once(
    text_chunk: str,
    url: str,
    trademark_hint: str,
    section_label: str = "",
) -> dict:
    import anthropic
    import sys as _sys

    api_key = os.environ["ANTHROPIC_API_KEY"]
    prompt = _build_analysis_prompt(text_chunk, url, trademark_hint, section_label)

    system_prompt = (
        "あなたは記事削除申請準備AIです。必ず純粋なJSONのみ返すこと。コードブロック不要。\n"
        "【最重要ルール】deletion_comment は必ず以下の構成で出力すること：\n"
        "法的根拠がある場合→1文目：これは〔法令名〕第〔条番号〕（〔条名称〕）に該当する可能性がある。2文目：〔行為〕は〔問題の本質〕にあたる。\n"
        "法的根拠がない場合→1文のみ：〔行為〕は〔問題の本質〕にあたる。\n"
        "条文テーブルに該当がある場合は積極的に法条項を記載すること。\n"
        "例：「これは不正競争防止法第2条第1項第21号（営業誹謗行為）に該当する可能性がある。"
        "対象の人名で流入した読者を競合サービスへ誘導する構成は、商標名を利用した不正な集客誘導にあたる。」\n"
        "条文のフルテキスト引用は絶対禁止。legal_basis は「○○法第○条第○項（条名称）」の短い形式のみ。\n"
        "explanation・deletion_comment・legal_basis に「A類型」「B類型」等の記号を絶対に書かない。"
    )

    def _call():
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    try:
        text = await asyncio.to_thread(_call)
    except Exception as e:
        err = str(e)
        print(f"[Claude Error] {err}", flush=True, file=_sys.stderr)
        if "rate_limit" in err.lower() or "429" in err:
            raise HTTPException(429, f"Claude APIレート制限: {err[:200]}")
        if "authentication" in err.lower() or "401" in err:
            raise HTTPException(400, f"Claude APIキーエラー: {err[:200]}")
        raise HTTPException(500, f"Claude APIエラー: {err[:200]}")

    print(f"[Claude] response (first 500): {text[:500]}", flush=True, file=_sys.stderr)
    try:
        parsed = _parse_ai_response(text)
    except ValueError as e:
        raise HTTPException(500, f"AIレスポンスの解析に失敗しました（JSON形式エラー）: {str(e)[:200]}")
    print(f"[Claude] parsed: violations={len(parsed.get('violations', []))}, title={parsed.get('article_title','')[:30]}", flush=True, file=_sys.stderr)
    return parsed


async def _analyze_once(
    text_chunk: str,
    url: str,
    trademark_hint: str,
    section_label: str = "",
) -> dict:
    """Claudeで1チャンクを解析する（429は自動リトライ）。"""
    import sys as _sys

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(500, "ANTHROPIC_API_KEY を Railway の環境変数に設定してください。")

    for attempt in range(3):
        try:
            return await _call_claude_once(text_chunk, url, trademark_hint, section_label)
        except HTTPException as e:
            if e.status_code == 400:
                raise
            if e.status_code == 429:
                wait = 30
                print(f"[Claude] 429 rate limit, waiting {wait}s (attempt {attempt+1})", flush=True, file=_sys.stderr)
                await asyncio.sleep(wait)
                continue
            raise
    raise HTTPException(500, "Claude APIが繰り返しエラーになりました。しばらく待ってから再試行してください。")


async def analyze_with_claude(text: str, url: str, trademark: str = "") -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(500, "ANTHROPIC_API_KEY が設定されていません。")

    trademark_hint = (
        f"なお、ユーザーより対象商標として「{trademark}」が指定されています。"
        if trademark else ""
    )

    # キャッシュチェック（同じURLを再解析してAPIレート制限を回避）
    cache_key = (url, trademark)
    cached = _analysis_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < CACHE_TTL:
        return cached["result"]

    result = await _analyze_once(text[:20000], url, trademark_hint)

    # 結果が正常な場合のみキャッシュに保存（0件・タイトルが"記事"は失敗扱いでキャッシュしない）
    if result.get("article_title", "記事") != "記事" or result.get("violations"):
        _analysis_cache[cache_key] = {"result": result, "ts": time.time()}
    # 古いキャッシュを掃除（最大50件）。同時アクセスによるdict変更エラーを防ぐためtry/exceptで保護
    if len(_analysis_cache) > 50:
        try:
            oldest = min(_analysis_cache, key=lambda k: _analysis_cache[k]["ts"])
            del _analysis_cache[oldest]
        except (RuntimeError, KeyError):
            pass

    return result


async def analyze_area_with_ai(text: str, trademark: str = "") -> dict:
    """選択エリアのテキストをClaudeで解析して違反カテゴリと理由を返す。"""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"type": "手動追加", "explanation": "ユーザーにより手動で追加された指摘箇所"}

    trademark_hint = f"対象商標：{trademark}" if trademark else ""
    prompt = f"""以下のテキストはネガティブ記事の一部です。{trademark_hint}
削除申請・修正要求に使える客観的な指摘として、違反カテゴリと理由を特定してください。

【分析の視点】
- 事実の摘示か意見・感想かを切り分ける
- 根拠の弱い断定表現かどうかを見る
- 社会的評価の低下・営業信用毀損・業務妨害につながるかを判断する
- 感情的表現ではなく削除申請に転用できる法的表現で指摘する

テキスト：「{text[:500]}」

以下のJSON形式のみで回答してください（前置き不要）：
{{
  "type": "違反カテゴリ（名誉毀損/信用毀損・業務妨害/印象操作/虚偽・根拠なし/過大なネガティブ表現/営業妨害/寄生マーケティング/不正競合 のいずれか）",
  "explanation": "この記述が問題である具体的な理由（2〜3文。根拠の弱さ・断定の強さ・読者への印象を指摘し、削除申請に直接転用できる表現で記載してください）"
}}"""

    def _call():
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",  # 手動追加は軽量モデルで高速処理
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    try:
        response_text = await asyncio.to_thread(_call)
        parsers = [
            lambda t: json.loads(_fix_json_text(t)),
            lambda t: json.loads(_fix_json_text(re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", t).group(1))),
            lambda t: json.loads(_fix_json_text(t[t.find("{") : t.rfind("}") + 1])),
        ]
        for parser in parsers:
            try:
                r = parser(response_text)
                if "type" in r and "explanation" in r:
                    return r
            except Exception:
                continue
    except Exception as e:
        import sys as _sys
        print(f"[analyze_area] error: {e}", flush=True, file=_sys.stderr)

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
    優先順位: ① ページ上部200pt・下部80pt以外 かつ 高さ/幅が異常でない本文領域
              ② それ以外ならy座標が最も大きい（本文に最も近い）もの
    除外条件:
      - y0 < 200: ヘッダー・ナビゲーション・パンくずリスト
      - y1 > page_height - 80: フッター・コピーライト
      - 高さ < 5pt: 細すぎてUI装飾の可能性大
    """
    def _is_body(r: fitz.Rect) -> bool:
        if r.y0 < 280:            # ヘッダー・ナビ・パンくず除外（200→280に拡大）
            return False
        if r.y1 > page_height - 80:
            return False
        if (r.y1 - r.y0) < 8:    # 薄い矩形除外（5→8ptに引き上げ）
            return False
        return True

    body = [r for r in rects if _is_body(r)]
    # 本文候補がなければ、高さが十分な候補だけに絞って再試行（ナビ完全除外は諦めない）
    if not body:
        body = [r for r in rects if (r.y1 - r.y0) >= 8]
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
    # 最短でも15文字までしか短縮しない（12→15に引き上げ）
    for length in [len(normalized_text), 40, 30, 20, 15]:
        cand = normalized_text[:length].strip()
        if len(cand) < 10:
            break
        rects = page.search_for(cand, quads=False)
        if rects:
            result = _best_rect(rects, ph)
            # 本文領域外のマッチのみの場合はスキップして短縮版で再試行
            if result.y0 < 280 or (result.y1 - result.y0) < 8:
                continue
            return result

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
    try:
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
                        "legal_basis": v.get("legal_basis", ""),
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
                    "legal_basis": v.get("legal_basis", ""),
                    "deletion_comment": v.get("deletion_comment", ""),
                    "text": raw_text,
                    "auto_placed": True,
                })
                not_found_count += 1

        return positions
    finally:
        doc.close()


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


def _draw_violation_on_page(page: fitz.Page, item: dict, font_path, occupied: list = None) -> None:
    """1件の違反を指定ページに描画する（赤枠＋注釈テキストボックス）。occupied に既配置ボックスを渡すと重複回避する。"""
    r = item["rect"]
    rect = fitz.Rect(r[0], r[1], r[2], r[3]) if isinstance(r, (list, tuple)) else r
    # アルファベット記号・類型番号表記を除去（例: "D: "、"E類型"）
    _TYPE_MAP = {
        "A類型": "名誉毀損リスク", "B類型": "信用毀損リスク", "C類型": "営業妨害",
        "D類型": "商標権侵害・ブランド毀損", "E類型": "事実誤認", "F類型": "根拠不明な断定",
        "G類型": "構成的ミスリード", "H類型": "競合誘導バイアス",
        "商標権侵害・不正競争": "商標権侵害・ブランド毀損",
        "印象操作・ミスリード": "構成的ミスリード",
    }
    raw_type = item.get("type", "")
    vtype = _TYPE_MAP.get(raw_type, raw_type)
    vtype = re.sub(r'^[A-H]:\s*', '', vtype)
    # 注釈ボックスには deletion_comment（法律込みの2文）を優先表示
    explanation = item.get("deletion_comment") or item.get("explanation", "")
    legal_basis = ""  # deletion_commentに統合したため個別表示しない
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
    ann_size = item.get("annotation_size")
    if ann_size and len(ann_size) == 2:
        box_w = float(ann_size[0])
        box_h = float(ann_size[1])
    else:
        box_w = 240.0
        exp_lines = max(2, (len(explanation) + 26) // 27)
        law_lines = max(0, (len(legal_basis) + 26) // 27) if legal_basis else 0
        box_h = float(min(max(16 + (exp_lines + law_lines) * 13 + 14, 70), 220))
    chars = max(10, int(box_w / 8.5))

    ann_pos = item.get("annotation_pos")
    if ann_pos and len(ann_pos) == 2:
        ax, ay = float(ann_pos[0]), float(ann_pos[1])
    else:
        ax, ay = auto_place_annotation(expanded, page.rect, box_w, box_h, occupied=occupied)

    ax = max(0.0, min(ax, pw - box_w))
    ay = max(0.0, min(ay, ph - box_h))
    ann_rect = fitz.Rect(ax, ay, ax + box_w, ay + box_h)
    if occupied is not None:
        occupied.append(ann_rect)
    inner    = fitz.Rect(ax + 6, ay + 5, ax + box_w - 5, ay + box_h - 4)

    # ── 白背景ボックス ──
    page.draw_rect(ann_rect, color=(0.75, 0, 0), fill=(1, 1, 1), width=1.5)

    # ── テキスト挿入（方法1: insert_htmlbox） ──
    label = f"【{num}】{vtype}" if num else vtype
    inserted = False
    if hasattr(page, "insert_htmlbox"):
        law_html = (
            f'<br><span style="font-size:8pt;color:#555555;font-style:italic;">{legal_basis}</span>'
            if legal_basis else ""
        )
        html = (
            f'<span style="font-size:10pt;font-weight:bold;color:#cc0000;">{label}</span>'
            f'<br>'
            f'<span style="font-size:9pt;color:#333333;">{explanation}</span>'
            f'{law_html}'
        )
        try:
            page.insert_htmlbox(inner, html)
            inserted = True
        except Exception:
            pass

    # ── テキスト挿入（方法2: CJKフォントファイルで insert_text） ──
    if not inserted and font_path:
        try:
            page.insert_text(fitz.Point(ax + 6, ay + 14), label,
                             fontsize=10, color=(0.75, 0, 0), fontfile=font_path)
            ty, text = ay + 28, explanation
            while text and ty < ay + box_h - 5:
                page.insert_text(fitz.Point(ax + 6, ty), text[:chars],
                                 fontsize=9, color=(0.15, 0.15, 0.15), fontfile=font_path)
                text, ty = text[chars:], ty + 13
            if legal_basis and ty < ay + box_h - 5:
                law_text = legal_basis
                while law_text and ty < ay + box_h - 5:
                    page.insert_text(fitz.Point(ax + 6, ty), law_text[:chars],
                                     fontsize=8, color=(0.35, 0.35, 0.35), fontfile=font_path)
                    law_text, ty = law_text[chars:], ty + 12
            inserted = True
        except Exception:
            pass

    # ── テキスト挿入（方法3: ASCII フォールバック） ──
    if not inserted:
        page.insert_text(fitz.Point(ax + 4, ay + 13), label[:30],
                         fontsize=8, color=(0.75, 0, 0), fontname="Helv")
        page.insert_text(fitz.Point(ax + 4, ay + 25), explanation[:55],
                         fontsize=7, color=(0.15, 0.15, 0.15), fontname="Helv")
        if legal_basis:
            page.insert_text(fitz.Point(ax + 4, ay + 37), legal_basis[:60],
                             fontsize=6, color=(0.35, 0.35, 0.35), fontname="Helv")


def build_annotated_pdf(pdf_path: str, violations: list) -> fitz.Document:
    """
    グレースケール変換＋赤枠注釈を行い fitz.Document を返す。
    ポイント: 赤枠を「先に描画」してから画像を overlay=False で背面挿入する。
    これにより画像が赤枠を隠す問題を回避する。
    """
    font_path = _get_cjk_font_path()
    orig_doc = fitz.open(pdf_path)
    try:
        new_doc = fitz.open()

        for page_num in range(len(orig_doc)):
            orig_page = orig_doc[page_num]
            pw, ph = orig_page.rect.width, orig_page.rect.height
            new_page = new_doc.new_page(width=pw, height=ph)

            # ① 赤枠＋注釈テキストを先に描画（これが最前面になる）
            occupied_rects: list = []
            for item in violations:
                if item.get("page_num") == page_num and item.get("rect"):
                    _draw_violation_on_page(new_page, item, font_path, occupied=occupied_rects)

            # ② グレースケール画像を背面に挿入（overlay=False = 既存の描画の後ろ）
            mat = fitz.Matrix(2, 2)
            pix = orig_page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            new_page.insert_image(
                fitz.Rect(0, 0, pw, ph),
                pixmap=pix,
                overlay=False,  # ← 背面に挿入。赤枠が隠れない。
            )

        return new_doc
    finally:
        orig_doc.close()


def auto_place_annotation(expanded: fitz.Rect, page_rect: fitz.Rect, box_w: float, box_h: float, margin: float = 6, occupied: list = None) -> tuple:
    """赤枠の近くで最もスペースのある位置を返す（下→右→左→上の優先順）。occupied に既配置ボックスを渡すと重複回避する。"""
    pw, ph = page_rect.width, page_rect.height
    rx0, ry0, rx1, ry1 = expanded.x0, expanded.y0, expanded.x1, expanded.y1

    def clamp(ax, ay):
        return (max(0.0, min(float(ax), pw - box_w)), max(0.0, min(float(ay), ph - box_h)))

    def overlaps_any(ax, ay):
        if not occupied:
            return False
        r = fitz.Rect(ax, ay, ax + box_w, ay + box_h)
        return any(r.intersects(o) for o in occupied)

    step = box_h + margin

    # 候補を方向ごとに生成（各方向で最大5ステップずらして試みる）
    candidates = []
    for k in range(6):
        cy = ry1 + margin + k * step
        if cy + box_h <= ph:
            candidates.append(clamp(rx0, cy))          # 下
    for k in range(6):
        cy = ry0 + k * step
        if rx1 + box_w + margin <= pw and cy + box_h <= ph:
            candidates.append(clamp(rx1 + margin, cy))  # 右
    for k in range(6):
        cy = ry0 + k * step
        if rx0 - box_w - margin >= 0 and cy + box_h <= ph:
            candidates.append(clamp(rx0 - box_w - margin, cy))  # 左
    for k in range(6):
        cy = ry0 - box_h - margin - k * step
        if cy >= 0:
            candidates.append(clamp(rx0, cy))           # 上

    for pos in candidates:
        if not overlaps_any(*pos):
            return pos

    # 全候補が重複する場合は最初の候補（下）を返す
    if candidates:
        return candidates[0]
    return clamp(rx0, ph - box_h - 5)


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
    annotation_size: list = []  # [width, height] PDF座標
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
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {
        "status": "ok",
        "anthropic_key_set": bool(anthropic_key),
        "anthropic_key_preview": (anthropic_key[:8] + "...") if anthropic_key else "未設定",
    }


@app.get("/api/debug/{session_id}")
async def debug_session(session_id: str):
    """デバッグ用：セッションの違反データを確認する。"""
    session = get_session(session_id)
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

    claude_task = None
    try:
        # ── 並列処理：ページ取得とClaude解析を重ねて実行 ──
        text_queue = asyncio.Queue()

        async def run_playwright():
            try:
                await url_to_pdf(request.url, pdf_path, text_queue)
            except Exception as e:
                if text_queue.empty():
                    await text_queue.put(e)
                raise

        pdf_task = asyncio.create_task(run_playwright())

        # テキストが届くまで待つ（ページ読み込み完了のタイミング）
        try:
            text_or_err = await asyncio.wait_for(text_queue.get(), timeout=60)
        except asyncio.TimeoutError:
            pdf_task.cancel()
            try:
                await pdf_task
            except (asyncio.CancelledError, Exception):
                pass
            raise HTTPException(400, "ページの読み込みがタイムアウトしました。時間をおいて再試行してください。")

        if isinstance(text_or_err, BaseException):
            # pdf_taskは既に失敗済み。awaitしてcleanupを確実に完了させる
            try:
                await pdf_task
            except Exception:
                pass
            # エラー種別に応じて適切なHTTPExceptionに変換する
            if isinstance(text_or_err, HTTPException):
                raise text_or_err
            err = str(text_or_err)
            if "net::" in err or "ERR_" in err:
                raise HTTPException(400, f"URLにアクセスできませんでした。（{err}）")
            raise HTTPException(400, f"ページの読み込みに失敗しました: {err}")

        text = text_or_err
        if len(text.strip()) < 80:
            pdf_task.cancel()
            try:
                await pdf_task
            except (asyncio.CancelledError, Exception):
                pass
            raise HTTPException(400, "このページはテキストを抽出できませんでした。Cloudflare等のボット対策で保護されている可能性があります。")

        # Claude解析をPDF生成と並列スタート
        claude_task = asyncio.create_task(
            analyze_with_claude(text, request.url, request.trademark)
        )

        # PDF生成の完了を待つ
        try:
            await pdf_task
        except HTTPException:
            raise
        except Exception as e:
            err = str(e)
            if "net::" in err or "ERR_" in err:
                raise HTTPException(400, f"URLにアクセスできませんでした。（{err}）")
            raise HTTPException(400, f"ページの読み込みに失敗しました: {err}")

        # Claude解析の完了を待つ（PDF生成中に並列実行済み）
        analysis = await claude_task
        claude_task = None

        violations_raw = analysis.get("violations", [])
        positions = find_violation_positions(pdf_path, violations_raw)

        doc = fitz.open(pdf_path)
        try:
            page_count = len(doc)
            page_dims = [{"w": p.rect.width, "h": p.rect.height} for p in doc]
        finally:
            doc.close()

        session_data = {
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
        sessions[session_id] = session_data
        sort_and_renumber(positions)
        _save_session(session_id, session_data)
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
        if claude_task and not claude_task.done():
            claude_task.cancel()
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError:
                pass
        raise
    except Exception as e:
        if claude_task and not claude_task.done():
            claude_task.cancel()
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError:
                pass
        raise HTTPException(500, f"処理中にエラーが発生しました: {e}")


@app.get("/api/preview/{session_id}/{page_num}")
async def preview_page(session_id: str, page_num: int):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "セッションが見つかりません。")

    pdf_path = session["pdf_path"]
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "PDFが見つかりません。")

    doc = fitz.open(pdf_path)
    try:
        if page_num < 0 or page_num >= len(doc):
            raise HTTPException(404, "ページが見つかりません。")

        page = doc[page_num]
        pdf_w = page.rect.width
        pdf_h = page.rect.height
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img_bytes = pix.tobytes("png")
    finally:
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
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "セッションが見つかりません。")

    violations = session["violations"]
    if idx < 0 or idx >= len(violations):
        raise HTTPException(404, "指摘が見つかりません。")

    violations.pop(idx)
    sort_and_renumber(violations)
    _save_session(session_id, session)

    return {"ok": True, "violations": violations}


@app.patch("/api/session/{session_id}/violation/{idx}")
async def update_violation(session_id: str, idx: int, request: UpdateViolationRequest):
    session = get_session(session_id)
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
    if request.annotation_size and len(request.annotation_size) == 2:
        violations[idx]["annotation_size"] = request.annotation_size
    if request.rect and len(request.rect) == 4:
        violations[idx]["rect"] = request.rect
    _save_session(session_id, session)
    return {"ok": True, "violation": violations[idx]}


@app.post("/api/session/{session_id}/add")
async def add_area(session_id: str, request: AddAreaRequest):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "セッションが見つかりません。")

    pdf_path = session["pdf_path"]
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "元PDFが存在しません。再度「解析する」を押してください。")
    doc = fitz.open(pdf_path)
    try:
        if request.page_num < 0 or request.page_num >= len(doc):
            raise HTTPException(404, "ページが見つかりません。")
        page = doc[request.page_num]
        clip = fitz.Rect(*request.rect)
        area_text = page.get_text("text", clip=clip).strip() or "（テキストなし）"
    finally:
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
    _save_session(session_id, session)

    return {"ok": True, "violation": new_v, "violations": violations}


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    session = get_session(request.session_id)
    if not session:
        raise HTTPException(404, "セッションが見つかりません（1時間で自動削除されます）。再度「解析する」を押してください。")

    pdf_path = session["pdf_path"]
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "元PDFが存在しません。再度「解析する」を押してください。")

    try:
        violations = session["violations"]
        article_title = session.get("article_title", "記事")

        final_doc = build_annotated_pdf(pdf_path, violations)
        final_path = f"/tmp/{request.session_id}_final.pdf"
        try:
            final_doc.save(final_path)
        finally:
            final_doc.close()

        session["output_path"] = final_path
        _save_session(request.session_id, session)

        with open(final_path, "rb") as f:
            pdf_bytes = f.read()

        # ファイル名に記事タイトルを使用（安全な文字のみ）
        safe_title = clean_filename(article_title)
        filename = f"removal-{safe_title}.pdf"

        from urllib.parse import quote
        filename_encoded = quote(filename, safe="-_.~")
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename_encoded}"},
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
