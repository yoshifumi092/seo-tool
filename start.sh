#!/bin/bash
# SEO削除申請ツール 起動スクリプト

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SEO削除申請ツール ==="
echo ""

# Python仮想環境チェック
if [ ! -d ".venv" ]; then
  echo ">>> 仮想環境を作成しています..."
  python3 -m venv .venv
fi

source .venv/bin/activate

# 依存パッケージのインストール
echo ">>> 依存パッケージを確認しています..."
pip install -q -r requirements.txt

# Playwrightブラウザのインストール
echo ">>> Playwright (Chromium) を確認しています..."
playwright install chromium --with-deps 2>/dev/null || playwright install chromium

# .envファイルの読み込み
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

# APIキーチェック
if [ -z "$GROQ_API_KEY" ]; then
  echo ""
  echo "エラー: GROQ_API_KEY が設定されていません。"
  echo "  setup_key.py を実行してAPIキーを設定してください。"
  echo ""
  exit 1
fi

echo ""
echo ">>> サーバーを起動しています..."
echo ">>> ブラウザで http://localhost:8000 を開いてください"
echo ">>> 停止するには Ctrl+C を押してください"
echo ""

python main.py
