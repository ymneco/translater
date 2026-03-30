@echo off
chcp 65001 >nul 2>&1
echo ========================================
echo   Image Stitcher - セットアップ＆起動
echo ========================================
echo.

REM Pythonの存在確認
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [エラー] Pythonが見つかりません。
    echo Python 3.8以上をインストールしてください。
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] 必要なライブラリをインストール中...
pip install -r "%~dp0requirements.txt" --quiet
if %errorlevel% neq 0 (
    echo [警告] 一部ライブラリのインストールに失敗しました。手動で確認してください。
)

echo [2/3] ブラウザエンジンをインストール中 (HTML画像化用)...
python -m playwright install chromium --with-deps 2>nul
if %errorlevel% neq 0 (
    echo [注意] Playwright Chromiumのインストールに失敗しました。
    echo        HTML画像化機能は使用できませんが、画像合成機能は動作します。
)

echo [3/3] Image Stitcher を起動中...
echo.
python "%~dp0image_stitcher.py"
