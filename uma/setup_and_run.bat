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

echo [1/2] 必要なライブラリをインストール中...
pip install -r "%~dp0requirements.txt" --quiet
if %errorlevel% neq 0 (
    echo [警告] 一部ライブラリのインストールに失敗しました。手動で確認してください。
)

echo [2/2] Image Stitcher を起動中...
echo.
python "%~dp0image_stitcher.py"
