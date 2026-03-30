"""
画像スティッチングツール - Image Stitcher
===========================================
機能1: HTMLファイルを直接選択 → フルページ画像として保存
機能2: 複数のスクリーンショットを重複部分を自動検出して1枚のPNGに合成

使い方:
  - HTMLから画像化: 「HTML → 画像」ボタンでHTMLファイルを選択
  - スクショ合成: 画像を追加 → 並べ替え → 「合成して保存」

対応形式: HTML, PNG, JPG, JPEG, BMP, TIFF
"""

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import tempfile
import subprocess
from pathlib import Path


# ============================================================
#  HTML → 画像変換エンジン
# ============================================================

class HtmlRenderer:
    """HTMLファイルをヘッドレスブラウザでフルページ画像に変換する"""

    @staticmethod
    def check_playwright() -> tuple[bool, str]:
        """Playwrightとブラウザの利用可否を確認"""
        try:
            from playwright.sync_api import sync_playwright
            return True, "OK"
        except ImportError:
            return False, "playwright未インストール: pip install playwright"

    @staticmethod
    def render_html_to_png(html_path: str, output_path: str,
                           width: int = 1200,
                           progress_callback=None) -> dict:
        """
        HTMLファイルをフルページPNGに変換する。

        Args:
            html_path: HTMLファイルのパス
            output_path: 出力PNGファイルのパス
            width: ビューポート幅 (px)
            progress_callback: (message: str) を受け取る関数

        Returns:
            {"width": int, "height": int, "path": str}
        """
        from playwright.sync_api import sync_playwright

        if progress_callback:
            progress_callback("ブラウザを起動中...")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": 800})

            if progress_callback:
                progress_callback("HTMLファイルを読み込み中...")

            file_url = Path(html_path).resolve().as_uri()
            page.goto(file_url, wait_until="networkidle")

            # ページの実際のサイズを取得
            page.wait_for_timeout(500)

            if progress_callback:
                progress_callback("フルページスクリーンショットを撮影中...")

            page.screenshot(path=output_path, full_page=True)

            # サイズを取得
            dimensions = page.evaluate("""() => {
                return {
                    width: Math.max(
                        document.body.scrollWidth,
                        document.documentElement.scrollWidth
                    ),
                    height: Math.max(
                        document.body.scrollHeight,
                        document.documentElement.scrollHeight
                    )
                }
            }""")

            browser.close()

            if progress_callback:
                progress_callback("完了")

            return {
                "width": dimensions["width"],
                "height": dimensions["height"],
                "path": output_path
            }


# ============================================================
#  Core: 画像スティッチングエンジン
# ============================================================

class ImageStitcher:
    """重複領域を自動検出して縦方向に画像を合成する"""

    @staticmethod
    def find_overlap(img_top: np.ndarray, img_bottom: np.ndarray,
                     min_overlap: int = 20, max_overlap_ratio: float = 0.6) -> int:
        """
        2つの画像の縦方向の重複領域をテンプレートマッチングで検出する。
        img_top の下部と img_bottom の上部が重なっている想定。

        Returns:
            重複ピクセル数（0 = 重複なし）
        """
        h_top, w_top = img_top.shape[:2]
        h_bot, w_bot = img_bottom.shape[:2]

        # 幅を揃える（小さい方に合わせる）
        w_min = min(w_top, w_bot)
        top_gray = cv2.cvtColor(img_top[:, :w_min], cv2.COLOR_BGR2GRAY)
        bot_gray = cv2.cvtColor(img_bottom[:, :w_min], cv2.COLOR_BGR2GRAY)

        max_overlap = int(min(h_top, h_bot) * max_overlap_ratio)
        if max_overlap < min_overlap:
            return 0

        best_score = -1
        best_overlap = 0

        # テンプレートの高さを変えながらマッチング
        for template_h in range(min_overlap, max_overlap, 4):
            template = bot_gray[:template_h, :]
            search_region = top_gray[h_top - max_overlap:, :]

            if template.shape[0] > search_region.shape[0]:
                break
            if template.shape[1] != search_region.shape[1]:
                break

            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score and max_val > 0.85:
                best_score = max_val
                actual_overlap = template_h + (search_region.shape[0] - template_h - max_loc[1])
                best_overlap = actual_overlap

        # 精密検索: best付近を1px刻みで再探索
        if best_overlap > 0:
            search_start = max(min_overlap, best_overlap - 10)
            search_end = min(max_overlap, best_overlap + 10)
            for template_h in range(search_start, search_end):
                template = bot_gray[:template_h, :]
                search_region = top_gray[h_top - max_overlap:, :]
                if template.shape[0] > search_region.shape[0]:
                    break
                result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    best_overlap = template_h + (search_region.shape[0] - template_h - max_loc[1])

        return best_overlap if best_score > 0.85 else 0

    @staticmethod
    def find_overlap_score(img_top: np.ndarray, img_bottom: np.ndarray,
                           min_overlap: int = 20,
                           max_overlap_ratio: float = 0.6) -> tuple[int, float]:
        """
        重複ピクセル数とマッチスコアの両方を返す。
        自動並べ替え用。

        Returns:
            (重複ピクセル数, マッチスコア 0.0〜1.0)
        """
        h_top, w_top = img_top.shape[:2]
        h_bot, w_bot = img_bottom.shape[:2]
        w_min = min(w_top, w_bot)
        top_gray = cv2.cvtColor(img_top[:, :w_min], cv2.COLOR_BGR2GRAY)
        bot_gray = cv2.cvtColor(img_bottom[:, :w_min], cv2.COLOR_BGR2GRAY)
        max_overlap = int(min(h_top, h_bot) * max_overlap_ratio)
        if max_overlap < min_overlap:
            return 0, 0.0

        best_score = -1.0
        best_overlap = 0
        for template_h in range(min_overlap, max_overlap, 4):
            template = bot_gray[:template_h, :]
            search_region = top_gray[h_top - max_overlap:, :]
            if template.shape[0] > search_region.shape[0]:
                break
            if template.shape[1] != search_region.shape[1]:
                break
            result = cv2.matchTemplate(search_region, template,
                                       cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                actual_overlap = (template_h +
                                  (search_region.shape[0] - template_h - max_loc[1]))
                best_overlap = actual_overlap

        return (best_overlap, best_score) if best_score > 0.5 else (0, 0.0)

    def auto_sort(self, images: list[np.ndarray],
                  start_index: int = 0,
                  progress_callback=None) -> list[int]:
        """
        start_index の画像を先頭にして、重複スコアが最も高い順に
        画像のインデックスを並べ替える。

        Args:
            images: 画像リスト
            start_index: 先頭にする画像のインデックス
            progress_callback: (current, total) を受け取る関数

        Returns:
            並べ替え後のインデックスリスト
        """
        n = len(images)
        if n <= 1:
            return list(range(n))

        ordered = [start_index]
        remaining = set(range(n)) - {start_index}
        total_steps = n - 1

        for step in range(total_steps):
            if progress_callback:
                progress_callback(step + 1, total_steps)

            current_img = images[ordered[-1]]
            best_idx = -1
            best_score = -1.0

            for candidate in remaining:
                _, score = self.find_overlap_score(current_img,
                                                   images[candidate])
                if score > best_score:
                    best_score = score
                    best_idx = candidate

            if best_idx >= 0:
                ordered.append(best_idx)
                remaining.remove(best_idx)
            else:
                # スコアが全て低い場合、残りの中で最初のものを追加
                next_idx = min(remaining)
                ordered.append(next_idx)
                remaining.remove(next_idx)

        return ordered

    @staticmethod
    def blend_overlap(img_top: np.ndarray, img_bottom: np.ndarray,
                      overlap: int) -> np.ndarray:
        """
        重複領域をアルファブレンドで滑らかに合成する
        """
        h_top = img_top.shape[0]
        w_max = max(img_top.shape[1], img_bottom.shape[1])

        # 幅を統一（足りない分は黒で埋めて中央揃え）
        top = ImageStitcher._pad_center(img_top, w_max)
        bot = ImageStitcher._pad_center(img_bottom, w_max)

        # 非重複部分
        top_only = top[:h_top - overlap]
        bot_only = bot[overlap:]

        # 重複部分をグラデーションブレンド
        overlap_top = top[h_top - overlap:].astype(np.float32)
        overlap_bot = bot[:overlap].astype(np.float32)

        alpha = np.linspace(1, 0, overlap).reshape(-1, 1, 1)
        blended = (overlap_top * alpha + overlap_bot * (1 - alpha)).astype(np.uint8)

        return np.vstack([top_only, blended, bot_only])

    @staticmethod
    def simple_concat(img_top: np.ndarray, img_bottom: np.ndarray) -> np.ndarray:
        """重複なしの場合、単純に縦連結"""
        w_max = max(img_top.shape[1], img_bottom.shape[1])
        top = ImageStitcher._pad_center(img_top, w_max)
        bot = ImageStitcher._pad_center(img_bottom, w_max)
        return np.vstack([top, bot])

    @staticmethod
    def _pad_center(img: np.ndarray, target_w: int) -> np.ndarray:
        """画像を target_w に中央揃えし、足りない部分は黒(0,0,0)で埋める"""
        h, w = img.shape[:2]
        if w >= target_w:
            return img[:, :target_w]
        pad_total = target_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        left = np.zeros((h, pad_left, 3), dtype=np.uint8)
        right = np.zeros((h, pad_right, 3), dtype=np.uint8)
        return np.hstack([left, img, right])

    def stitch(self, images: list[np.ndarray],
               progress_callback=None) -> tuple[np.ndarray, list[dict]]:
        """
        複数の画像を縦方向にスティッチングする

        Returns:
            (合成画像, 各結合の情報リスト)
        """
        if not images:
            raise ValueError("画像がありません")
        if len(images) == 1:
            return images[0], []

        # 全画像の最大幅を取得し、事前に中央揃え＋黒埋めで統一
        global_w = max(img.shape[1] for img in images)
        images = [self._pad_center(img, global_w) for img in images]

        result = images[0].copy()
        info_list = []

        for i in range(1, len(images)):
            if progress_callback:
                progress_callback(i, len(images) - 1)

            overlap = self.find_overlap(result, images[i])
            info = {
                "pair": f"画像{i} → 画像{i+1}",
                "overlap_px": overlap,
                "method": "ブレンド合成" if overlap > 0 else "単純連結"
            }
            info_list.append(info)

            if overlap > 0:
                result = self.blend_overlap(result, images[i], overlap)
            else:
                result = self.simple_concat(result, images[i])

        return result, info_list


# ============================================================
#  GUI: ドラッグ＆ドロップ対応のGUIアプリ
# ============================================================

class StitcherApp:
    SUPPORTED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("画像スティッチャー - Image Stitcher")
        self.root.geometry("820x700")
        self.root.minsize(700, 500)
        self.root.configure(bg="#f0f0f0")

        self.image_paths: list[str] = []
        self.stitcher = ImageStitcher()

        self._setup_styles()
        self._build_ui()
        self._setup_dnd()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Meiryo UI", 14, "bold"),
                        background="#f0f0f0")
        style.configure("Info.TLabel", font=("Meiryo UI", 9),
                        background="#f0f0f0", foreground="#555")
        style.configure("Action.TButton", font=("Meiryo UI", 11, "bold"),
                        padding=(20, 10))
        style.configure("Small.TButton", font=("Meiryo UI", 9), padding=(8, 4))

    def _build_ui(self):
        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=15, pady=(15, 5))
        ttk.Label(title_frame, text="📷 画像スティッチャー",
                  style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_frame,
                  text="複数のスクリーンショットを重複部分を検出して1枚のPNGに合成",
                  style="Info.TLabel").pack(side=tk.LEFT, padx=(15, 0))

        # Drop zone + file list
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Left: file list
        list_frame = ttk.LabelFrame(main_frame, text=" 画像リスト（上から順に合成） ")
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.listbox = tk.Listbox(list_frame, font=("Meiryo UI", 10),
                                  selectmode=tk.SINGLE, activestyle="none",
                                  bg="white", relief=tk.FLAT,
                                  highlightthickness=1,
                                  highlightcolor="#1565C0")
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                  command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y, pady=5)

        # Right: buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))

        # --- HTML → 画像変換 (目立つボタン) ---
        html_frame = ttk.LabelFrame(btn_frame, text=" HTML → 画像 ")
        html_frame.pack(fill=tk.X, pady=(0, 15))
        self.html_btn = ttk.Button(
            html_frame, text="🌐 HTMLファイルを画像化",
            style="Small.TButton",
            command=self._html_to_image)
        self.html_btn.pack(fill=tk.X, padx=4, pady=(4, 2))
        # 幅設定
        self._html_width_var = tk.StringVar(value="1200")
        w_row = ttk.Frame(html_frame)
        w_row.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Label(w_row, text="幅:", font=("Meiryo UI", 8)).pack(side=tk.LEFT)
        ttk.Entry(w_row, textvariable=self._html_width_var,
                  width=6, font=("Meiryo UI", 8)).pack(side=tk.LEFT, padx=2)
        tk.Label(w_row, text="px", font=("Meiryo UI", 8)).pack(side=tk.LEFT)

        # --- 画像スティッチング用 ---
        ttk.Button(btn_frame, text="📂 ファイル追加",
                   style="Small.TButton",
                   command=self._add_files).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_frame, text="📁 フォルダ追加",
                   style="Small.TButton",
                   command=self._add_folder).pack(fill=tk.X, pady=(0, 15))
        ttk.Button(btn_frame, text="🔼 上に移動",
                   style="Small.TButton",
                   command=self._move_up).pack(fill=tk.X, pady=(0, 3))
        ttk.Button(btn_frame, text="🔽 下に移動",
                   style="Small.TButton",
                   command=self._move_down).pack(fill=tk.X, pady=(0, 15))

        # --- 並べ替え機能 ---
        sort_frame = ttk.LabelFrame(btn_frame, text=" 並べ替え ")
        sort_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Button(sort_frame, text="🔝 選択を先頭に",
                   style="Small.TButton",
                   command=self._set_as_first).pack(fill=tk.X, padx=4, pady=(4, 2))
        self.auto_sort_btn = ttk.Button(
            sort_frame, text="🔄 自動並べ替え",
            style="Small.TButton",
            command=self._auto_sort)
        self.auto_sort_btn.pack(fill=tk.X, padx=4, pady=(2, 2))
        ttk.Button(sort_frame, text="🔢 番号で並べ替え",
                   style="Small.TButton",
                   command=self._manual_reorder).pack(fill=tk.X, padx=4, pady=(2, 4))
        # ヒントラベル
        hint = tk.Label(sort_frame,
                        text="先頭画像を選択→自動並べ替え\nで重複を元に最適な順番に",
                        font=("Meiryo UI", 7), fg="#888",
                        justify=tk.LEFT, wraplength=160)
        hint.pack(padx=4, pady=(0, 4))

        ttk.Button(btn_frame, text="❌ 選択を削除",
                   style="Small.TButton",
                   command=self._remove_selected).pack(fill=tk.X, pady=(0, 3))
        ttk.Button(btn_frame, text="🗑 全て削除",
                   style="Small.TButton",
                   command=self._clear_all).pack(fill=tk.X, pady=(0, 15))

        # Preview
        preview_frame = ttk.LabelFrame(btn_frame, text=" プレビュー ")
        preview_frame.pack(fill=tk.BOTH, expand=True)
        self.preview_label = tk.Label(preview_frame, bg="#e8e8e8",
                                      text="選択で\nプレビュー",
                                      font=("Meiryo UI", 8),
                                      fg="#999")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # Drop zone hint
        self.drop_hint = tk.Label(
            list_frame,
            text="ここに画像ファイルをドラッグ＆ドロップ\nまたは「ファイル追加」ボタンで追加",
            font=("Meiryo UI", 11), fg="#aaa", bg="white",
            justify=tk.CENTER
        )
        self._update_drop_hint()

        # Bottom: action buttons + progress
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        self.progress = ttk.Progressbar(bottom_frame, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(0, 8))

        self.status_label = ttk.Label(bottom_frame, text="画像を追加してください",
                                      style="Info.TLabel")
        self.status_label.pack(side=tk.LEFT)

        self.stitch_btn = ttk.Button(
            bottom_frame, text="🔗 合成して保存",
            style="Action.TButton",
            command=self._do_stitch
        )
        self.stitch_btn.pack(side=tk.RIGHT)

    def _setup_dnd(self):
        """tkinterdnd2 が使えればドラッグ＆ドロップを有効にする"""
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
            # root を TkinterDnD に差し替え（初回のみ）
            # 既に作成済みなので、代わりにdrop_target_registerを試す
            self.listbox.drop_target_register(DND_FILES)
            self.listbox.dnd_bind('<<Drop>>', self._on_drop)
            self.dnd_available = True
        except Exception:
            self.dnd_available = False
            # DnD非対応でもファイル選択ダイアログで動作

    def _on_drop(self, event):
        """ドラッグ＆ドロップでファイルを追加"""
        files = self.root.tk.splitlist(event.data)
        for f in files:
            f = f.strip('{}')
            if Path(f).suffix.lower() in self.SUPPORTED_EXT:
                if f not in self.image_paths:
                    self.image_paths.append(f)
        self._refresh_listbox()

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="画像ファイルを選択",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("全てのファイル", "*.*")
            ]
        )
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
        self._refresh_listbox()

    def _add_folder(self):
        folder = filedialog.askdirectory(title="フォルダを選択")
        if folder:
            for f in sorted(Path(folder).iterdir()):
                if f.suffix.lower() in self.SUPPORTED_EXT:
                    path = str(f)
                    if path not in self.image_paths:
                        self.image_paths.append(path)
            self._refresh_listbox()

    def _move_up(self):
        sel = self.listbox.curselection()
        if sel and sel[0] > 0:
            i = sel[0]
            self.image_paths[i-1], self.image_paths[i] = \
                self.image_paths[i], self.image_paths[i-1]
            self._refresh_listbox()
            self.listbox.selection_set(i-1)

    def _move_down(self):
        sel = self.listbox.curselection()
        if sel and sel[0] < len(self.image_paths) - 1:
            i = sel[0]
            self.image_paths[i+1], self.image_paths[i] = \
                self.image_paths[i], self.image_paths[i+1]
            self._refresh_listbox()
            self.listbox.selection_set(i+1)

    def _remove_selected(self):
        sel = self.listbox.curselection()
        if sel:
            del self.image_paths[sel[0]]
            self._refresh_listbox()

    def _clear_all(self):
        self.image_paths.clear()
        self._refresh_listbox()

    # ----------------------------------------------------------
    #  HTML → 画像変換
    # ----------------------------------------------------------

    def _html_to_image(self):
        """HTMLファイルを選択してフルページPNGに変換する"""
        # Playwright確認
        ok, msg = HtmlRenderer.check_playwright()
        if not ok:
            messagebox.showerror("エラー", f"HTML画像化に必要な環境がありません:\n{msg}")
            return

        # HTMLファイル選択
        html_path = filedialog.askopenfilename(
            title="HTMLファイルを選択",
            filetypes=[
                ("HTMLファイル", "*.html *.htm"),
                ("全てのファイル", "*.*")
            ]
        )
        if not html_path:
            return

        # 保存先選択
        stem = Path(html_path).stem
        save_path = filedialog.asksaveasfilename(
            title="画像の保存先",
            defaultextension=".png",
            filetypes=[("PNG画像", "*.png")],
            initialfile=f"{stem}.png"
        )
        if not save_path:
            return

        # 幅取得
        try:
            width = int(self._html_width_var.get())
            if width < 100 or width > 10000:
                raise ValueError
        except ValueError:
            messagebox.showwarning("警告", "幅は100〜10000の整数で指定してください。")
            return

        self.html_btn.config(state="disabled")
        self.progress["value"] = 0
        self.progress["maximum"] = 3
        self.root.update()

        try:
            step = [0]

            def on_progress(message):
                step[0] += 1
                self.progress["value"] = step[0]
                self.status_label.config(text=f"🌐 {message}")
                self.root.update()

            result = HtmlRenderer.render_html_to_png(
                html_path, save_path, width, on_progress)

            self.progress["value"] = self.progress["maximum"]
            self.status_label.config(
                text=f"✅ HTML画像化完了: {Path(save_path).name}")

            messagebox.showinfo(
                "HTML画像化完了",
                f"HTMLファイルを画像化しました。\n\n"
                f"入力: {Path(html_path).name}\n"
                f"出力: {save_path}\n"
                f"サイズ: {result['width']} x {result['height']} px\n\n"
                f"手動スクショ不要で、正確な1枚の画像が生成されます。"
            )

        except Exception as e:
            messagebox.showerror("エラー", f"HTML画像化に失敗:\n{e}")
            self.status_label.config(text="❌ HTML画像化エラー")

        finally:
            self.html_btn.config(state="normal")

    # ----------------------------------------------------------
    #  並べ替え機能
    # ----------------------------------------------------------

    def _set_as_first(self):
        """選択した画像を先頭に移動する"""
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("情報", "先頭にしたい画像をリストから選択してください。")
            return
        i = sel[0]
        if i == 0:
            return
        path = self.image_paths.pop(i)
        self.image_paths.insert(0, path)
        self._refresh_listbox()
        self.listbox.selection_set(0)
        self.status_label.config(
            text=f"✅ 「{Path(path).name}」を先頭に移動しました")

    def _auto_sort(self):
        """
        選択中の画像を先頭にして、重複スコアで最適な順番に自動並べ替え。
        未選択の場合は現在の先頭画像を起点にする。
        """
        if len(self.image_paths) < 2:
            messagebox.showwarning("警告", "2枚以上の画像が必要です。")
            return

        sel = self.listbox.curselection()
        start_index = sel[0] if sel else 0
        start_name = Path(self.image_paths[start_index]).name

        answer = messagebox.askyesno(
            "自動並べ替え",
            f"「{start_name}」を先頭にして、\n"
            f"重複検出をもとに最適な順番に並べ替えますか？\n\n"
            f"※ 別の画像を先頭にしたい場合は、\n"
            f"   リストで選択してからこのボタンを押してください。"
        )
        if not answer:
            return

        self.auto_sort_btn.config(state="disabled")
        self.progress["value"] = 0
        self.progress["maximum"] = len(self.image_paths) - 1
        self.status_label.config(text="🔄 自動並べ替え中... 画像を解析しています")
        self.root.update()

        try:
            # 画像を読み込み
            images = []
            for p in self.image_paths:
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"読み込み失敗: {Path(p).name}")
                images.append(img)

            def on_progress(current, total):
                self.progress["value"] = current
                self.status_label.config(
                    text=f"🔄 自動並べ替え中... ({current}/{total})")
                self.root.update()

            # 自動並べ替え
            sorted_indices = self.stitcher.auto_sort(
                images, start_index, on_progress)

            # パスリストを並べ替え
            old_paths = self.image_paths.copy()
            self.image_paths = [old_paths[i] for i in sorted_indices]

            self._refresh_listbox()
            self.progress["value"] = self.progress["maximum"]

            # 結果のサマリーを表示
            summary = "並べ替え結果:\n"
            for rank, idx in enumerate(sorted_indices):
                name = Path(old_paths[idx]).name
                marker = " ← 先頭" if rank == 0 else ""
                summary += f"  {rank+1}. {name}{marker}\n"

            self.status_label.config(text="✅ 自動並べ替え完了")
            messagebox.showinfo("自動並べ替え完了", summary)

        except Exception as e:
            messagebox.showerror("エラー", f"自動並べ替えに失敗:\n{e}")
            self.status_label.config(text="❌ 自動並べ替えエラー")

        finally:
            self.auto_sort_btn.config(state="normal")

    def _manual_reorder(self):
        """番号で順番を直接指定するダイアログを表示"""
        if len(self.image_paths) < 2:
            messagebox.showwarning("警告", "2枚以上の画像が必要です。")
            return

        # サブウィンドウ
        dialog = tk.Toplevel(self.root)
        dialog.title("番号で並べ替え")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog,
                 text="各画像の表示順を番号で指定してください（1から開始）",
                 font=("Meiryo UI", 10), pady=10).pack()

        # スクロール可能なフレーム
        canvas = tk.Canvas(dialog, bg="white")
        scroll = ttk.Scrollbar(dialog, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        scroll.pack(side=tk.LEFT, fill=tk.Y)

        entries = []
        for i, p in enumerate(self.image_paths):
            row = ttk.Frame(inner)
            row.pack(fill=tk.X, pady=2, padx=5)
            ttk.Label(row, text=f"現在 {i+1}:",
                      font=("Meiryo UI", 9), width=8).pack(side=tk.LEFT)
            entry = ttk.Entry(row, width=5, font=("Meiryo UI", 10))
            entry.insert(0, str(i + 1))
            entry.pack(side=tk.LEFT, padx=(0, 10))
            ttk.Label(row, text=Path(p).name,
                      font=("Meiryo UI", 9)).pack(side=tk.LEFT)
            entries.append(entry)

        def apply_order():
            try:
                new_order = [int(e.get()) for e in entries]
                n = len(self.image_paths)
                # バリデーション
                if sorted(new_order) != list(range(1, n + 1)):
                    messagebox.showerror(
                        "入力エラー",
                        f"1から{n}までの番号を重複なく入力してください。",
                        parent=dialog)
                    return
                old = self.image_paths.copy()
                self.image_paths = [old[i - 1] for i in new_order]
                self._refresh_listbox()
                self.status_label.config(text="✅ 番号指定で並べ替え完了")
                dialog.destroy()
            except ValueError:
                messagebox.showerror(
                    "入力エラー", "数字のみ入力してください。",
                    parent=dialog)

        btn_frame_d = ttk.Frame(dialog)
        btn_frame_d.pack(pady=10)
        ttk.Button(btn_frame_d, text="適用",
                   command=apply_order).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_d, text="キャンセル",
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for i, p in enumerate(self.image_paths):
            name = Path(p).name
            self.listbox.insert(tk.END, f"  {i+1}. {name}")
        self._update_status()
        self._update_drop_hint()

    def _update_drop_hint(self):
        if self.image_paths:
            self.drop_hint.place_forget()
        else:
            self.drop_hint.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def _update_status(self):
        n = len(self.image_paths)
        if n == 0:
            self.status_label.config(text="画像を追加してください")
        elif n == 1:
            self.status_label.config(text="⚠ 2枚以上の画像を追加してください")
        else:
            self.status_label.config(text=f"✅ {n}枚の画像が追加されています")

    def _on_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        path = self.image_paths[sel[0]]
        try:
            img = Image.open(path)
            # プレビューサイズに縮小
            img.thumbnail((180, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=photo, text="")
            self.preview_label._photo = photo  # 参照保持
        except Exception:
            self.preview_label.config(image="", text="読込失敗")

    def _do_stitch(self):
        if len(self.image_paths) < 2:
            messagebox.showwarning("警告", "2枚以上の画像を追加してください。")
            return

        # 保存先を選択
        save_path = filedialog.asksaveasfilename(
            title="合成画像の保存先",
            defaultextension=".png",
            filetypes=[("PNG画像", "*.png"), ("JPEG画像", "*.jpg")],
            initialfile="stitched_output.png"
        )
        if not save_path:
            return

        self.stitch_btn.config(state="disabled")
        self.progress["value"] = 0
        self.progress["maximum"] = len(self.image_paths) - 1
        self.root.update()

        try:
            # 画像を読み込み
            images = []
            for p in self.image_paths:
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"読み込み失敗: {Path(p).name}")
                images.append(img)

            def on_progress(current, total):
                self.progress["value"] = current
                self.status_label.config(
                    text=f"合成中... ({current}/{total})")
                self.root.update()

            # スティッチング実行
            result, info_list = self.stitcher.stitch(images, on_progress)

            # 保存
            cv2.imwrite(save_path, result,
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])

            # 結果レポート
            h, w = result.shape[:2]
            report = f"合成完了！\n\n"
            report += f"出力サイズ: {w} x {h} px\n"
            report += f"保存先: {save_path}\n\n"
            for info in info_list:
                report += f"  {info['pair']}: "
                if info['overlap_px'] > 0:
                    report += f"重複 {info['overlap_px']}px ({info['method']})\n"
                else:
                    report += f"重複なし ({info['method']})\n"

            self.progress["value"] = self.progress["maximum"]
            self.status_label.config(text=f"✅ 保存完了: {Path(save_path).name}")
            messagebox.showinfo("合成完了", report)

        except Exception as e:
            messagebox.showerror("エラー", f"合成に失敗しました:\n{e}")
            self.status_label.config(text="❌ エラーが発生しました")

        finally:
            self.stitch_btn.config(state="normal")

    def run(self):
        self.root.mainloop()


# ============================================================
#  CLI モード（GUIなしで使う場合）
# ============================================================

def cli_mode():
    import argparse
    parser = argparse.ArgumentParser(
        description="画像スティッチャー: 複数画像を重複検出して1枚に合成")
    parser.add_argument("images", nargs="+", help="合成する画像ファイル（順番通り）")
    parser.add_argument("-o", "--output", default="stitched_output.png",
                        help="出力ファイルパス (default: stitched_output.png)")
    args = parser.parse_args()

    stitcher = ImageStitcher()
    images = []
    for p in args.images:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ERROR] 読み込み失敗: {p}")
            sys.exit(1)
        images.append(img)
        print(f"  読み込み: {Path(p).name} ({img.shape[1]}x{img.shape[0]})")

    print(f"\n合成中... ({len(images)}枚)")
    result, info_list = stitcher.stitch(images)

    cv2.imwrite(args.output, result, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"\n✅ 保存完了: {args.output}")
    print(f"   サイズ: {result.shape[1]} x {result.shape[0]} px")
    for info in info_list:
        overlap_str = f"重複 {info['overlap_px']}px" if info['overlap_px'] > 0 else "重複なし"
        print(f"   {info['pair']}: {overlap_str} ({info['method']})")


# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-gui"):
        cli_mode()
    else:
        app = StitcherApp()
        app.run()
