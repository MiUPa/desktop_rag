import os
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, font
import fitz  # PyMuPDFをインポート
import glob
import traceback
import threading
import time
from src.llm_model import LLMModel

class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("日本語RAGアプリ")
        self.root.geometry("800x600")  # ウィンドウサイズ設定
        self.pdf_files = []
        
        # ストリーミング制御用の変数
        self.streaming_active = False
        self.streaming_thread = None
        
        # モデルパスの設定
        retrieval_model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        # ローカルモデルパスの確認
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        generation_model_path = os.path.join(models_dir, "japanese-gpt2-medium")
        
        if not os.path.exists(generation_model_path):
            generation_model_path = "rinna/japanese-gpt2-medium"  # フォールバック
        
        print(f"情報検索モデル: {retrieval_model_path}")
        print(f"文章生成モデル: {generation_model_path}")
        
        try:
            # モデルの初期化
            self.llm_model = LLMModel(retrieval_model_path, generation_model_path)
            
            # UI要素の配置
            self.create_widgets()
            
            # data/documentsディレクトリからPDFを読み込む
            self.load_pdfs_from_documents()
            
            # 起動時にREADMEを表示
            self.show_readme()
            
        except Exception as e:
            messagebox.showerror("初期化エラー", f"モデルの初期化中にエラーが発生しました:\n{str(e)}")

    def create_widgets(self):
        # フレームの作成
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # PDFを追加ボタン
        self.add_pdf_button = tk.Button(top_frame, text="PDFを追加", command=self.add_pdf)
        self.add_pdf_button.pack(side=tk.LEFT, padx=5)
        
        # PDF一覧をクリアするボタン
        self.clear_pdf_button = tk.Button(top_frame, text="PDF一覧クリア", command=self.clear_pdfs)
        self.clear_pdf_button.pack(side=tk.LEFT, padx=5)
        
        # data/documentsから再読込するボタン
        self.reload_docs_button = tk.Button(top_frame, text="documentsから読込", command=self.load_pdfs_from_documents)
        self.reload_docs_button.pack(side=tk.LEFT, padx=5)
        
        # PDFリスト表示ラベル
        pdf_label = tk.Label(self.root, text="追加されたPDF:")
        pdf_label.pack(anchor=tk.W, padx=10)
        
        # PDFリスト表示エリア
        self.pdf_list_text = scrolledtext.ScrolledText(self.root, width=80, height=5)
        self.pdf_list_text.pack(fill=tk.X, padx=10, pady=5)
        
        # 質問入力フレーム
        query_frame = tk.Frame(self.root)
        query_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 質問ラベル
        query_label = tk.Label(query_frame, text="質問:", font=('Helvetica', 12, 'bold'))
        query_label.pack(side=tk.LEFT, padx=5)
        
        # 質問入力欄（例を含む）- 視認性向上
        custom_font = font.Font(family="Helvetica", size=12)
        self.query_entry = tk.Entry(query_frame, width=50, font=custom_font, bg="#F0F8FF")
        self.query_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.query_entry.insert(0, "例: 統計検定1級の試験範囲は何ですか？")
        self.query_entry.config(fg="gray")
        
        # 例のテキストをクリアするためのフォーカスイベント
        def on_entry_focus_in(event):
            if self.query_entry.get() == "例: 統計検定1級の試験範囲は何ですか？":
                self.query_entry.delete(0, tk.END)
                self.query_entry.config(fg="black")
        
        def on_entry_focus_out(event):
            if self.query_entry.get() == "":
                self.query_entry.insert(0, "例: 統計検定1級の試験範囲は何ですか？")
                self.query_entry.config(fg="gray")
                
        self.query_entry.bind("<FocusIn>", on_entry_focus_in)
        self.query_entry.bind("<FocusOut>", on_entry_focus_out)
        self.query_entry.bind("<Return>", lambda event: self.search())
        
        # 検索ボタン
        self.query_button = tk.Button(query_frame, text="質問する", font=('Helvetica', 11), 
                                      bg="#4CAF50", fg="white", command=self.search)
        self.query_button.pack(side=tk.RIGHT, padx=5)
        
        # 結果表示ラベル
        result_label = tk.Label(self.root, text="回答:", font=('Helvetica', 12, 'bold'))
        result_label.pack(anchor=tk.W, padx=10)
        
        # 結果表示領域（スクロール可能なテキストエリア）- 視認性向上
        result_font = font.Font(family="Helvetica", size=11)
        self.result_text = scrolledtext.ScrolledText(self.root, width=80, height=20, 
                                                    font=result_font, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # スタイル設定
        self.result_text.tag_configure("question", font=('Helvetica', 11, 'bold'))
        self.result_text.tag_configure("answer", font=('Helvetica', 11))
        self.result_text.tag_configure("loading", foreground="blue", font=('Helvetica', 11, 'italic'))
        self.result_text.tag_configure("error", foreground="red", font=('Helvetica', 11, 'bold'))

    def load_pdfs_from_documents(self):
        """data/documentsディレクトリからPDFファイルを読み込む"""
        # PDFファイル一覧をクリア
        self.pdf_files = []
        
        # data/documentsディレクトリのパス
        documents_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "documents")
        
        if os.path.exists(documents_dir):
            # PDFファイルを検索
            pdf_paths = glob.glob(os.path.join(documents_dir, "*.pdf"))
            
            # 見つかったPDFを追加
            for pdf_path in pdf_paths:
                self.pdf_files.append(pdf_path)
            
            # PDF一覧を更新
            self.update_pdf_list()
            
            if pdf_paths:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"{len(pdf_paths)}件のPDFを data/documents から読み込みました。\n")
                for i, path in enumerate(pdf_paths, 1):
                    filename = os.path.basename(path)
                    self.result_text.insert(tk.END, f"{i}. {filename}\n")
            else:
                messagebox.showinfo("情報", "data/documentsディレクトリにPDFファイルが見つかりませんでした。")
        else:
            messagebox.showwarning("警告", "data/documentsディレクトリが見つかりません。")

    def show_readme(self):
        """READMEファイルの内容を表示する"""
        readme_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
            self.result_text.delete(1.0, tk.END)  # 既存のテキストをクリア
            self.result_text.insert(tk.END, "=== READMEプレビュー ===\n\n")
            self.result_text.insert(tk.END, readme_content)
            self.result_text.see(1.0)  # スクロールを先頭に戻す

    def add_pdf(self):
        """PDFファイルを追加する"""
        file_paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
        for file_path in file_paths:
            if file_path and file_path not in self.pdf_files:
                self.pdf_files.append(file_path)
        
        # PDF一覧を更新
        self.update_pdf_list()

    def clear_pdfs(self):
        """追加されたPDFファイル一覧をクリアする"""
        self.pdf_files = []
        self.update_pdf_list()
        messagebox.showinfo("情報", "PDF一覧をクリアしました")

    def update_pdf_list(self):
        """PDF一覧表示を更新する"""
        self.pdf_list_text.delete(1.0, tk.END)
        if not self.pdf_files:
            self.pdf_list_text.insert(tk.END, "PDFがまだ追加されていません。「PDFを追加」ボタンをクリックしてください。")
        else:
            for i, pdf_file in enumerate(self.pdf_files, 1):
                filename = os.path.basename(pdf_file)
                self.pdf_list_text.insert(tk.END, f"{i}. {filename}\n")

    def stream_answer(self, query, context):
        """回答をストリーミングで表示する処理"""
        try:
            # 処理開始を表示
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"質問: ", "question")
            self.result_text.insert(tk.END, f"{query}\n\n", "question")
            self.result_text.insert(tk.END, f"回答: \n", "answer")
            
            # カーソル位置を保存
            answer_start_position = self.result_text.index(tk.END)
            self.result_text.mark_set("answer_start", answer_start_position)
            
            # ストリーミング開始のフラグを設定
            self.streaming_active = True
            
            # コールバック関数を定義
            def on_token(token):
                if self.streaming_active:
                    # UIスレッドで実行する必要がある
                    def update_ui():
                        self.result_text.insert(tk.END, token, "answer")
                        self.result_text.see(tk.END)  # スクロールして最新部分を表示
                    
                    # UIスレッドでの実行をスケジュール
                    self.root.after(1, update_ui)
            
            # llm_modelにself.streaming_activeへの参照を渡す
            self.llm_model.streaming_active = self.streaming_active
            
            # 生成処理を開始
            answer = self.llm_model.generate_answer_streaming(query, context, callback=on_token)
            
            # ストリーミングが終了したことを示す
            self.streaming_active = False
            
            # 最終的な回答が返されたら、表示を更新
            def finalize_answer():
                if not answer:
                    # 回答が空の場合
                    self.result_text.insert(tk.END, "\n\n生成が完了しましたが、有効な回答が得られませんでした。", "error")
                
                # キャレット（カーソル）を文末に移動してスクロール
                self.result_text.see(tk.END)
            
            # UIスレッドでの最終更新
            self.root.after(100, finalize_answer)
            
        except Exception as e:
            # エラー情報を表示
            error_details = traceback.format_exc()
            print(f"ストリーミング中にエラーが発生しました: {error_details}")
            
            # UIスレッドでエラーメッセージを表示
            def show_error():
                self.result_text.insert(tk.END, f"\n\nエラーが発生しました: {str(e)}", "error")
            
            self.root.after(1, show_error)
            
            # ストリーミングフラグをリセット
            self.streaming_active = False

    def search(self):
        """質問に対する回答を生成する"""
        query = self.query_entry.get().strip()
        
        # 例文のままなら処理しない
        if query == "例: 統計検定1級の試験範囲は何ですか？":
            messagebox.showwarning("警告", "質問を入力してください")
            return
        
        if not query:
            messagebox.showwarning("警告", "質問を入力してください")
            return
            
        if not self.pdf_files:
            messagebox.showwarning("警告", "PDFファイルを追加してください")
            return
            
        # 前回のストリーミングがあれば停止
        if self.streaming_active and self.streaming_thread is not None:
            self.streaming_active = False
            if self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=0.5)  # スレッドを待機
        
        # 質問欄をクリア（新機能）
        original_query = query  # 質問を保存
        self.query_entry.delete(0, tk.END)
        
        # 処理中表示
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "PDFからコンテキストを抽出中...\n", "loading")
        self.root.update()
        
        try:
            # PDFからテキストを抽出
            context = self.extract_context_from_pdfs()
            if not context:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "PDFからテキストを抽出できませんでした。別のPDFファイルを試してください。", "error")
                return
            
            # 回答生成前の表示を更新
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "回答を生成中です。しばらくお待ちください...\n", "loading")
            self.root.update()
            
            # ストリーミングスレッドを開始
            self.streaming_thread = threading.Thread(
                target=self.stream_answer, 
                args=(original_query, context)
            )
            self.streaming_thread.daemon = True
            self.streaming_thread.start()
            
        except Exception as e:
            # 詳細なエラー情報を取得
            error_details = traceback.format_exc()
            print(f"エラー詳細:\n{error_details}")
            
            # ユーザーにはシンプルなエラーメッセージを表示
            messagebox.showerror("エラー", f"処理中にエラーが発生しました:\n{str(e)}")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"エラーが発生しました。\n\n可能な解決策:\n", "error")
            self.result_text.insert(tk.END, "1. 別の質問を試してください\n")
            self.result_text.insert(tk.END, "2. PDFファイルを変更してください\n")
            self.result_text.insert(tk.END, "3. アプリケーションを再起動してください\n")

    def extract_context_from_pdfs(self):
        """PDFファイルからテキストを抽出する"""
        context = ""
        success_count = 0
        
        for pdf_file in self.pdf_files:
            try:
                with fitz.open(pdf_file) as doc:
                    file_content = ""
                    for page_num, page in enumerate(doc):
                        page_text = page.get_text()
                        
                        # ページテキストが空でないか確認
                        if page_text.strip():
                            file_content += f"--- ページ {page_num+1} ---\n{page_text}\n\n"
                    
                    if file_content.strip():
                        context += f"=== ファイル: {os.path.basename(pdf_file)} ===\n{file_content}\n\n"
                        success_count += 1
                        print(f"PDFファイル '{os.path.basename(pdf_file)}' からテキストを抽出しました")
                    else:
                        print(f"PDFファイル '{os.path.basename(pdf_file)}' からテキストを抽出できませんでした")
            except Exception as e:
                print(f"PDFファイル '{os.path.basename(pdf_file)}' の読み込み中にエラーが発生しました: {e}")
        
        if success_count == 0 and self.pdf_files:
            print("すべてのPDFファイルからテキストを抽出できませんでした")
            
        # 抽出されたテキストのサンプルを表示（デバッグ用）
        if context:
            context_preview = context[:500] + "..." if len(context) > 500 else context
            print(f"抽出されたテキストサンプル: {context_preview}")
        
        return context

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()
