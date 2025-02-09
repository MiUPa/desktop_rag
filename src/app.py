import os
import tkinter as tk
from tkinter import filedialog
from transformers import AutoTokenizer, AutoModelForCausalLM
import fitz  # PyMuPDFをインポート
from llm_model import LLMModel

class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAGアプリ")
        self.pdf_files = []
        
        # DeepSeekモデルのパスを指定
        retrieval_model_path = "path/to/deepseek/retrieval_model"  # 適切なパスに変更
        generation_model_path = "path/to/deepseek/generation_model"  # 適切なパスに変更
        self.llm_model = LLMModel(retrieval_model_path, generation_model_path)  # モデルのインスタンスを作成

        self.add_pdf_button = tk.Button(root, text="PDFを追加", command=self.add_pdf)
        self.add_pdf_button.pack()

        self.query_entry = tk.Entry(root)
        self.query_entry.pack()

        self.query_button = tk.Button(root, text="検索", command=self.search)
        self.query_button.pack()

        self.result_text = tk.Text(root)
        self.result_text.pack()

    def add_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.pdf_files.append(file_path)
            self.result_text.insert(tk.END, f"追加されたPDF: {os.path.basename(file_path)}\n")

    def search(self):
        query = self.query_entry.get()
        context = self.extract_context_from_pdfs()  # PDFからコンテキストを抽出
        answer = self.llm_model.generate_answer(query, context)
        self.result_text.insert(tk.END, f"AIの回答: {answer}\n")

    def extract_context_from_pdfs(self):
        context = ""
        for pdf_file in self.pdf_files:
            with fitz.open(pdf_file) as doc:
                for page in doc:
                    context += page.get_text()  # ページからテキストを抽出
        return context

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()
