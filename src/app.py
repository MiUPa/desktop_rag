import os
import tkinter as tk
from tkinter import filedialog
from pdf_parser import extract_text_from_pdf
from ranking_model import rank_results
from llm_model import LLMModel

class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAGアプリ")
        self.pdf_files = []
        self.llm_model = LLMModel()  # GPT-2モデルのインスタンスを作成

        self.add_pdf_button = tk.Button(root, text="PDFを追加", command=self.add_pdf)
        self.add_pdf_button.pack()

        self.query_entry = tk.Entry(root)
        self.query_entry.pack()

        self.query_button = tk.Button(root, text="検索", command=self.search)
        self.query_button.pack()

        self.result_text = tk.Text(root)
        self.result_text.pack()

        # アプリ起動時にPDFを読み込む
        self.load_pdfs()

    def load_pdfs(self):
        documents_dir = 'data/documents'
        if not os.path.exists(documents_dir):
            os.makedirs(documents_dir)  # ディレクトリが存在しない場合は作成

        for filename in os.listdir(documents_dir):
            if filename.endswith('.pdf'):
                self.pdf_files.append(os.path.join(documents_dir, filename))
                self.result_text.insert(tk.END, f"読み込まれたPDF: {filename}\n")

    def add_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            # PDFをdocumentsディレクトリにコピー
            documents_dir = 'data/documents'
            new_file_path = os.path.join(documents_dir, os.path.basename(file_path))
            with open(file_path, 'rb') as fsrc, open(new_file_path, 'wb') as fdst:
                fdst.write(fsrc.read())
            self.pdf_files.append(new_file_path)
            self.result_text.insert(tk.END, f"追加されたPDF: {os.path.basename(file_path)}\n")

    def search(self):
        query = self.query_entry.get()
        results = self.perform_search(query)
        self.result_text.insert(tk.END, f"AIの回答: {results}\n")

    def perform_search(self, query):
        all_texts = [extract_text_from_pdf(pdf) for pdf in self.pdf_files]
        ranked_results = rank_results(query, all_texts)

        if ranked_results:
            best_document = ranked_results[0]
            answer = self.llm_model.generate_answer(query, best_document)
            return answer
        else:
            return "関連する文書が見つかりませんでした。"

    def generate_answer(self, query, document):
        return f"クエリ: {query}\n関連文書に基づく回答: {document[:200]}..."

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()
