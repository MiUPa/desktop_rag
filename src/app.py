import tkinter as tk
from tkinter import filedialog
from pdf_parser import extract_text_from_pdf
from ranking_model import rank_results

class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAGアプリ")
        self.pdf_files = []

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
            self.result_text.insert(tk.END, f"追加されたPDF: {file_path}\n")

    def search(self):
        query = self.query_entry.get()
        results = self.perform_search(query)
        self.result_text.insert(tk.END, f"検索結果: {results}\n")

    def perform_search(self, query):
        # PDF解析と検索ロジックをここに実装
        all_texts = [extract_text_from_pdf(pdf) for pdf in self.pdf_files]
        ranked_results = rank_results(query, all_texts)
        return ranked_results

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()
