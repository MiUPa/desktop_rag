# RAGアプリ

このアプリは、PDFを検索対象に追加し、対話形式でユーザーとやり取りし、GPT-2モデルを使用して回答を生成します。

## ディレクトリ構成

```
RAGApp/
│
├── src/
│   ├── app.py          # メインアプリケーション
│   ├── pdf_parser.py   # PDF解析用スクリプト
│   ├── ranking_model.py # リランキングモデル用スクリプト
│   └── llm_model.py    # GPT-2モデルを使用するスクリプト
│
├── data/
│   └── documents/      # ユーザーが追加したPDFファイルを保存するディレクトリ
│
├── requirements.txt     # 必要なパッケージのリスト
└── README.md            # プロジェクトの説明
```

## 必要なライブラリのインストール

以下のコマンドを実行して、必要なライブラリをインストールします。

```
pip install transformers torch
```

## モデルの設定

`src/llm_model.py`を以下のように設定します。

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class LLMModel:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_answer(self, query, context):
        input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        output = self.model.generate(input_ids, max_length=150)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer.split("Answer:")[-1].strip()  # "Answer:"以降の部分を返す
```

## アプリの設定

`src/app.py`を以下のように設定します。

```python
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

    def add_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.pdf_files.append(file_path)
            self.result_text.insert(tk.END, f"追加されたPDF: {file_path}\n")

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

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()
```

## アプリの実行

以下のコマンドを実行してアプリを起動します。

```
python src/app.py
```

## アプリの使用

1. **PDFの追加**: 「PDFを追加」ボタンをクリックして、検索対象のPDFファイルを追加します。
2. **質問の入力**: 質問を入力し、「検索」ボタンをクリックします。
3. **AIの回答**: AIが関連する文書をもとに回答を生成し、結果が表示されます。

## 注意点

- モデルのサイズやリソースに注意してください。特に大きなモデルを使用する場合は、十分なメモリが必要です。
- エラーが発生した場合は、エラーメッセージを確認し、必要に応じて修正を行ってください。