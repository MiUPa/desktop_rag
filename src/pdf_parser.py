import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text()
            print("【DEBUG】ページテキストの先頭:", page_text[:100])  # 先頭100文字をログ出力
            text += page_text
    return text
