import os
import fitz  # PyMuPDF
import docx
from bs4 import BeautifulSoup
import markdown

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    elif ext == ".html" or ext == ".htm":
        return extract_html(file_path)
    elif ext == ".md":
        return extract_markdown(file_path)
    elif ext == ".txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return ""

def extract_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text()

def extract_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html = markdown.markdown(f.read())
    return BeautifulSoup(html, "html.parser").get_text()



from pathlib import Path

def load_documents(path):
    docs = []
    for fname in os.listdir(path):
        full_path = os.path.join(path, fname)
        if os.path.isfile(full_path):
            text = extract_text(full_path)
            if text.strip():  # Avoid empty files
                docs.append((fname, text))
    return docs


