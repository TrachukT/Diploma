import io
import csv

import PyPDF2
import docx


def extract_text_from_pdf(contents: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(contents))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


def extract_text_from_csv(contents: bytes) -> str:
    buffer = io.StringIO(contents.decode("utf-8"))
    reader = csv.reader(buffer)
    lines = ["; ".join(row) for row in reader]
    return "\n".join(lines)


def extract_text_from_docx(contents: bytes) -> str:
    doc = docx.Document(io.BytesIO(contents))
    text = "\n".join([p.text for p in doc.paragraphs])
    return text
