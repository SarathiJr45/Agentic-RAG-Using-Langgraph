# backend/data_ingestion/pdf_loader.py

import os
import glob
from PyPDF2 import PdfReader

def extract_text_from_pdfs(pdf_folder):

    

    pdf_paths = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    page_texts = []

    for path in pdf_paths:
        reader = PdfReader(path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                page_texts.append(text)
    
    return page_texts  # Now returning list of pages

# backend/data_ingestion/pdf_loader.py (continued)

def chunk_text(pages, max_tokens=300):
    from textwrap import wrap

    chunks = []
    for page in pages:
        chunks.extend(wrap(page, max_tokens))
    
    return chunks
