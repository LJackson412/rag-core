



from __future__ import annotations

from typing import Any, Dict, List

import fitz  # type: ignore


def load_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Load PDF metadata as dict.
    """
    with fitz.open(pdf_path) as doc:
        meta = dict(doc.metadata or {})  
        meta.setdefault("page_count", doc.page_count)
        meta.setdefault("is_encrypted", doc.is_encrypted)
    return meta


def load_texts_from_pdf(pdf_path: str) -> List[str]:
    """
    Load every PDF page as one string.
    """
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text")
            pages.append(text.strip())

    return pages


def load_tables_from_pdf(pdf_path: str) -> list[str]:
    "load tables from pdf as str"
    return []

def load_imgs_from_pdf(pdf_path: str) -> list[str]:
    "load imgs from pdf as base64"
    return []

def load_page_as_image_from_pdf(pdf_path: str) -> list[str]:
    "load every pdf page as image as base64"
    return []

