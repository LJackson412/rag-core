import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List

import fitz  # type: ignore
from pdf2image import convert_from_path
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf


@dataclass(slots=True)
class PDFImage:
    img_base64: str
    ext: str  
    page_number: int

    @property
    def image_url(self) -> str:
        # ext normalisieren für MIME-Type
        ext = self.ext.lower()
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{self.img_base64}"


@dataclass(slots=True)
class PDFText:
    text: str
    page_number: int


@dataclass(slots=True)
class PDFTable:
    text: str | None
    html: str | None
    page_number: int | None


def load_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Load PDF metadata as dict.
    """
    with fitz.open(pdf_path) as doc:
        meta = dict(doc.metadata or {})
        meta.setdefault("page_count", doc.page_count)
        meta.setdefault("is_encrypted", doc.is_encrypted)
    return meta


def load_texts_from_pdf(pdf_path: str) -> List[PDFText]:
    """
    Load every PDF page as one Text object.

    Returns:
        List[Text]: Eine Liste von Text-Objekten, je Seite ein Eintrag.
                    page_number ist als String und 1-basiert ("1", "2", ...).
    """
    pages: List[PDFText] = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append(
                PDFText(
                    text=text.strip(),
                    page_number=page_index,
                )
            )

    return pages


def load_imgs_from_pdf(pdf_path: str) -> List[PDFImage]:
    images: List[PDFImage] = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")  
                b64_str = base64.b64encode(image_bytes).decode("ascii")

                images.append(
                    PDFImage(
                        img_base64=b64_str,
                        ext=ext,
                        page_number=page_index,
                    )
                )

    return images


def load_page_imgs_from_pdf(pdf_path: str) -> list[PDFImage]:
    """
    Load every PDF page as an image (Base64-kodiert) und liefere eine Liste von PDFImage.
    Jede Seite wird als PNG in Graustufen gerendert.
    """
    imgs = convert_from_path(pdf_path, dpi=120)

    imgs_gray = [img.convert("L") for img in imgs]

    pdf_imgs: list[PDFImage] = []

    for page_number, img in enumerate(imgs_gray, start=1):
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        img_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        pdf_imgs.append(
            PDFImage(
                img_base64=img_base64,
                ext="png",
                page_number=page_number,
            )
        )

    return pdf_imgs


def load_tables_from_pdf(pdf_path: str) -> list[PDFTable]:
    "detectron2_onnx: used for  document layout"
    "tesseract: for ocr"

    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=False,
        infer_table_structure=True,
        pdf_infer_table_structure=True,
        languages=["eng", "deu"],
    )

    tables = [e for e in elements if e.category == "Table"]

    pdf_tables = []
    for t in tables:
        pdf_table = PDFTable(t.text, t.metadata.text_as_html, t.metadata.page_number)

        pdf_tables.append(pdf_table)

    return pdf_tables

@dataclass(slots=True)
class Segment:
    id: str
    category: str         
    text: str
    lang: List[str]
    page_number: int
    file_directory: str
    filename: str
    last_modified: str

    text_as_html: str  # used for tables
    img_base64: str
    img_mime_type: str
    metadata: Dict[str, Any]
    
    @property
    def image_url(self) -> str:
        return f"data:{self.img_mime_type};base64,{self.img_base64}"


@dataclass(slots=True)
class CompositeSegment:
    id: str
    type: str             
    text: str
    lang: List[str]
    page_number: int
    file_directory: str
    filename: str
    last_modified: str

    text_as_html: str      
    metadata: Dict[str, Any]


def load_and_split_pdf(path: str, lang: list[str] | None = None) -> Dict[str, List[Segment] | List[CompositeSegment]]:
    if lang is None:
        lang = ["eng", "deu"]
    
    elements = partition_pdf(
        filename=path,
        strategy="hi_res",
        infer_table_structure=True,
        pdf_infer_table_structure=True,
        extract_images_in_pdf=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        languages=lang,
    )

    chunks = chunk_by_title(elements, include_orig_elements=True)

    segments: List[Segment] = []
    for e in elements:
        md = e.metadata.to_dict()

        segment = Segment(
            id=e.id,
            category=e.category,
            text=e.text,
            lang=md.get("languages") or [],
            page_number=md.get("page_number") or -1,
            file_directory=md.get("file_directory") or "",
            filename=md.get("filename") or "",
            last_modified=md.get("last_modified") or "",
            text_as_html=md.get("text_as_html") or "",
            img_base64=md.get("image_base64") or "",
            img_mime_type=md.get("image_mime_type") or "",
            metadata=md,
        )

        segments.append(segment)

    composite_segments: List[CompositeSegment] = []
    for chunk in chunks:
        md = chunk.metadata.to_dict()

        composite_segments.append(
            CompositeSegment(
                id=chunk.id,
                type=chunk.category,           
                text=chunk.text,
                lang=md.get("languages") or [],
                page_number=md.get("page_number") or -1,
                file_directory=md.get("file_directory") or "",
                filename=md.get("filename") or "",
                last_modified=md.get("last_modified") or "",
                text_as_html=md.get("text_as_html") or "",
                metadata=md,
            )
        )

    return {
        "segments": segments,
        "composite_segments": composite_segments,
    }




# if __name__ == "__main__":
    
#     from rag_app.config.settings import settings
    
#     elements = load_and_split_pdf(path="./data/Cancom/240514_CANCOM_Zwischenmitteilung.pdf", lang=None)
    
#     segments = elements["segments"]
#     composite_segments = elements["composite_segments"]
    
#     for s in segments:
#         pprint(asdict(s))
    
#     for cs in composite_segments:
#         pprint(asdict(cs))
    