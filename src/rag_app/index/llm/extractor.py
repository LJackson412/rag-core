# import asyncio
# import base64
# import logging
# from io import BytesIO
# from typing import Any, cast

# from langchain_core.language_models import BaseChatModel
# from langchain_core.language_models.base import LanguageModelInput
# from langchain_core.messages import HumanMessage
# from openai import ContentFilterFinishReasonError, LengthFinishReasonError
# from pdf2image import convert_from_path
# from PIL import Image
# from pypdf import PdfReader

# logger = logging.getLogger(__name__)


# def _pdf_to_image(pdf_path: str) -> list[Image.Image]:
#     imgs = convert_from_path(pdf_path, dpi=120)
#     imgs_gray = [img.convert("L") for img in imgs]
#     return imgs_gray


# def _pil_image_to_base64(imgs: list[Image.Image]) -> list[str]:
#     imgs_base64 = []
#     for img in imgs:
#         buffer = BytesIO()
#         img.save(buffer, format="PNG")
#         img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
#         imgs_base64.append(img_base64)

#     return imgs_base64


# def _get_pdf_metadata(pdf_path: str) -> dict[str, Any]:
#     reader = PdfReader(pdf_path)
#     info = reader.metadata
#     if info is None:
#         return {}
#     return dict(info)


# # Todo: Mapping auf ExtractedData in "schema.py"?


# async def extract_from_pdf(
#     pdf_path: str,
#     llm: BaseChatModel,
#     extract_data_prompt: str,
#     extraction_data: type[LLMExtractedData],
# ) -> list[ExtractedData]:

#     pdf_imgs = await asyncio.to_thread(_pdf_to_image, pdf_path)
#     imgs_base64 = await asyncio.to_thread(_pil_image_to_base64, pdf_imgs)

#     structured_llm = llm.with_structured_output(extraction_data)

#     inputs: list[LanguageModelInput] = []
#     for img_base64 in imgs_base64:
#         messages = [
#             HumanMessage(
#                 content=[
#                     {"type": "text", "text": extract_data_prompt},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/png;base64,{img_base64}",
#                         },
#                     },
#                 ]
#             )
#         ]
#         inputs.append(messages)

#     # NOTE:
#     # The global rate limiter of the OpenAI client primarily operates based on tokens per minute (TPM),
#     # not on the number of concurrent requests. However, if abatch starts multiple requests in parallel,
#     # their tokens can accumulate and exceed the TPM limit before the rate limiter kicks in.
#     # Therefore, max_concurrency limits the actual parallelism and prevents such TPM bursts.
#     extracted_data = cast(
#         list[LLMExtractedData],
#         await structured_llm.abatch(
#             inputs,
#             config={"max_concurrency": 2},
#             return_exceptions=True,
#         ),
#     )

#     pdf_metadata = await asyncio.to_thread(_get_pdf_metadata, pdf_path)

#     enriched_data = []

#     skip_exceptions = (LengthFinishReasonError, ContentFilterFinishReasonError)

#     for page_number, page_data in enumerate(extracted_data, start=1):
#         if isinstance(page_data, skip_exceptions):
#             logger.warning(
#                 "Skipping page %s due to extraction limit: %s",
#                 page_number,
#                 page_data.__class__.__name__,
#             )
#             continue

#         if isinstance(page_data, Exception):
#             raise page_data

#         base_metadata = getattr(page_data, "metadata", {}) or {}
#         merged_metadata = {
#             **base_metadata,
#             **pdf_metadata,
#             "source": pdf_path,
#             "page": base_metadata.get("page", page_number),
#         }
#         enriched_data.append(
#             ExtractedData(
#                 **page_data.model_dump(),
#                 metadata=merged_metadata,
#             )
#         )

#     return enriched_data
