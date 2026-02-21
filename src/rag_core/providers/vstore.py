from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


def get_chroma_vstore(
    embedding_model: Embeddings,
    collection_name: str = "chroma_collection",
    persist_directory: str = ".chroma_directory",
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )


def get_vstore(
    embedding_model: Embeddings,
    provider: str = "chroma",
    collection_name: str = "chroma_collection",
    persist_directory: str = ".chroma_directory",
) -> VectorStore:
    match provider:
        case "chroma":
            return get_chroma_vstore(
                embedding_model=embedding_model,
                collection_name=collection_name,
                persist_directory=persist_directory,
            )
        case _:
            raise ValueError(f"Unknown provider: {provider}")


# if __name__ == "__main__":
#     from db.providers.genaihub.azure_openai.embedding import get_azure_openai_embedding
#     from rag_core.index.graph import PERSIST_DIRECTORY

#     # from rag_core.providers.embedding import get_openai_embedding
#     from rag_core.utils.utils import pretty_print_docs

#     # COLLECTION = "Test"
#     # DOC_ID = "240514_CANCOM"

#     # PATH = "./data/Prod/M_00003356"
#     PATH = "./data/Prod/M_00003822"

#     audit_input = read_input_data(PATH)
#     COLLECTION = audit_input.workitem_element_id
#     DOC_ID = "Berechtigungskonzept_DigiBef_V3.0.xlsx"

#     # emb = get_openai_embedding()
#     emb = get_azure_openai_embedding()

#     vstore = get_chroma_vstore(
#         emb, collection_name=COLLECTION, persist_directory=PERSIST_DIRECTORY
#     )

#     count = vstore._collection.count()
#     print("Anzahl Docs in Collection ----> ", count)

#     # K = 3
#     K = count

#     # # search_kwargs = {
#     # #     "k": k,
#     # #     "filter": {
#     # #         "$and": [
#     # #             {"doc_id": {"$eq": doc_id}},
#     # #             {"category": {"$eq": "Table"}},
#     # #             {"enrich_mode": {"$eq": "none"}},
#     # #         ]
#     # #     }
#     # # }

#     search_kwargs = {
#         "k": K,
#         "filter": {
#             "$and": [
#                 {"doc_id": {"$eq": DOC_ID}},
#                 # {"category": {"$eq": "Image"}},
#                 {"category": {"$eq": "Table"}},
#                 # {"page_number": {"$eq": 6}},
#             ]
#         },
#     }

#     # search_kwargs = {
#     #     "k": K,
#     #     "filter": {"category": {"$eq": "Image"}},
#     # }

#     # search_kwargs = {
#     #     "k": K
#     # }

#     retr = vstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

#     docs = retr.invoke("TEST")
#     pretty_print_docs(docs)
#     print("Anzahl Docs nach Query ----> ", len(docs))
