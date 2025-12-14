from typing import cast

from langchain_core.runnables import RunnableConfig

from rag_app.providers.composition import get_provider_factory
from rag_app.retrieval.config import RetrievalConfig
from rag_app.utils.utils import pretty_print_docs

if __name__ == "__main__":

    config = RunnableConfig(
        configurable={"doc_id": "Cancom_240514", "collection_id": "Cancom_OCR"}
    )

    # config = RunnableConfig(
    #     configurable={"doc_id": "Cancom_240514", "collection_id": "Cancom_LLM"}
    # )

    index_config = RetrievalConfig.from_runnable_config(config)

    provider_factory = get_provider_factory()
    vstore = provider_factory.build_vstore(
        index_config.embedding_model, index_config.collection_id
    )

    k = vstore._collection.count()

    filter = cast(str, {"doc_id": index_config.doc_id})
    retriever = vstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": filter},
    )

    docs = retriever.invoke("Query")

    pretty_print_docs(docs)
