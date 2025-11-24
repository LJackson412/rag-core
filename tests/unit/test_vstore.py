from langchain_core.runnables import RunnableConfig

from rag_app.factory.factory import build_vstore
from rag_app.index.config import IndexConfig
from rag_app.utils.utils import pretty_print_docs

if __name__ == "__main__":

    config = RunnableConfig(
        configurable={
            "doc_id": "DB_ZB_25_Test",
            "collection_id": "DB_ZB_Test",
        }
    )

    config = RunnableConfig(
        configurable={"doc_id": "DB_ZB_25", "collection_id": "DB_ZB"}
    )

    index_config = IndexConfig.from_runnable_config(config)

    vstore = build_vstore(index_config.embedding_model, index_config.collection_id)

    k = vstore._collection.count()

    filter: dict[str, str] = {"doc_id": index_config.doc_id}
    retriever = vstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": filter},
    )

    docs = retriever.invoke("")

    pretty_print_docs(docs)
