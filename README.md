# LLM Extractor & RAG App

RAG-App mit zwei Indexprozessen "OCR-basiert mit llm enrichment und LLM-basierte Extraktion" und einem "2-Stage-Retrievalprozess mit LLM-Compression"

**Tech Stack:** 
- LangGraph/LangChain
- Chroma vector store (in-memory) – interchangeable with your own providers.
- OpenAI models – interchangeable with your own providers.

## Index-Process

### OCR-Indexer: OCR-Extraction with llm metadata enrichment

![UML: OCR-Indexer](/docs/index_ocr_graph.png)
![Studio: OCR-Indexer](/docs/index_ocr_graph.png)

### LLM-Indexer: LLM-Extraction and Splitting with metadata enrichment

![UML: LLM-Indexer](/docs/index_ocr_graph.png)
![Studio: OCR-Indexer](/docs/index_ocr_graph.png)

## Retrieval-Process: 2-Stage with LLM-Compression

![UML: Retrieval-Graph](/docs/index_ocr_graph.png)
![Studio: OCR-Indexer](/docs/index_ocr_graph.png)



## Setup & Launch

1. **Prerequisites:**  
   - Install *poppler* for `pdf2image`, and adjust the path in `.env` if necessary:  
     - https://github.com/oschwartz10612/poppler-windows/releases/  
     - Define the path in `.env` as `POPLER_PATH`
   - Add environment variables like `OPENAI_API_KEY` in `.env`
   - Optional for LangGraph/LangSmith UI: `LANGSMITH_API_KEY`

2. **Install dependencies & start the local runtime:**

```bash
python -m venv .venv          # alternatively: py -3.12 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .[dev]
langgraph dev 
```

## How to Use the Graphs

### LangSmith UI

1. **Select a graph:** After running `langgraph dev`, you can switch between `index_graph` and `retrieval_graph` via the LangGraph UI (or LangSmith, if connected).
2. **Set parameters:** Each graph accepts structured inputs. Before execution, ensure that `collection_id`, `doc_id`, and (if applicable) `chunking_config` match your documents.

**Retrieval Graph**
Example with an already indexed collection in `path: ./data/DB_ZB25.pdf`:

* `collection_id` = DB_ZB, `doc_id` = DB_ZB_S55 (95776bb8-7907-486c-b1fb-04e456c44e2c)
* `query`:
  * Was macht die DB in Bezug auf Nachhaltigkeit?
  * Wie viele Arbeitskräfte hat der DB Konzern auf der ganzen Welt?

## Indexer
- Both Collections are already indexed in ".chroma_directory", you can use it with der "retriever"

### LLM-Indexer: `collection_id: Cancom_LLM`

* `doc_id: Cancom_240514`
* `path: ./data/Cancom/240514_CANCOM_Zwischenmitteilung.pdf`

* `doc_id: Cancom_20241112`
* `path: ./data/Cancom/20241112_CANCOM_Zwischenmitteilung.pdf`


### OCR-Indexer: `collection_id: Cancom_OCR`

* `doc_id: Cancom_240514`
* `path: ./data/Cancom/240514_CANCOM_Zwischenmitteilung.pdf`

* `doc_id: Cancom_20241112`
* `path: ./data/Cancom/20241112_CANCOM_Zwischenmitteilung.pdf`


## Retrieval 


### Question the whole Collection
* `collection_id: Cancom_LLM / Cancom_OCR`
* `questions: Cancom_LLM`
* `doc_id: None`
- Wie beschreibt CANCOM in den Zwischenmitteilungen Q1 2024 und Q3 2024 die Auswirkungen der Übernahme der KBC- / CANCOM-Austria-Gruppe auf Umsatz, Rohertrag und EBITDA der CANCOM Gruppe bzw. des Segments International?

### Question one Doc in the Collection 
* `collection_id: Cancom_LLM / Cancom_OCR`
* `doc_id: Cancom_240514`
* `questions:`
- Wo hat die Cancom SE ihren Sitz, und unter welcher Telefonnummer ist sie erreichbar?
  - Evidenz on page 13
- Wie hoch ist der prozentuale und absolute Unterschied der Vorräte zwischen dem 31.12.2023 und dem 31.03.2024?
  - Evidenz on page 18


## Operational Notes & Tips

* Set the **search language** to match the document language to improve semantic similarity – also for multimodal embeddings.
* The **extraction** process tags each segment’s language and instructs the model to read content in that same language.
* **Rate limits** vary between models; if you use a mixed provider stack, consider applying a rate limiter per model family (see OpenAI limits: [https://platform.openai.com/settings/organization/limits](https://platform.openai.com/settings/organization/limits)).
* **Logging/Debugging:** Exceptions from the extractor are output via the logger and stored in state for the client

### Known Exception During PDF Indexing

* When processing large, densely written pages, the model may stop with a `LengthFinishReasonError`.
-  Each page is converted into a 200/120-DPI PNG and sent (as base64) together with the prompt and expected structured output. Large image embeddings combined with a long prompt can exceed the model’s response limit.
- Error loged and werden an den Client über den State zurück gegeben

* Aktuell verarbeitet der "extract_text_node" im OCR-Indexer auch text aus Tabellen, zusätzlich macht das der Node extract_tables, doppelte Segmente
* Exceptions werden an den Client weitergegeben 
* Prompts sind nicht optimiert, können an unterschiedliche Domänen angepasst werden 


### Future Implementations
* Optionales LLM-Enrichment in OCR-Graph
* Differnet 1-Stage-Retriever implementations
* Ähnlichkeitsscore nach Abfrage in Document Metadata schreiben
* Prompt-Optimierung

