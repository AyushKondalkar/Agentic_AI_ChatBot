
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from rag.jina_embeddings import JinaHostedEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "chroma_db_jina"
COLLECTION_NAME = "user_documents"

embeddings = JinaHostedEmbeddings()


def ingest_pdf(pdf_path: str, document_id: str):
    """Load a PDF, split it into chunks, embed them, and store them."""

    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1. Load PDF pages
    documents = PyPDFLoader(str(pdf_path)).load()

    # 2. Split pages into smaller text chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No readable text found in the PDF.")

    # 3. Add metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["document_id"] = document_id
        chunk.metadata["source"] = pdf_path.name
        chunk.metadata["chunk_index"] = i

    # 4. Connect to the Jina-specific Chroma database
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )

    # 5. Delete previous chunks for this document, if any
    existing = vector_db.get(
        where={"document_id": document_id}
    )

    if existing["ids"]:
        vector_db.delete(ids=existing["ids"])

    # 6. Store the newly embedded chunks
    ids = [f"{document_id}_{i}" for i in range(len(chunks))]

    vector_db.add_documents(
        documents=chunks,
        ids=ids,
    )

    # 7. Return counts for reporting in the UI
    return len(documents), len(chunks)
