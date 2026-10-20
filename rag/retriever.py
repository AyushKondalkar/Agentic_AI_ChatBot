
from pathlib import Path

from langchain_chroma import Chroma

from rag.jina_embeddings import JinaHostedEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "chroma_db_jina"
COLLECTION_NAME = "user_documents"

embeddings = JinaHostedEmbeddings()


def get_retriever(document_id: str, k: int = 4):
    """Return a retriever limited to the selected document."""

    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )

    return vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"document_id": document_id},
        },
    )


def search_documents(query: str, document_id: str, k: int = 4):
    """Find relevant PDF chunks for a user's question."""

    retriever = get_retriever(document_id, k)
    return retriever.invoke(query)
