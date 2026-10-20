
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_NAME = "jina-embeddings-v3"
API_URL = "https://api.jina.ai/v1/embeddings"


class JinaHostedEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")

        if not self.api_key:
            raise ValueError("JINA_API_KEY not found in .env")

    def _embed(
        self, texts: list[str], task: str
    ) -> list[list[float]]:
        if not texts:
            return []

        payload = json.dumps({
            "model": MODEL_NAME,
            "input": texts,
            "task": task,
        }).encode("utf-8")

        request = Request(
            API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))

        items = sorted(
            result["data"],
            key=lambda item: item["index"],
        )
        return [item["embedding"] for item in items]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []

        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            vectors.extend(
                self._embed(batch, task="retrieval.passage")
            )

        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed(
            [text],
            task="retrieval.query",
        )[0]
