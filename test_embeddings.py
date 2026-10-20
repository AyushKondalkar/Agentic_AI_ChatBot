from rag.jina_embeddings import JinaHostedEmbeddings

embeddings = JinaHostedEmbeddings()

# Test document embeddings
documents = [
    "RAG retrieves relevant information from documents.",
    "LangGraph helps orchestrate AI workflows.",
]

document_vectors = embeddings.embed_documents(documents)

print("Document embedding test passed!")
print("Number of embeddings:", len(document_vectors))
print("Embedding dimensions:", len(document_vectors[0]))

# Test query embedding
query_vector = embeddings.embed_query("What is RAG?")

print("Query embedding test passed!")
print("Query embedding dimensions:", len(query_vector))