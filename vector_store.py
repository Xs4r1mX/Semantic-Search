import chromadb

from config import CHROMA_DB_PATH


class VectorStore:

    def __init__(self):

        # Initialize persistent ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH)
        )

        # Create or retrieve collection
        self.collection = self.client.get_or_create_collection(
            name="semantic_documents",
            metadata={
                "hnsw:space": "cosine"
            }
        )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict]
    ):

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Stored {len(documents)} document chunks.")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3
    ):

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )