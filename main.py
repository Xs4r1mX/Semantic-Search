import argparse

from config import DOCUMENTS_PATH

from embedding_service import EmbeddingService
from vector_store import VectorStore
from document_processor import DocumentProcessor


class SemanticSearchEngine:

    def __init__(self):

        self.embedding_service = EmbeddingService()

        self.vector_store = VectorStore()

        self.document_processor = DocumentProcessor(
            documents_path=DOCUMENTS_PATH,
            chunk_size=500,
            chunk_overlap=50
        )

    def index_documents(self):

        # 1. Read documents and create chunks
        documents = self.document_processor.load_documents()

        if not documents:
            print("No documents found.")
            return

        print(f"Loaded {len(documents)} chunks.")

        # 2. Extract text
        texts = [
            document["text"]
            for document in documents
        ]

        # 3. Generate embeddings through OpenAI
        print("Generating embeddings...")

        embeddings = self.embedding_service.generate_embeddings(
            texts
        )

        # 4. Store embeddings in ChromaDB
        print("Storing vectors...")

        self.vector_store.add_documents(
            ids=[
                document["id"]
                for document in documents
            ],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                document["metadata"]
                for document in documents
            ]
        )

        print("Indexing completed successfully!")

    def search_documents(self, query: str, top_k: int = 3):

        # 1. Generate embedding for the search query
        query_embedding = self.embedding_service.generate_embedding(
            query
        )

        # 2. Retrieve similar documents
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # 3. Display results
        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        if not documents:
            print("No matching documents found.")
            return

        print(f"\nSearch query: {query}")
        print("=" * 60)

        for index, document in enumerate(documents):

            distance = distances[index]

            similarity = 1 - distance

            metadata = metadatas[index]

            print(f"\nResult {index + 1}")
            print(f"Source: {metadata['source']}")
            print(f"Chunk: {metadata['chunk_index']}")
            print(f"Similarity: {similarity:.4f}")
            print(f"Content: {document}")

            print("-" * 60)


def main():

    parser = argparse.ArgumentParser(
        description="Semantic Document Search Engine"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # Index command
    subparsers.add_parser("index")

    # Search command
    search_parser = subparsers.add_parser("search")

    search_parser.add_argument(
        "query",
        type=str
    )

    search_parser.add_argument(
        "--top-k",
        type=int,
        default=3
    )

    args = parser.parse_args()

    engine = SemanticSearchEngine()

    if args.command == "index":
        engine.index_documents()

    elif args.command == "search":
        engine.search_documents(
            query=args.query,
            top_k=args.top_k
        )


if __name__ == "__main__":
    main()