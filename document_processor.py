from pathlib import Path


class DocumentProcessor:

    def __init__(
        self,
        documents_path: Path,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):

        self.documents_path = documents_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "Chunk overlap must be smaller than chunk size."
            )

    def load_documents(self):

        documents = []

        for file_path in self.documents_path.glob("*.txt"):

            text = file_path.read_text(
                encoding="utf-8"
            ).strip()

            if not text:
                continue

            chunks = self.chunk_text(text)

            for index, chunk in enumerate(chunks):

                documents.append({
                    "id": f"{file_path.stem}_{index}",
                    "text": chunk,
                    "metadata": {
                        "source": file_path.name,
                        "chunk_index": index
                    }
                })

        return documents

    def chunk_text(self, text: str):

        chunks = []

        step = self.chunk_size - self.chunk_overlap

        for start in range(0, len(text), step):

            chunk = text[start:start + self.chunk_size]

            if chunk.strip():
                chunks.append(chunk.strip())

        return chunks