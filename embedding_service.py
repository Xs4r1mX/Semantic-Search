from openai import OpenAI

from config import OPENAI_API_KEY, EMBEDDING_MODEL


class EmbeddingService:

    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

    def generate_embedding(self, text: str) -> list[float]:

        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )

        return response.data[0].embedding

    def generate_embeddings(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        if not texts:
            return []

        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )

        # Maintain the original input order
        sorted_data = sorted(
            response.data,
            key=lambda item: item.index
        )

        return [
            item.embedding
            for item in sorted_data
        ]