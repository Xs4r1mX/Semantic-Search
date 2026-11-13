# Semantic Document Search Engine

A Python-based semantic document search engine using OpenAI Embeddings and ChromaDB.

## Features

- Generate text embeddings using OpenAI API.
- Store embeddings in ChromaDB.
- Persistent local vector storage.
- Document ingestion from text files.
- Text chunking with configurable overlap.
- Semantic similarity search.
- Retrieve top-K relevant document chunks.
- Store and retrieve document metadata.

## Tech Stack

- Python
- OpenAI Embeddings API
- ChromaDB
- Cosine Similarity

## Project Structure

```text
semantic-search/
├── documents/
├── config.py
├── embedding_service.py
├── vector_store.py
├── document_processor.py
├── main.py
├── requirements.txt
├── .env
└── .gitignore
```

## Setup

### 1. Clone the repository

```bash
git clone REPOSITORY_URL
cd semantic-search
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
```

### 6. Index documents

```bash
python main.py index
```

### 7. Perform semantic search

```bash
python main.py search "How do background tasks work?"
```

Specify the number of results:

```bash
python main.py search "Python backend development" --top-k 5
```

## Architecture

1. Documents are loaded from the documents directory.
2. Documents are divided into smaller chunks.
3. OpenAI generates embeddings for each chunk.
4. ChromaDB stores embeddings, original text and metadata.
5. User queries are converted into embeddings.
6. ChromaDB retrieves semantically similar documents.

## Future Improvements

- PDF document ingestion.
- Token-based chunking.
- Metadata filtering.
- Retrieval quality optimization.
- Retrieval Augmented Generation (RAG).
- Django REST Framework integration.