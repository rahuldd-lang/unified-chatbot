"""
document_processor.py
─────────────────────
Handles PDF ingestion, text chunking, embedding, and ChromaDB indexing.

Design decisions:
- Chunk size 800 / overlap 150: balances context richness vs. retrieval precision
- ChromaDB DefaultEmbeddingFunction (ONNX all-MiniLM-L6-v2): no torch needed
- ChromaDB with persistent storage: survives app restarts
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple

import pypdf
import pdfplumber
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800          # Characters per chunk
CHUNK_OVERLAP = 150       # Overlap to preserve sentence continuity
# DefaultEmbeddingFunction uses ChromaDB's built-in ONNX all-MiniLM-L6-v2
# — no torch or torchvision dependency required
COLLECTION_NAME = "rag_documents"


class DocumentProcessor:
    """
    Processes PDFs into searchable chunks stored in ChromaDB.

    Attributes
    ----------
    persist_dir : str
        Directory where ChromaDB stores its data on disk.
    client : chromadb.PersistentClient
        ChromaDB client for vector storage.
    collection : chromadb.Collection
        The active collection holding all document chunks.
    splitter : RecursiveCharacterTextSplitter
        LangChain splitter that respects sentence/paragraph boundaries.
    embed_fn : DefaultEmbeddingFunction
        ChromaDB's built-in ONNX embedding function.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # ChromaDB's built-in ONNX embedding function (all-MiniLM-L6-v2)
        # Lighter than sentence-transformers: no torch/torchvision needed
        self.embed_fn = DefaultEmbeddingFunction()

        # Persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},  # Cosine distance for semantic search
        )

        # Text splitter that prefers paragraph > sentence > word boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def process_pdf(self, pdf_bytes: bytes, filename: str) -> Tuple[int, str]:
        """
        Extract text from a PDF, split into chunks, and upsert into ChromaDB.

        Parameters
        ----------
        pdf_bytes : bytes
            Raw bytes of the uploaded PDF file.
        filename : str
            Original filename (used as document identifier).

        Returns
        -------
        (chunk_count, doc_id) : tuple
            Number of chunks indexed and a stable document identifier.
        """
        # Derive a stable doc_id from filename + content hash
        content_hash = hashlib.md5(pdf_bytes).hexdigest()[:8]
        doc_id = f"{Path(filename).stem}_{content_hash}"

        # Skip re-indexing if document is already in the collection (single query)
        existing = self.collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            logger.info("Document '%s' already indexed (%d chunks).", filename, len(existing["ids"]))
            return len(existing["ids"]), doc_id

        # Extract text + page numbers for metadata
        pages = self._extract_text_with_pages(pdf_bytes)

        # Split each page's text into chunks; carry page metadata
        all_chunks, all_ids, all_metadata = [], [], []
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            chunks = self.splitter.split_text(page_text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_p{page_num}_c{i}"
                all_ids.append(chunk_id)
                all_chunks.append(chunk)
                all_metadata.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page_num,
                    "chunk_index": i,
                })

        if not all_chunks:
            raise ValueError(f"Could not extract any text from '{filename}'.")

        # Upsert in batches of 100 (ChromaDB limit per call is generous, but batching is safer)
        batch_size = 100
        for start in range(0, len(all_chunks), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=all_ids[start:end],
                documents=all_chunks[start:end],
                metadatas=all_metadata[start:end],
            )

        logger.info("Indexed '%s': %d chunks across %d pages.", filename, len(all_chunks), len(pages))
        return len(all_chunks), doc_id

    def get_all_documents(self) -> List[dict]:
        """
        Return a deduplicated list of indexed documents with chunk counts.
        """
        results = self.collection.get(include=["metadatas"])
        if not results["metadatas"]:
            return []

        doc_stats: dict[str, dict] = {}
        for meta in results["metadatas"]:
            did = meta["doc_id"]
            if did not in doc_stats:
                doc_stats[did] = {
                    "doc_id": did,
                    "filename": meta["filename"],
                    "pages": set(),
                    "chunk_count": 0,
                }
            doc_stats[did]["pages"].add(meta["page"])
            doc_stats[did]["chunk_count"] += 1

        return [
            {**v, "page_count": len(v["pages"])}
            for v in doc_stats.values()
        ]

    def get_full_text(self, doc_id: str) -> str:
        """
        Retrieve and reconstruct the full text of a document from its chunks.
        Chunks are sorted by page then chunk index for correct ordering.
        """
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )
        if not results["documents"]:
            return ""

        # Sort chunks back into document order
        paired = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda x: (x[1]["page"], x[1]["chunk_index"]),
        )
        return "\n\n".join(doc for doc, _ in paired)

    def collection_stats(self) -> dict:
        """Return total chunk count and document count for the sidebar."""
        count = self.collection.count()
        docs = self.get_all_documents()
        return {"total_chunks": count, "total_documents": len(docs)}

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
        self.collection.delete(where={"doc_id": doc_id})

    # ── Private helpers ──────────────────────────────────────────────────────

    def _extract_text_with_pages(self, pdf_bytes: bytes) -> List[Tuple[int, str]]:
        """
        Extract text from each PDF page using pdfplumber (better table/layout
        handling than pypdf).  Falls back to pypdf on extraction errors.

        Returns list of (page_number, text) tuples (1-indexed).
        """
        import io

        pages = []
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append((i, text))
        except Exception as e:
            logger.warning("pdfplumber failed (%s), falling back to pypdf.", e)
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append((i, text))

        return pages
