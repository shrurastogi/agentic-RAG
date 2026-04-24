"""
Vector Store implementation using Weaviate.

This module handles:
- Weaviate schema definition for 3 classes: TextChunk, Table, Figure
- Embedding generation using BGE
- Batch insertion with metadata
- Hybrid search (BM25 + semantic)
"""

from typing import List, Dict, Optional, Any
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np

from config.settings import settings
from src.ingestion.text_processor import TextChunk


class VectorStore:
    """
    Weaviate-based vector store with hybrid search capabilities.

    Manages three content types:
    - TextChunk: Regular text chunks from documents
    - Table: Extracted tables with structure
    - Figure: Figure descriptions and metadata
    """

    def __init__(
        self,
        weaviate_url: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize vector store.

        Args:
            weaviate_url: Weaviate instance URL (default from settings)
            embedding_model: Embedding model name (default from settings)
        """
        self.weaviate_url = weaviate_url or settings.WEAVIATE_URL
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL

        logger.info(f"Initializing VectorStore with Weaviate at {self.weaviate_url}")

        # Initialize Weaviate client
        self.client = None
        self._connect()

        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=settings.EMBEDDING_DEVICE
        )

        logger.info(f"Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")

    def _connect(self):
        """Connect to Weaviate instance."""
        try:
            self.client = weaviate.connect_to_local(
                host=self.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                port=int(self.weaviate_url.split(":")[-1]) if ":" in self.weaviate_url else 8080
            )

            if self.client.is_ready():
                logger.info("Successfully connected to Weaviate")
            else:
                logger.error("Weaviate is not ready")
                raise ConnectionError("Weaviate is not ready")

        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    def create_schema(self):
        """
        Create Weaviate schema with three classes: TextChunk, Table, Figure.

        This defines the structure for storing different content types.
        """
        logger.info("Creating Weaviate schema")

        try:
            # Delete existing collections if they exist
            for class_name in [settings.TEXT_CHUNK_CLASS, settings.TABLE_CLASS, settings.FIGURE_CLASS]:
                try:
                    self.client.collections.delete(class_name)
                    logger.info(f"Deleted existing collection: {class_name}")
                except Exception:
                    pass

            # Create TextChunk collection
            self.client.collections.create(
                name=settings.TEXT_CHUNK_CLASS,
                description="Text chunks from documents with semantic embeddings",
                vectorizer_config=Configure.Vectorizer.none(),  # We provide our own vectors
                properties=[
                    Property(name="chunk_id", data_type=DataType.TEXT, description="Unique chunk identifier"),
                    Property(name="doc_id", data_type=DataType.TEXT, description="Document identifier"),
                    Property(name="text", data_type=DataType.TEXT, description="Chunk text content"),
                    Property(name="page_num", data_type=DataType.INT, description="Page number"),
                    Property(name="section_title", data_type=DataType.TEXT, description="Section title"),
                    Property(name="chunk_index", data_type=DataType.INT, description="Chunk index in document"),
                    Property(name="token_count", data_type=DataType.INT, description="Approximate token count"),
                    Property(name="bbox", data_type=DataType.OBJECT, description="Bounding box coordinates"),
                    Property(name="contains_statistics", data_type=DataType.BOOL, description="Contains statistical data"),
                ]
            )
            logger.info(f"Created collection: {settings.TEXT_CHUNK_CLASS}")

            # Create Table collection
            self.client.collections.create(
                name=settings.TABLE_CLASS,
                description="Extracted tables with structure and summaries",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="table_id", data_type=DataType.TEXT, description="Unique table identifier"),
                    Property(name="doc_id", data_type=DataType.TEXT, description="Document identifier"),
                    Property(name="page_num", data_type=DataType.INT, description="Page number"),
                    Property(name="markdown", data_type=DataType.TEXT, description="Table in markdown format"),
                    Property(name="summary", data_type=DataType.TEXT, description="LLM-generated table summary"),
                    Property(name="json_structure", data_type=DataType.TEXT, description="JSON representation"),
                    Property(name="bbox", data_type=DataType.OBJECT, description="Bounding box coordinates"),
                    Property(name="num_rows", data_type=DataType.INT, description="Number of rows"),
                    Property(name="num_cols", data_type=DataType.INT, description="Number of columns"),
                ]
            )
            logger.info(f"Created collection: {settings.TABLE_CLASS}")

            # Create Figure collection
            self.client.collections.create(
                name=settings.FIGURE_CLASS,
                description="Figures with vision model descriptions",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="figure_id", data_type=DataType.TEXT, description="Unique figure identifier"),
                    Property(name="doc_id", data_type=DataType.TEXT, description="Document identifier"),
                    Property(name="page_num", data_type=DataType.INT, description="Page number"),
                    Property(name="description", data_type=DataType.TEXT, description="Vision model description"),
                    Property(name="image_path", data_type=DataType.TEXT, description="Path to extracted image"),
                    Property(name="bbox", data_type=DataType.OBJECT, description="Bounding box coordinates"),
                    Property(name="figure_type", data_type=DataType.TEXT, description="Type of figure (chart, graph, etc.)"),
                    Property(name="ocr_text", data_type=DataType.TEXT, description="OCR extracted text"),
                ]
            )
            logger.info(f"Created collection: {settings.FIGURE_CLASS}")

            logger.info("Schema creation complete")

        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            raise

    def generate_embeddings(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding (default from settings)

        Returns:
            Array of embeddings (shape: [len(texts), embedding_dim])
        """
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        logger.info(f"Generating embeddings for {len(texts)} texts")

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )

        return embeddings

    def insert_text_chunks(self, chunks: List[TextChunk], batch_size: int = 100):
        """
        Insert text chunks into Weaviate.

        Args:
            chunks: List of TextChunk objects
            batch_size: Batch size for insertion
        """
        if not chunks:
            logger.warning("No chunks to insert")
            return

        logger.info(f"Inserting {len(chunks)} text chunks")

        # Generate embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)

        # Get collection
        collection = self.client.collections.get(settings.TEXT_CHUNK_CLASS)

        # Insert in batches
        with collection.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                properties = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "page_num": chunk.page_num,
                    "section_title": chunk.section_title or "",
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "bbox": chunk.bbox.to_dict() if chunk.bbox else {},
                    "contains_statistics": chunk.metadata.get("contains_statistics", False),
                }

                batch.add_object(
                    properties=properties,
                    vector=embeddings[i].tolist()
                )

        logger.info(f"Successfully inserted {len(chunks)} chunks")

    def insert_table(
        self,
        table_id: str,
        doc_id: str,
        page_num: int,
        markdown: str,
        summary: str,
        json_structure: str = "",
        bbox: Optional[Dict] = None,
        num_rows: int = 0,
        num_cols: int = 0
    ):
        """
        Insert a table into Weaviate.

        Args:
            table_id: Unique table identifier
            doc_id: Document identifier
            page_num: Page number
            markdown: Table in markdown format
            summary: LLM-generated summary (for semantic search)
            json_structure: JSON representation
            bbox: Bounding box dictionary
            num_rows: Number of rows
            num_cols: Number of columns
        """
        # Generate embedding from summary (for semantic search)
        embedding = self.generate_embeddings([summary])[0]

        collection = self.client.collections.get(settings.TABLE_CLASS)

        properties = {
            "table_id": table_id,
            "doc_id": doc_id,
            "page_num": page_num,
            "markdown": markdown,
            "summary": summary,
            "json_structure": json_structure,
            "bbox": bbox or {},
            "num_rows": num_rows,
            "num_cols": num_cols,
        }

        collection.data.insert(
            properties=properties,
            vector=embedding.tolist()
        )

        logger.info(f"Inserted table: {table_id}")

    def insert_figure(
        self,
        figure_id: str,
        doc_id: str,
        page_num: int,
        description: str,
        image_path: str,
        bbox: Optional[Dict] = None,
        figure_type: str = "",
        ocr_text: str = ""
    ):
        """
        Insert a figure into Weaviate.

        Args:
            figure_id: Unique figure identifier
            doc_id: Document identifier
            page_num: Page number
            description: Vision model description (for semantic search)
            image_path: Path to extracted image
            bbox: Bounding box dictionary
            figure_type: Type of figure
            ocr_text: OCR extracted text
        """
        # Generate embedding from description
        embedding = self.generate_embeddings([description])[0]

        collection = self.client.collections.get(settings.FIGURE_CLASS)

        properties = {
            "figure_id": figure_id,
            "doc_id": doc_id,
            "page_num": page_num,
            "description": description,
            "image_path": image_path,
            "bbox": bbox or {},
            "figure_type": figure_type,
            "ocr_text": ocr_text,
        }

        collection.data.insert(
            properties=properties,
            vector=embedding.tolist()
        )

        logger.info(f"Inserted figure: {figure_id}")

    def search_text_chunks(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Search text chunks using semantic search.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters (e.g., {"doc_id": "doc123"})

        Returns:
            List of matching chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]

        collection = self.client.collections.get(settings.TEXT_CHUNK_CLASS)

        # Perform vector search
        results = collection.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )

        # Format results
        formatted_results = []
        for obj in results.objects:
            result = {
                "chunk_id": obj.properties["chunk_id"],
                "doc_id": obj.properties["doc_id"],
                "text": obj.properties["text"],
                "page_num": obj.properties["page_num"],
                "section_title": obj.properties["section_title"],
                "distance": obj.metadata.distance,
                "score": 1 - obj.metadata.distance  # Convert distance to similarity
            }
            formatted_results.append(result)

        logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")

        return formatted_results

    def delete_document(self, doc_id: str):
        """
        Delete all content for a specific document.

        Args:
            doc_id: Document identifier to delete
        """
        logger.info(f"Deleting all content for document: {doc_id}")

        for class_name in [settings.TEXT_CHUNK_CLASS, settings.TABLE_CLASS, settings.FIGURE_CLASS]:
            try:
                collection = self.client.collections.get(class_name)

                # Delete by doc_id filter
                collection.data.delete_many(
                    where={
                        "path": ["doc_id"],
                        "operator": "Equal",
                        "valueText": doc_id
                    }
                )

                logger.info(f"Deleted {class_name} entries for {doc_id}")

            except Exception as e:
                logger.error(f"Error deleting {class_name} for {doc_id}: {e}")

    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get statistics for all collections.

        Returns:
            Dictionary with counts for each collection
        """
        stats = {}

        for class_name in [settings.TEXT_CHUNK_CLASS, settings.TABLE_CLASS, settings.FIGURE_CLASS]:
            try:
                collection = self.client.collections.get(class_name)
                count = len(collection)
                stats[class_name] = count
            except Exception as e:
                logger.error(f"Error getting stats for {class_name}: {e}")
                stats[class_name] = 0

        return stats

    def close(self):
        """Close Weaviate connection."""
        if self.client:
            self.client.close()
            logger.info("Closed Weaviate connection")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
