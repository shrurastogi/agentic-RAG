"""
Central configuration management for the Agentic RAG system.
Uses Pydantic Settings for environment variable management.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Project paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    RAW_DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "raw")
    PROCESSED_DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "processed")
    CACHE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "cache")

    # LLM Configuration
    LLM_MODEL: str = "llama3.1:8b"
    LLM_BASE_URL: str = "http://localhost:11434"  # Ollama default
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_TOP_P: float = 0.9

    # Vision Model
    VISION_MODEL: str = "llama3.2-vision:11b"
    VISION_TEMPERATURE: float = 0.2

    # Embedding Configuration
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"  # "cpu" or "cuda"

    # Chunking Configuration
    CHUNK_SIZE: int = 512  # tokens
    CHUNK_OVERLAP: int = 100  # tokens
    MIN_CHUNK_SIZE: int = 128  # tokens
    PRESERVE_SECTION_BOUNDARIES: bool = True

    # Table Extraction
    TABLE_DETECTION_METHOD: str = "transformer"  # "transformer" or "camelot"
    TABLE_SUMMARY_MAX_LENGTH: int = 256
    GENERATE_TABLE_SUMMARIES: bool = True

    # Figure Processing
    FIGURE_DESCRIPTION_MAX_LENGTH: int = 512
    APPLY_OCR_TO_FIGURES: bool = True
    OCR_LANGUAGES: list[str] = Field(default_factory=lambda: ["en"])

    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 15  # Initial retrieval count
    TOP_K_RERANK: int = 8  # After reranking
    TOP_K_TABLES: int = 5
    TOP_K_FIGURES: int = 3
    HYBRID_ALPHA: float = 0.5  # Balance between BM25 (0) and vector (1)

    # Reranking
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    ENABLE_RERANKING: bool = True

    # Weaviate Configuration
    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_API_KEY: Optional[str] = None
    WEAVIATE_TIMEOUT: int = 30

    # Weaviate Collection Names
    TEXT_CHUNK_CLASS: str = "TextChunk"
    TABLE_CLASS: str = "Table"
    FIGURE_CLASS: str = "Figure"

    # Caching Configuration
    ENABLE_CACHE: bool = True
    CACHE_TTL_HOURS: int = 24
    EMBEDDING_CACHE_TTL_HOURS: int = 168  # 7 days
    RESPONSE_CACHE_TTL_HOURS: int = 168  # 7 days

    # Citation Configuration
    ENABLE_CITATIONS: bool = True
    CITATION_FORMAT: str = "inline"  # "inline" or "footnote"
    VERIFY_CITATIONS: bool = True

    # Agent Configuration
    MAX_AGENT_ITERATIONS: int = 10
    AGENT_TIMEOUT_SECONDS: int = 120
    ENABLE_MULTI_DOC_REASONING: bool = True

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False
    API_WORKERS: int = 1

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "rich"  # "json" or "rich"
    LOG_FILE: Optional[str] = None

    # Medical Domain Specific
    PRESERVE_STATISTICS: bool = True
    MEDICAL_TERMINOLOGY_MODE: bool = True
    CONFIDENCE_INTERVAL_PRECISION: int = 2

    # Performance
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 10
    ENABLE_GPU: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        for path_field in ["DATA_DIR", "RAW_DATA_DIR", "PROCESSED_DATA_DIR", "CACHE_DIR"]:
            path: Path = getattr(self, path_field)
            path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
