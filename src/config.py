"""Configuration management for the Research Citations MCP Server."""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings and LLM")

    # Paths
    papers_directory: Path = Field(
        ..., description="Directory containing research papers (PDFs)"
    )
    vector_db_path: Path = Field(
        default=Path("./vector_db"), description="Path to store vector database"
    )

    # Database settings
    collection_name: str = Field(
        default="research_papers", description="Name of the vector store collection"
    )

    # Text processing settings
    chunk_size: int = Field(
        default=1000, description="Size of text chunks for processing"
    )
    chunk_overlap: int = Field(
        default=200, description="Overlap between consecutive chunks"
    )

    # Model settings
    embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    llm_model: str = Field(
        default="gpt-4o", description="LLM model for query processing"
    )

    # Server settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")

    @field_validator("papers_directory")
    @classmethod
    def validate_papers_directory(cls, v: Path) -> Path:
        """Validate that papers directory exists."""
        if not v.exists():
            raise ValueError(f"Papers directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Papers directory is not a directory: {v}")
        return v.resolve()

    @field_validator("vector_db_path")
    @classmethod
    def resolve_vector_db_path(cls, v: Path) -> Path:
        """Resolve vector database path."""
        return v.resolve()


# Global settings instance
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global settings
    if settings is None:
        settings = Settings()
    return settings
