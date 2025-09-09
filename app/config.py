# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    API_VERSION: str = "v1"
    API_PREFIX: str = f"/api/{API_VERSION}"
    PROJECT_NAME: str = "AI Loan Support Backend"

    # LLM Configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Vector DB Configuration
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "loan-support-v1"
    PINECONE_ENVIRONMENT: str = "us-west1-gcp-free"  # (kept for compat if you use old client)
    PINECONE_NAMESPACE: str = "default"              # <-- ADD THIS
    SBERT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"  # <-- ADD THIS
    USE_LOCAL_FAISS: bool = True

    # Security
    BEARER_TOKEN: str = "dev-token-change-in-production"
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour

    # Performance
    MAX_RETRIEVAL_DOCS: int = 5
    RETRIEVAL_TIMEOUT: float = 1.2
    LLM_TIMEOUT: float = 3.5

    # PII Redaction
    MASK_PAN: bool = True
    MASK_AADHAAR: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()
