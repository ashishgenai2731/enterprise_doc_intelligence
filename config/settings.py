from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise Document Intelligence"
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "doc-intelligence"

    # OPENAI_API_KEY , need this for evaluation ragas but this need credit in OPENAI
    # so we can use local Ollama for evaluation ragas also, see the evaluate_ragas.py
    # code , it is updated.
    # we can add this as Optional.
    OPENAI_API_KEY: Optional[str] = ""

    # Model Configurations
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-large"
    BASE_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # Chunking Parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Redis Cache Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Use SettingsConfigDict for Pydantic V2 compatibility
    # The extra="ignore" Parameter: By default, Pydantic V2 has strict validation. If it
    # encounters any environment variable in your .env file or terminal session that isn't explicitly
    # defined as a field in your Settings class, it raises an Extra inputs are not permitted validation error.
    # Adding extra="ignore" tells Pydantic to silently skip any extra or undeclared environment variables
    # instead of crashing your application.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()