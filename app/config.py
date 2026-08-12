from functools import lru_cache
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "Enterprise Agentic RAG"
    gemini_api_key: str = ""
    gemini_embedding_model: str = "models/text-embedding-004"
    embedding_dimensions: int = Field(3072, ge=128, le=3072)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_documents"
    embedding_cache_size: int = Field(2048, ge=1, le=100_000)
    dense_candidate_limit: int = Field(16, ge=1)
    bm25_candidate_limit: int = Field(16, ge=1)
    rrf_candidate_limit: int = Field(8, ge=1)
    rerank_candidate_limit: int = Field(8, ge=1)
    top_k: int = Field(5, ge=1, le=100)
    context_token_budget: int = Field(1200, ge=200, le=8000)
    llm_provider: str = "local"
    portkey_api_key: str = ""
    portkey_virtual_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    @model_validator(mode="after")
    def validate_candidate_order(self):
        if self.rrf_candidate_limit > self.dense_candidate_limit + self.bm25_candidate_limit:
            raise ValueError("rrf_candidate_limit cannot exceed retrieval candidates")
        return self
    @property
    def qdrant_enabled(self) -> bool: return bool(self.qdrant_url)
@lru_cache
def get_settings() -> Settings: return Settings()
