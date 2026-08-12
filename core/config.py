import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    opencode_zen_api_key: str = os.getenv("OPENCODE_ZEN_API_KEY", "")
    ollama_url: str = os.getenv("OLLAMA_URL", "")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
    workers_ai_url: str = os.getenv("WORKERS_AI_URL", "")
    workers_ai_model: str = os.getenv("WORKERS_AI_MODEL", "@cf/qwen/qwen3-30b-a3b-fp8")

    chat_model: str = os.getenv("CHAT_MODEL", "gemini-3.6-flash")
    groq_chat_model: str = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
    opencode_chat_model: str = os.getenv("OPENCODE_CHAT_MODEL", "big-pickle")
    opencode_zen_base: str = os.getenv("OPENCODE_ZEN_BASE", "https://opencode.ai/zen/v1")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "300"))
    llm_connect_timeout: float = float(os.getenv("LLM_CONNECT_TIMEOUT", "20"))

    camara_base: str = os.getenv(
        "CAMARA_API_URL", "https://dadosabertos.camara.leg.br/api/v2"
    )

    database_url: str = os.getenv("DATABASE_URL", "")
    qdrant_endpoint: str = os.getenv("QDRANT_ENDPOINT", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_colecao: str = os.getenv("QDRANT_COLECAO", "proposicoes")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "gemini")
    embedding_fallbacks: list = [
        x.strip() for x in os.getenv("EMBEDDING_FALLBACKS", "ollama,groq").split(",") if x.strip()
    ]
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    groq_embedding_model: str = os.getenv("GROQ_EMBEDDING_MODEL", "nomic-embed-text-v1.5")
    hf_token: str = os.getenv("HF_TOKEN", "")
    hf_embedding_model: str = os.getenv("HF_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "8"))
    rag_desde: int = int(os.getenv("RAG_ANOS_DE", "2016"))

    @property
    def tem_chaves(self) -> bool:
        return bool(
            self.gemini_api_key
            or self.groq_api_key
            or self.opencode_zen_api_key
            or self.ollama_url
            or self.workers_ai_url
        )

    @property
    def tem_rag(self) -> bool:
        return bool(self.database_url or self.qdrant_endpoint)

    @property
    def tem_qdrant(self) -> bool:
        return bool(self.qdrant_endpoint and self.qdrant_api_key)


config = Config()