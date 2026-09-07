"""embeddings determinísticos com cadeia de provedores (Gemini -> Ollama local -> Groq).

Todos os modelos configurados retornam 768 dimensões, que é a dimensão fixa do
schema no banco (vector(768)). Se algum provedor devolver outra dimensão, a
ingestão falha alto — não muda as tabelas.
"""

from typing import Callable
import httpx

from .config import config


class EmbeddingsError(RuntimeError):
    pass


def quantidade_ok() -> str:
    """descreve a cadeia de provedores disponível para mensagens de status."""
    return ", ".join(_provedores()) or "nenhum"


_TAMANHO_LOTE = 100


def _provedores() -> list[str]:
    """ordem de tentativas: primário primeiro, depois fallbacks já configurados."""
    ordem = [config.embedding_provider, *config.embedding_fallbacks]
    vistos: set[str] = set()
    result = []
    for p in ordem:
        p = p.strip().lower()
        if not p or p in vistos:
            continue
        if p == "gemini" and not config.gemini_api_key:
            continue
        if p == "groq" and not config.groq_api_key:
            continue
        if p == "ollama" and not config.ollama_url:
            continue
        if p == "hf" and not config.hf_token:
            continue
        if p not in ("gemini", "groq", "ollama", "hf"):
            continue
        vistos.add(p)
        result.append(p)
    return result


def _gemini_batch(textos: list[str]) -> list[list[float] | None]:
    modelo = f"models/{config.embedding_model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{modelo}:batchEmbedContents?key={config.gemini_api_key}"
    headers = {"x-goog-api-key": config.gemini_api_key}
    try:
        with httpx.Client(timeout=60) as c:
            r = c.post(
                url,
                json={
                    "model": modelo,
                    "requests": [
                        {
                            "model": modelo,
                            "content": {"parts": [{"text": t}]},
                            **({"outputDimensionality": config.embedding_dim} if config.embedding_dim else {}),
                        }
                        for t in textos
                    ],
                },
                headers=headers,
            )
        if r.status_code != 200:
            raise EmbeddingsError(f"gemini respondeu {r.status_code}: {r.text[:200]}")
        return [list(e["values"]) for e in r.json().get("embeddings") or []]
    except httpx.HTTPError as e:
        raise EmbeddingsError(f"falha de rede no provedor gemini: {e}")


def _ollama_batch(textos: list[str]) -> list[list[float] | None]:
    base = config.ollama_url
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        # Adicionar autenticação se configurada
        auth = None
        if config.ollama_username and config.ollama_password:
            from httpx import BasicAuth
            auth = BasicAuth(config.ollama_username, config.ollama_password)
        
        with httpx.Client(timeout=120) as c:
            r = c.post(
                f"{base}/api/embed",
                json={"model": config.ollama_embedding_model, "input": textos},
                auth=auth
            )
        if r.status_code != 200:
            raise EmbeddingsError(f"ollama respondeu {r.status_code}: {r.text[:200]}")
        return [list(e) for e in r.json().get("embeddings") or []]
    except httpx.HTTPError as e:
        raise EmbeddingsError(f"falha de rede no provedor ollama: {e}")


def _groq_batch(textos: list[str]) -> list[list[float] | None]:
    url = "https://api.groq.com/openai/v1/embeddings"
    headers = {"Authorization": f"Bearer {config.groq_api_key}"}
    try:
        with httpx.Client(timeout=60) as c:
            r = c.post(
                url,
                json={"model": config.groq_embedding_model, "input": textos, "encoding_format": "float"},
                headers=headers,
            )
        if r.status_code != 200:
            raise EmbeddingsError(f"groq respondeu {r.status_code}: {r.text[:200]}")
        dados = sorted(r.json().get("data") or [], key=lambda d: d["index"])
        return [list(d["embedding"]) for d in dados]
    except httpx.HTTPError as e:
        raise EmbeddingsError(f"falha de rede no provedor groq: {e}")


def _hf_batch(textos: list[str]) -> list[list[float] | None]:
    """nomic-embed-text-v1.5 gratuito do Hugging Face (mesmo modelo do ollama local)."""
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{config.hf_embedding_model}"
    headers = {"Authorization": f"Bearer {config.hf_token}"}
    try:
        with httpx.Client(timeout=60) as c:
            r = c.post(
                url,
                json={"inputs": [t for t in textos], "options": {"wait_for_model": True}},
                headers=headers,
            )
        if r.status_code != 200:
            raise EmbeddingsError(f"hugging face respondeu {r.status_code}: {r.text[:200]}")
        dados = r.json()
        if not isinstance(dados, list) or not dados or not isinstance(dados[0], list):
            raise EmbeddingsError("hugging face devolveu formato inesperado")
        return [list(e) for e in dados]
    except httpx.HTTPError as e:
        raise EmbeddingsError(f"falha de rede no provedor hugging face: {e}")


_EMBED_FN = {
    "gemini": _gemini_batch,
    "ollama": _ollama_batch,
    "groq": _groq_batch,
    "hf": _hf_batch,
}


def _provedor_embed(fatia: list[str]) -> tuple[str | None, list[list[float] | None]]:
    for p in _provedores():
        try:
            saida = _EMBED_FN[p](fatia)
        except EmbeddingsError:
            continue
        if len(saida) < len(fatia):
            continue
        return p, saida
    return None, [None] * len(fatia)


def gerar_embeddings_lote(textos: list[str], progresso: Callable | None = None) -> list[list[float] | None]:
    """embeddings em lotes, tentando cada provedor da cadeia; None = falhou em todos."""
    if not _provedores():
        raise EmbeddingsError(
            "nenhum provedor de embeddings configurado — defina GEMINI_API_KEY, "
            "GROQ_API_KEY e/ou OLLAMA_URL"
        )
    dim = config.embedding_dim
    saida: list[list[float] | None] = [None] * len(textos)
    for ini in range(0, len(textos), _TAMANHO_LOTE):
        fatia = textos[ini : ini + _TAMANHO_LOTE]
        provedor, emb = _provedor_embed(fatia)
        for j, valores in enumerate(emb):
            if isinstance(valores, list):
                if dim and len(valores) != dim:
                    raise EmbeddingsError(
                        f"{provedor} devolveu {len(valores)} dimensões, schema espera {dim}"
                    )
                saida[ini + j] = valores
        if progresso:
            progresso(ini + len(fatia))
    return saida


def gerar_embedding(texto: str) -> list[float]:
    """embedding de um texto, seguindo a mesma cadeia de provedores."""
    res = gerar_embeddings_lote([texto])
    if not res or res[0] is None:
        raise EmbeddingsError("embedding falhou em todos os provedores disponíveis")
    return res[0]