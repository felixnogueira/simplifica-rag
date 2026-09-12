import time
from urllib.parse import urlparse

import httpx


class LlmError(RuntimeError):
    pass


class LmClient:
    """cliente openai-compatible para gemini e groq."""

    gemini_base = "https://generativelanguage.googleapis.com/v1beta"
    groq_base = "https://api.groq.com/openai/v1"
    zen_base = "https://opencode.ai/zen/v1"

    def __init__(self) -> None:
        pass

    def _post(self, url: str, api_key: str, payload: dict, auth: str = "bearer",
              tentativas: int = 3) -> dict:
        from .config import config

        basic_auth = (
            httpx.BasicAuth(config.ollama_username or "", config.ollama_password or "")
            if auth == "basic"
            else None
        )
        headers = (
            {"x-goog-api-key": api_key}
            if auth == "x-goog-api-key"
            else ({} if auth == "basic" else {"Authorization": f"Bearer {api_key}"})
        )
        ultimo_erro = None
        for n in range(tentativas):
            try:
                with httpx.Client(
                    timeout=httpx.Timeout(config.llm_timeout, connect=config.llm_connect_timeout),
                    auth=basic_auth,
                ) as client:
                    r = client.post(url, json=payload, headers=headers)
                if r.status_code in (429, 500, 502, 503, 504):
                    retry = self._retry_delay(r)
                    self._espera(retry)
                    ultimo_erro = httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                    continue
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError:
                self._espera(1.5 * (n + 1))
                continue
        if isinstance(ultimo_erro, httpx.HTTPError):
            raise ultimo_erro
        raise LlmError("sem resposta da api")

    @staticmethod
    def _retry_delay(resposta: httpx.Response) -> float:
        """espera sugerida pela api no corpo/header de 429 e 503."""
        try:
            detalhes = resposta.json().get("error", {}).get("details", [])
            for d in detalhes:
                atraso = (d.get("retryDelay") or "").rstrip("s")
                if atraso:
                    return max(2.0, float(atraso))
        except (ValueError, AttributeError, TypeError):
            pass
        tentativa = resposta.headers.get("retry-after")
        if tentativa:
            try:
                return max(2.0, float(tentativa))
            except ValueError:
                pass
        return 3.0

    @staticmethod
    def _espera(segundos: float) -> None:
        time.sleep(segundos)

    def completar(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        temperatura: float = 0.2,
    ) -> dict:
        """chamada openai-compatible; retorna a mensagem completa (pode conter tool_calls)."""
        from .config import config

        openai_compat = "https://generativelanguage.googleapis.com/v1beta/openai"
        base_payload = {
            "messages": messages,
            "temperature": temperatura,
            "max_tokens": config.llm_max_tokens,
        }
        if tools:
            base_payload["tools"] = tools
        providers = [
            (f"{config.ollama_url}/chat/completions",
             "ollama-local" if config.ollama_url else "",
             dict(base_payload, model=config.ollama_chat_model),
             "basic"),
            (f"{config.workers_ai_url}/v1/chat/completions",
             "workers-ai-local" if config.workers_ai_url else "",
             dict(base_payload, model=config.workers_ai_model),
             "bearer"),
            (f"{self.zen_base}/chat/completions", self._apikey_zen(),
             dict(base_payload, model=config.opencode_chat_model),
             "bearer"),
            (f"{openai_compat}/chat/completions", self._apikey_gemini(),
             dict(base_payload, model=model or config.chat_model),
             "x-goog-api-key"),
            (f"{self.groq_base}/chat/completions", self._apikey_groq(),
             dict(base_payload, model=config.groq_chat_model),
             "bearer"),
        ]
        def _varredura() -> tuple[dict | None, list[str], Exception | None]:
            falhas: list[str] = []
            ult_falha: Exception | None = None
            for url, _apikey, payload, esquema_auth in providers:
                if not url or not _apikey:
                    continue
                try:
                    mensagem = self._post(url, _apikey, payload, auth=esquema_auth)
                    return mensagem["choices"][0]["message"], [], None
                except Exception as e:
                    ult_falha = e
                    nome = (urlparse(url).hostname or url).split(".")[0]
                    falhas.append(f"{nome}: {type(e).__name__}")
            return None, falhas, ult_falha

        inicio = time.monotonic()
        mensagem, falhas, ult_falha = _varredura()
        if mensagem is None and time.monotonic() - inicio < 60:
            time.sleep(1.5)
            mensagem, falhas, ult_falha = _varredura()
        if mensagem is not None:
            return mensagem
        raise LlmError("falha ao chamar llm — provedores testados: " + "; ".join(falhas)) from ult_falha

    def chat(self, system: str, user: str, model: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        msg = self.completar(messages, model=model)
        return msg.get("content") or ""

    def _apikey_gemini(self) -> str:
        from .config import config

        return config.gemini_api_key

    def _apikey_zen(self) -> str:
        from .config import config

        return config.opencode_zen_api_key

    def _apikey_groq(self) -> str:
        from .config import config

        return config.groq_api_key


lm = LmClient()