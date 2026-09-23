# Simplifica Legislativo — backend

API de chat em linguagem simples sobre proposições legislativas da Câmara dos Deputados (âmbito nacional). As respostas são baseadas em **RAG**: o contexto é recuperado de forma determinística de um índice vetorial e a LLM **apenas redige** a resposta a partir desse contexto — ela não inventa dados.

## como funciona

1. **Ingestão** (local): baixa as proposições da API de dados abertos da Câmara, monta um documento por proposição, gera os embeddings e indexa no armazenamento configurado.
2. **Consulta on-line** (`POST /api/chat`): o embedding da pergunta é gerado na hora e busca o contexto no índice → a LLM compõe a resposta **somente a partir do contexto** recuperado, citando cada proposição (vira botão no frontend).
3. **Sem dependência da LLM**: se a busca não retornar nada, avisa que não encontrou; se não houver chave de LLM, responde de forma determinística formatando os documentos recuperados.

## rodar local

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn api.index:app --port 8000
```

## variáveis de ambiente

Copie `.env.example` para `.env` e preencha (chaves de LLM e do provedor de embeddings). A API responde mesmo sem chaves, via formatação determinística. `RAG_ANOS_DE` define o primeiro ano da ingestão padrão (padrão 2016) e `RAG_TOP_K`, quantos documentos são recuperados por pergunta (padrão 8).

## ingestão (RAG)

```bash
python -m core.ingestao --anos 2016-2026
```

Os arquivos baixados ficam cacheados em `dados/` (gitignored). A ingestão é **retomável por documento**: rode o mesmo comando de novo que ele pula o que já foi gravado. Para retomar, evite `--forcar` e `--limpar`.

## endpoints

- `GET /api/health` — status da api, chaves de llm e RAG
- `GET /api/rag/status` — total de proposições indexadas, abrangência de anos e modelo de embedding
- `GET /api/filtros` — anos, tipos de proposição e situações para montar filtros
- `GET /api/docs` — lista proposições (filtros `q`, `ano`, `tipo`, `autor`, `limit`, `pagina`)
- `POST /api/chat` — pergunta em linguagem natural; responde com `resposta` e `fontes` (proposições recuperadas do índice)