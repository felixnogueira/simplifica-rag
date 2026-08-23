import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core import agente, camara, rag
from core.config import config

app = FastAPI(title="Simplifica Legislativo", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _erro(msg: str, status: int = 503) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detalhe": msg})


def _doc(p: dict) -> dict:
    return {
        "titulo": f"{p.get('siglaTipo', '')} {p.get('numero', '')}/{p.get('ano', '')}".strip(),
        "tipo": p.get("siglaTipo", ""),
        "subtipo": "",
        "numero": str(p.get("numero", "")),
        "ano": p.get("ano"),
        "data": (p.get("dataApresentacao") or "")[:10],
        "situacao": "",
        "autores": "",
        "protocolo": "",
        "url": p.get("url", ""),
        "ementa": p.get("ementa"),
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "camara": "api nacional",
        "llm": bool(config.tem_chaves),
        "rag": bool(config.tem_rag),
    }


@app.get("/api/rag/status")
def rag_status():
    return {
        "disponivel": bool(config.tem_rag),
        "indice": rag.status(),
        "modelo_embedding": config.embedding_model,
        "ano_vigente": date.today().year,
    }


@app.get("/api/filtros")
def filtros():
    try:
        ano_atual = date.today().year
        return {
            "anos": list(range(ano_atual - 4, ano_atual + 1))[::-1],
            "tipos": [x["sigla"] for x in camara.siglas_tipos() if x.get("sigla")][:50],
            "situacoes": [x["nome"] for x in camara.situacoes_proposicao() if x.get("nome")][:50],
            "autores": [],
        }
    except RuntimeError as e:
        return _erro(str(e))


@app.get("/api/docs")
def docs(
    q: str | None = Query(default=None),
    ano: int | None = Query(default=None),
    tipo: str | None = Query(default=None),
    situacao: str | None = Query(default=None),
    autor: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    pagina: int = Query(default=1, ge=1),
):
    try:
        params = {"itens": limit, "pagina": pagina}
        if ano:
            params["ano"] = ano
        if tipo:
            params["sigla_tipo"] = tipo
        if q:
            params["keywords"] = q
        if autor:
            params["id_autor"] = camara.buscar_deputado(autor)["id"]
        itens = camara.proposicoes(**params)
        return {"count": len(itens), "results": [_doc(p) for p in itens], "pagina": pagina}
    except (RuntimeError, camara.CamaraError) as e:
        return _erro(str(e))


@app.post("/api/chat")
def chat(payload: dict):
    pergunta = (payload.get("pergunta") or "").strip()
    if not pergunta:
        return _erro("pergunta vazia", 400)
    try:
        out = agente.responder(pergunta)
        return {
            "resposta": out["resposta"],
            "fontes": out["fontes"],
            "raciocinio": out.get("raciocinio") or "",
            "aviso": out.get("aviso") or "",
            "ano_vigente": out.get("ano_vigente"),
        }
    except (RuntimeError, camara.CamaraError) as e:
        return _erro(str(e))