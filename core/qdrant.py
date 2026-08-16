"""busca vetorial RAG sobre proposições da Câmara (Qdrant Cloud).

ponto id = id da proposição; payload = todos os campos do documento mais
listas planas (autores_nomes, autores_resumo, autores_partidos, autores_ufs,
temas_lista) para filtragem e as agregações determinísticas via facet.
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models as m

from .config import config

_SEP = "\x1f"


class RAGError(RuntimeError):
    pass


_cliente: QdrantClient | None = None


def conectar() -> QdrantClient:
    global _cliente
    if _cliente is not None:
        return _cliente
    endpoint = config.qdrant_endpoint
    if not endpoint:
        raise RAGError("QDRANT_ENDPOINT ausente — configure o Qdrant Cloud")
    try:
        if endpoint == ":memory:":
            _cliente = QdrantClient(":memory:")
        else:
            _cliente = QdrantClient(
                url=endpoint, api_key=config.qdrant_api_key, timeout=60
            )
    except Exception as e:
        raise RAGError(f"falha ao conectar no Qdrant: {e}")
    return _cliente


def _colecao() -> str:
    return config.qdrant_colecao


def _existe(c, nome: str) -> bool:
    try:
        c.get_collection(nome)
        return True
    except Exception:
        return False


def _garantir_indices(c) -> None:
    """índices de payload exigidos por facet/filtros/texto (idempotente)."""
    inds = {
        "ano": m.PayloadSchemaType.INTEGER,
        "sigla_tipo": m.PayloadSchemaType.KEYWORD,
        "autores_nomes": m.PayloadSchemaType.KEYWORD,
        "autores_resumo": m.PayloadSchemaType.KEYWORD,
        "autores_partidos": m.PayloadSchemaType.KEYWORD,
        "autores_ufs": m.PayloadSchemaType.KEYWORD,
        "temas_lista": m.PayloadSchemaType.KEYWORD,
        "doc": m.PayloadSchemaType.TEXT,
    }
    for campo, schema in inds.items():
        try:
            c.create_payload_index(
                _colecao(), campo, field_schema=schema, wait=True
            )
        except Exception:
            pass


def garantir_esquema() -> None:
    c = conectar()
    if not _existe(c, _colecao()):
        c.create_collection(
            collection_name=_colecao(),
            vectors_config=m.VectorParams(
                size=config.embedding_dim, distance=m.Distance.COSINE
            ),
        )
    _garantir_indices(c)


def recriar_esquema() -> None:
    c = conectar()
    if _existe(c, _colecao()):
        c.delete_collection(_colecao())
    garantir_esquema()


def _autores_planos(autores_detalhe: list[dict]) -> dict:
    """converte autores_detalhe em listas planas deduplicadas para payload."""
    nomes, resumo, partidos, ufs = [], [], [], []
    vistos = set()
    for a in autores_detalhe:
        nome = (a.get("nome") or "").strip()
        partido = (a.get("partido") or "").upper()
        uf = (a.get("uf") or "").upper()
        if not nome:
            continue
        chave = (nome, partido, uf)
        if chave in vistos:
            continue
        vistos.add(chave)
        nomes.append(nome)
        resumo.append(f"{nome}{_SEP}{partido}{_SEP}{uf}")
        if partido:
            partidos.append(partido)
        if uf:
            ufs.append(uf)
    return {
        "autores_nomes": nomes,
        "autores_resumo": resumo,
        "autores_partidos": partidos,
        "autores_ufs": ufs,
    }


def _payload(d: dict) -> dict:
    pl = {k: d[k] for k in (
        "id", "sigla_tipo", "numero", "ano", "ementa", "temas", "autores",
        "siglas_autores", "situacao", "url", "data", "doc",
    ) if k in d}
    pl["ano"] = int(d.get("ano") or 0)
    pl.update(_autores_planos(d.get("autores_detalhe") or []))
    pl["temas_lista"] = list(dict.fromkeys(d.get("temas_lista") or []))
    return pl


def upsert_docs(docs: list[dict]) -> int:
    if not docs:
        return 0
    pontos = []
    for d in docs:
        emb = d.get("embedding")
        if not emb:
            continue
        pontos.append(m.PointStruct(id=int(d["id"]), vector=list(emb), payload=_payload(d)))
    if not pontos:
        return 0
    c = conectar()
    for i in range(0, len(pontos), 64):
        c.upsert(_colecao(), points=pontos[i : i + 64], wait=True)
    return len(pontos)


def registrados(ids: list[int]) -> set[int]:
    ids = list(ids)
    if not ids:
        return set()
    c = conectar()
    encontrados: set[int] = set()
    for i in range(0, len(ids), 500):
        try:
            pts = c.retrieve(
                _colecao(), ids=ids[i : i + 500],
                with_payload=False, with_vectors=False,
            )
            encontrados.update(p.id for p in pts)
        except Exception:
            break
    return encontrados


def _filtros(filtros: dict | None) -> m.Filter | None:
    """filtro determinístico a partir de anos/sigla_tipo/partido/uf."""
    f = filtros or {}
    condicoes = []
    anos = [int(a) for a in (f.get("anos") or [])]
    if anos:
        condicoes.append(m.FieldCondition(key="ano", match=m.MatchAny(any=anos)))
    sigla = (f.get("sigla_tipo") or "").strip().upper()
    if sigla:
        condicoes.append(m.FieldCondition(key="sigla_tipo", match=m.MatchValue(value=sigla)))
    partido = (f.get("partido") or "").strip().upper()
    if partido:
        condicoes.append(
            m.FieldCondition(key="autores_partidos", match=m.MatchAny(any=[partido]))
        )
    uf = (f.get("uf") or "").strip().upper()
    if uf:
        condicoes.append(
            m.FieldCondition(key="autores_ufs", match=m.MatchAny(any=[uf]))
        )
    if not condicoes:
        return None
    return m.Filter(must=condicoes)


def _colunas(ponto) -> dict:
    p = ponto.payload
    return {
        "id": p.get("id"),
        "sigla_tipo": p.get("sigla_tipo", ""),
        "numero": p.get("numero", ""),
        "ano": p.get("ano"),
        "ementa": p.get("ementa", ""),
        "temas": p.get("temas", ""),
        "autores": p.get("autores", ""),
        "situacao": p.get("situacao", ""),
        "url": p.get("url", ""),
        "data": p.get("data", ""),
        "doc": p.get("doc", ""),
    }


def buscar(pergunta: str, filtros: dict | None = None, top_k: int | None = None) -> list[dict]:
    from . import embeddings

    top_k = top_k or config.rag_top_k
    emb = embeddings.gerar_embedding(pergunta or "")
    if not emb:
        return []
    c = conectar()
    resp = c.query_points(
        _colecao(),
        query=list(emb),
        query_filter=_filtros(filtros),
        limit=top_k,
        with_payload=True,
    )
    return [{**_colunas(p), "score": float(p.score)} for p in resp.points]


def busca_texto(filtros: dict | None = None, termos: str = "", top_k: int | None = None) -> list[dict]:
    """fallback sem embeddings: MatchText sobre doc/ementa + filtro em python."""
    top_k = top_k or config.rag_top_k
    base = _filtros(filtros)
    termos_limpos = [t for t in (termos or "").lower().split() if len(t) > 3]
    if palavras := termos_limpos:
        texto = m.FieldCondition(
            key="doc",
            match=m.MatchText(text=" ".join(palavras)),
        )
        filtro = m.Filter(must=((base.must or []) + [texto]) if base else [texto])
    else:
        filtro = base
    if not filtro:
        return []
    c = conectar()
    linhas: list[dict] = []
    offset = None
    while len(linhas) < top_k:
        pts, offset = c.scroll(
            _colecao(), scroll_filter=filtro, limit=200,
            offset=offset, with_payload=True,
        )
        for p in pts:
            doc = (p.payload.get("doc") or "").lower()
            ementa = (p.payload.get("ementa") or "").lower()
            if any(t in doc for t in palavras) or any(t in ementa for t in palavras):
                linhas.append(_colunas(p))
            if len(linhas) >= top_k * 3:
                break
        if len(linhas) >= top_k * 3 or offset is None:
            break
    linhas.sort(key=lambda r: (r["ano"] or 0), reverse=True)
    return linhas[:top_k]


def _facet(key: str, filtro: m.Filter | None, limite: int) -> list[dict]:
    """hits do facet: [{valor, total}] ordenado por total desc."""
    c = conectar()
    try:
        resp = c.facet(
            _colecao(), key=key,
            facet_filter=filtro, limit=limite, exact=True,
        )
    except Exception:
        return []
    out = []
    for h in resp.hits:
        valor = h.value
        if isinstance(valor, str):
            out.append({"valor": valor, "total": int(h.count)})
    return out


class _AutorResumo:
    """composite 'nome\x1fPARTIDO\x1fUF' com parse."""

    @staticmethod
    def separar(valor: str) -> tuple[str, str, str]:
        partes = valor.split(_SEP)
        while len(partes) < 3:
            partes.append("")
        return partes[0], partes[1].upper(), partes[2].upper()


def rank_deputados(uf: str | None = None, partido: str | None = None,
                   ano: int | None = None, limite: int = 10) -> list[dict]:
    """deputados (com partido/uf e total de proposições autoradas)."""
    filtro = _filtros({"anos": [ano] if ano else [], "partido": partido, "uf": uf})
    hits = _facet("autores_resumo", filtro, max(limite * 10, 200))
    soma: dict = {}
    for h in hits:
        nome, hp, hu = _AutorResumo.separar(h["valor"])
        if partido and hp and hp != partido:
            continue
        if uf and hu and hu != uf:
            continue
        resumo = soma.setdefault(nome, {"nome": nome, "partido": hp, "uf": hu, "total": 0})
        resumo["total"] += h["total"]
        if not resumo["partido"] and hp:
            resumo["partido"] = hp
        if not resumo["uf"] and hu:
            resumo["uf"] = hu
    linhas = sorted(soma.values(), key=lambda r: (-r["total"], r["nome"]))
    return [
        {"nome": r["nome"], "partido": r["partido"], "uf": r["uf"], "total": r["total"]}
        for r in linhas[:int(limite)]
    ]


def count_partido(partido: str, uf: str | None = None, ano: int | None = None) -> int | None:
    """total de proposições cujos autores (deputados) são do partido."""
    try:
        c = conectar()
        res = c.count(
            _colecao(),
            count_filter=_filtros({"anos": [ano] if ano else [], "partido": partido, "uf": uf}),
            exact=True,
        )
        return int(res.count)
    except Exception:
        return None


def rank_partidos(uf: str | None = None, ano: int | None = None, limite: int = 10) -> list[dict]:
    """partidos (sigla e total de proposições autoradas por seus deputados)."""
    filtro = _filtros({"anos": [ano] if ano else [], "uf": uf})
    hits = _facet("autores_partidos", filtro, int(limite))
    return [
        {"partido": h["valor"], "total": h["total"]}
        for h in hits[: int(limite)]
    ]


def dados_deputado(nome: str) -> dict | None:
    """partido/uf atual (mais frequente) de um deputado no índice."""
    if not nome:
        return None
    busca = nome.casefold()
    hits = _facet("autores_resumo", None, 200)
    melhor: dict | None = None
    for h in hits:
        hi_nome, hp, hu = _AutorResumo.separar(h["valor"])
        if busca not in hi_nome.casefold():
            continue
        if melhor is None or h["total"] > melhor["total"]:
            melhor = {
                "nome": hi_nome, "partido": hp, "uf": hu, "total": h["total"],
            }
    return melhor


def temas_top(uf: str | None = None, partido: str | None = None,
              ano: int | None = None, limite: int = 10) -> list[dict]:
    """temas mais frequentes num grupo de proposições do índice."""
    filtro = _filtros({"anos": [ano] if ano else [], "partido": partido, "uf": uf})
    hits = _facet("temas_lista", filtro, int(limite))
    return [
        {"tema": h["valor"], "total": h["total"]}
        for h in hits[: int(limite)]
    ]


def temas_de_deputado(nome: str, ano: int | None = None, limite: int = 3) -> list[dict]:
    """temas mais frequentes nas proposições de um deputado."""
    filtro = m.Filter(must=[
        m.FieldCondition(key="autores_nomes", match=m.MatchValue(value=nome)),
    ])
    if ano:
        filtro.must.append(m.FieldCondition(key="ano", match=m.MatchValue(value=int(ano))))
    c = conectar()
    pts, _ = c.scroll(_colecao(), scroll_filter=filtro, limit=200, with_payload=True)
    soma: dict = {}
    for p in pts:
        for t in p.payload.get("temas_lista") or []:
            soma[t] = soma.get(t, 0) + 1
    out = sorted(soma.items(), key=lambda kv: (-kv[1], kv[0]))[: int(limite)]
    return [{"tema": t, "total": n} for t, n in out]


def proposicoes_de_deputado(nome: str, ano: int | None = None, limite: int = 6) -> list[dict]:
    """proposições de um deputado (para fontes/botões)."""
    filtro = m.Filter(must=[
        m.FieldCondition(key="autores_nomes", match=m.MatchValue(value=nome)),
    ])
    if ano:
        filtro.must.append(m.FieldCondition(key="ano", match=m.MatchValue(value=int(ano))))
    c = conectar()
    pts, _ = c.scroll(_colecao(), scroll_filter=filtro, limit=200, with_payload=True)
    linhas = [
        {
            "id": p.payload.get("id"), "sigla_tipo": p.payload.get("sigla_tipo", ""),
            "numero": p.payload.get("numero", ""), "ano": p.payload.get("ano"),
            "ementa": p.payload.get("ementa", ""), "url": p.payload.get("url", ""),
            "data": p.payload.get("data", ""),
        }
        for p in pts
    ]
    linhas.sort(key=lambda r: ((r.get("data") or ""), r.get("id") or 0), reverse=True)
    return linhas[: int(limite)]


def status() -> dict:
    try:
        c = conectar()
        total = int(c.count(_colecao(), exact=True).count)
        try:
            min_res = c.scroll(
                _colecao(), limit=1, with_payload=True,
                order_by=m.OrderBy(key="ano", direction=m.Direction.ASC),
            )[0]
            max_res = c.scroll(
                _colecao(), limit=1, with_payload=True,
                order_by=m.OrderBy(key="ano", direction=m.Direction.DESC),
            )[0]
            ano_min = min_res[0].payload.get("ano") if min_res else None
            ano_max = max_res[0].payload.get("ano") if max_res else None
        except Exception:
            ano_min = ano_max = None
        return {"total": total, "ano_minimo": ano_min, "ano_maximo": ano_max}
    except Exception:
        return {"total": 0, "ano_minimo": None, "ano_maximo": None}