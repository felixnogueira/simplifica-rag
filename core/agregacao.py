"""agregações determinísticas: perguntas de contagem/ranking/temas respondidas por
SQL sobre o índice (com fallback na API estruturada) — sem depender da LLM."""

import re
import unicodedata

from . import camara, rag
from .config import config

_STOP_NOME = {
    "em", "e", "votou", "apresentou", "teve", "tem", "foi", "apresenta",
    "apresentaram", "por", "com", "sobre", "quantos", "qual", "quais",
    "apresentado", "na", "no", "nas", "nos", "votar", "virou", "sancionada",
    "do", "da", "de", "dos", "das", "autorou", "autora", "autoraram",
    "apresentado", "esse", "este", "ano", "ano", "projetos", "projeto",
}

_PARTIDOS = {
    "PL", "PT", "PSDB", "MDB", "PSOL", "PSD", "PP", "REPUBLICANOS", "PDT",
    "PODEMOS", "PSB", "PCdoB", "NOVO", "SOLIDARIEDADE", "UNIÃO", "UNIAO",
    "AVANTE", "CIDADANIA", "DC", "DEM", "PATRIOTA", "PROS", "PSL", "PSC",
    "PR", "PRB", "PRTB", "PHS", "PPS", "PMB", "PMN", "PTB", "PV", "REDE",
    "PSTU", "PCB", "PCO", "UP",
}

_LEGISLATURAS = (
    (2023, 2026, 57),
    (2019, 2022, 56),
    (2015, 2018, 55),
    (2011, 2014, 54),
    (2007, 2010, 53),
)


def _norm(texto: str) -> str:
    s = unicodedata.normalize("NFD", texto or "")
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def _ano_simples(anos: list | None) -> int | None:
    return anos[0] if anos and len(anos) == 1 else None


def _legislatura(ano: int) -> int | None:
    for ini, fim, leg in _LEGISLATURAS:
        if ini <= ano <= fim:
            return leg
    return None


def _nome_deputado(pergunta: str) -> str:
    m = re.search(
        r"(?:deputad[ao]s?|parlamentar)\s+([A-Za-zÀ-ú]+(?:\s+[A-Za-zÀ-ú]+){0,5})",
        pergunta,
    )
    if not m:
        return ""
    partes = []
    for tok in m.group(1).split():
        if tok.casefold() in _STOP_NOME or tok.upper() in _PARTIDOS \
                or re.fullmatch(r"(?:19|20)\d{2}", tok):
            break
        partes.append(tok)
    return " ".join(partes)


def detectar(pergunta: str, filtros: dict) -> dict | None:
    """reconhece intenções de agregação; devolve None se a pergunta for de busca."""
    p_texto = _norm(pergunta)
    anos = filtros.get("anos") or []
    ano = _ano_simples(anos)
    tem_deputado = "deputad" in p_texto or "parlamentar" in p_texto
    tem_projeto = bool(re.search(r"\bprojetos?\b", p_texto))
    tem_tema = bool(re.search(r"\btemas?\b|\bassuntos?\b", p_texto))
    rank = bool(re.search(r"maior(?:es)? numero de projetos|mais projetos|mais apresent|mais proposi|que mais|ranking|quem mais", p_texto))
    quantos = bool(re.search(r"\bquantos projetos\b", p_texto))
    tem_partido = bool(re.search(r"\bpartidos?\b", p_texto) or filtros.get("partido"))

    base = {"uf": filtros.get("uf"), "partido": filtros.get("partido"),
            "ano": ano, "anos": anos, "temas": tem_tema}
    nome = _nome_deputado(pergunta)
    if tem_deputado and quantos and nome:
        return {**base, "tipo": "quantos_projetos", "nome": nome}
    if quantos and tem_partido and not nome:
        return {**base, "tipo": "quantos_partido"}
    if tem_deputado and rank:
        return {**base, "tipo": "rank_deputados"}
    if rank and tem_partido:
        return {**base, "tipo": "rank_partidos"}
    if tem_tema and tem_projeto and not tem_deputado:
        return {**base, "tipo": "temas_top"}
    return None


# ---- execução no índice ---------------------------------------------------

def _rank_deputados(inten):
    anos = inten.get("anos") or []
    uf, partido = inten.get("uf"), inten.get("partido")
    if not anos:
        return rag.rank_deputados(uf, partido, None, config.rag_top_k + 2)
    soma: dict = {}
    for a in anos:
        for r in rag.rank_deputados(uf, partido, a, 50):
            soma.setdefault(r["nome"], {"nome": r["nome"], "partido": r["partido"],
                                        "uf": r["uf"], "total": 0})
            soma[r["nome"]]["total"] += r["total"]
            if not soma[r["nome"]]["partido"] and r["partido"]:
                soma[r["nome"]]["partido"] = r["partido"]
    return sorted(soma.values(), key=lambda r: (-r["total"], r["nome"]))[: config.rag_top_k]


def executar(inten: dict) -> dict | None:
    """roda a agregação no índice; se o índice estiver sem os dados, cai na API."""
    tipo = inten.get("tipo")
    if tipo == "rank_deputados":
        return _executar_rank_deputados(inten)
    if tipo == "rank_partidos":
        return _executar_rank_partidos(inten)
    if tipo == "quantos_projetos":
        return _executar_quantos(inten)
    if tipo == "quantos_partido":
        return _executar_quantos_partido(inten)
    if tipo == "temas_top":
        return _executar_temas_top(inten)
    return None


def _formatar_ranking(inten, linhas: list[dict], cabecalho: str, via_api: bool = False) -> dict:
    if not linhas:
        return {"resposta": _sem_dados(inten), "fontes": []}
    anos = inten.get("anos") or []
    ano = _ano_simples(anos)
    sufixo = f" em {ano}" if ano else ""
    partes = [f"{cabecalho}{sufixo}:", ""]
    for i, r in enumerate(linhas[:10], 1):
        linha = f"{i}. {r['nome']} ({r.get('partido') or ''}/{r.get('uf') or ''}) — {r['total']} projeto{'s' if r['total'] != 1 else ''}"
        if via_api and r["total"] >= 100:
            linha += "*"
        if inten.get("temas"):
            temas = (
                _temas_deputado_api(r["nome"], ano)
                if via_api
                else rag.temas_de_deputado(r["nome"], ano, 3)
            )
            if temas:
                linha += " | Temas: " + ", ".join(f"{t['tema']} ({t['total']})" for t in temas)
        partes.append(linha)
    partes.append("")
    partes.append("Base: proposições oficiais da Câmara dos Deputados (Dados Abertos).")
    if via_api and any(r["total"] >= 100 for r in linhas):
        partes.append("* contagem limitada pela consulta pública (máx. 100 proposições).")
    fontes = []
    if linhas:
        fontes = _fontes_deputado(linhas[0]["nome"], ano) if not via_api else []
    return {"resposta": "\n".join(partes), "fontes": fontes}


def _fontes_deputado(nome: str, ano: int | None) -> list[dict]:
    out = []
    for p in rag.proposicoes_de_deputado(nome, ano, 5):
        out.append({
            "titulo": f"{p.get('sigla_tipo') or ''} {p.get('numero') or ''}/{p.get('ano') or ''}".strip() or "Proposição",
            "url": p["url"], "tipo": p.get("sigla_tipo") or "",
            "data": p.get("data") or "", "ano": p.get("ano"), "score": None,
        })
    return out


def _sem_dados(inten: dict) -> str:
    return (
        "Não encontrei dados suficientes para calcular esse ranking nos anos "
        + ("e no estado informados." if inten.get("uf") else "informados.")
        + " Verifique se os dados desse período já foram indexados."
    )


def _executar_rank_deputados(inten: dict) -> dict | None:
    linhas = _rank_deputados(inten)
    via_api = False
    if not linhas:
        linhas = _rank_api(inten)
        via_api = bool(linhas)
    uf = inten.get("uf") or ""
    cab = f"Deputados com mais projetos na Câmara{', em ' + uf if uf else ''}"
    return _formatar_ranking(inten, linhas, cab, via_api)


def _executar_rank_partidos(inten: dict) -> dict | None:
    anos = inten.get("anos") or []
    linhas = rag.rank_partidos(inten.get("uf"), _ano_simples(anos), config.rag_top_k)
    if not linhas and _ano_simples(anos):
        linhas = _rank_partidos_api(inten)
    if not linhas:
        return {"resposta": _sem_dados(inten), "fontes": []}
    partes = [f"Partidos com mais projetos na Câmara" + (f" em {anos[0]}" if len(anos) == 1 else "") + ":", ""]
    for i, r in enumerate(linhas[:10], 1):
        partes.append(f"{i}. {r['partido']} — {r['total']} projeto{'s' if r['total'] != 1 else ''}")
    partes.append("")
    partes.append("Base: proposições oficiais da Câmara dos Deputados (Dados Abertos).")
    return {"resposta": "\n".join(partes), "fontes": []}


def _executar_quantos(inten: dict) -> dict | None:
    nome = inten.get("nome") or ""
    if not nome:
        return None
    reg = rag.dados_deputado(nome)
    ano = inten.get("ano")
    if reg is None and nome:
        try:
            leg = _legislatura(ano) if ano else None
            lista = (
                camara.deputados(nome=nome, id_legislatura=leg, itens=50)
                if leg
                else camara.deputados(nome=nome, itens=50)
            )
            dep = lista[0] if lista else None
        except camara.CamaraError:
            dep = None
        if dep:
            total = len(camara.proposicoes(ano=ano, id_autor=dep["id"], itens=100)) if ano else 0
            reg = {"nome": dep.get("nome"), "partido": dep.get("partido") or "",
                   "uf": dep.get("uf") or "", "total": total}
    if not reg:
        return None
    linha = f"O deputado {reg['nome']} ({reg.get('partido') or ''}/{reg.get('uf') or ''}) apresentou {reg['total']} projeto{'s' if reg['total'] != 1 else ''}"
    if ano:
        linha += f" em {ano}"
    linha += " na Câmara dos Deputados."
    fontes = _fontes_deputado(reg["nome"], ano)
    return {"resposta": linha, "fontes": fontes}


def _executar_quantos_partido(inten: dict) -> dict | None:
    partido = inten.get("partido")
    if not partido:
        return None
    uf = inten.get("uf")
    anos = inten.get("anos") or []
    total = 0
    if config.tem_rag:
        for a in anos or [None]:
            n = rag.count_partido(partido, uf, a)
            total += n if n is not None else 0
    via_api = False
    if total == 0:
        total, via_api = _count_partido_api(partido, uf, _ano_simples(anos), inten.get("ano"))
    if not total:
        return None
    partes = [f"Deputados do {partido}{' em ' + uf if uf else ''} autoraram {total} "
              f"projeto{'s' if total != 1 else ''} na Câmara"]
    if _ano_simples(anos):
        partes[0] += f" em {_ano_simples(anos)}"
    partes[0] += "."
    if via_api:
        partes.append("* contagem limitada pela consulta pública (máx. 60 deputados, 100 proposições cada).")
    partes.append("Base: proposições oficiais da Câmara dos Deputados (Dados Abertos).")
    return {"resposta": "\n".join(partes), "fontes": []}


def _count_partido_api(partido: str, uf: str | None, ano: int | None, ano_leg: int | None) -> tuple[int, bool]:
    """fallback: soma proposições dos deputados do partido (limitado, sinalizado)."""
    try:
        leg = _legislatura(ano_leg or ano) if (ano_leg or ano) else None
        deps = camara.deputados(
            sigla_partido=partido, sigla_uf=uf,
            id_legislatura=leg if leg else None, itens=100,
        )
    except camara.CamaraError:
        return 0, False
    total = 0
    for d in deps[:60]:
        try:
            n = len(camara.proposicoes(ano=ano, id_autor=d.get("id"), itens=100))
        except camara.CamaraError:
            n = 0
        total += n
    return total, True


def _executar_temas_top(inten: dict) -> dict | None:
    anos = inten.get("anos") or []
    linhas = rag.temas_top(inten.get("uf"), inten.get("partido"), _ano_simples(anos), 10)
    if not linhas and _ano_simples(anos):
        linhas = _temas_top_api(inten)
    if not linhas:
        return {"resposta": "Não encontrei dados de temas nos anos informados.", "fontes": []}
    partes = [f"Principais temas das proposições na Câmara" + (f" em {anos[0]}" if len(anos) == 1 else "") + ":", ""]
    for i, r in enumerate(linhas[:10], 1):
        s = "proposição" if r["total"] == 1 else "proposições"
        partes.append(f"{i}. {r['tema']} — {r['total']} {s}")
    partes.append("")
    partes.append("Base: classificação temática oficial da Câmara dos Deputados (Dados Abertos).")
    return {"resposta": "\n".join(partes), "fontes": []}


# ---- fallback via API estruturada (lento, usado quando o índice não tem o ano) -----

def _api_deputados(inten: dict) -> list[dict] | None:
    """fallback só para o ano corrente (legislatura 57): a combinação siglaUf+idLegislatura
    dá timeout na API, e os deputados atuais são exatamente os de 2026."""
    ano = inten.get("ano")
    if not ano or _legislatura(ano) != 57:
        return None
    try:
        deputados = camara.deputados(sigla_uf=inten.get("uf"), itens=100)
    except camara.CamaraError:
        return None
    if inten.get("partido"):
        deputados = [d for d in deputados if (d.get("partido") or "").upper() == inten["partido"].upper()]
    return deputados


def _rank_api(inten: dict) -> list[dict]:
    deputados = _api_deputados(inten)
    if not deputados:
        return []
    linhas = []
    for d in deputados[:45]:
        try:
            n = len(camara.proposicoes(ano=inten.get("ano"), id_autor=d["id"], itens=100))
        except camara.CamaraError:
            n = 0
        if n:
            linhas.append({"nome": d.get("nome"), "partido": d.get("partido") or "",
                           "uf": d.get("uf") or "", "total": n})
    linhas.sort(key=lambda r: (-r["total"], r["nome"]))
    return linhas[: config.rag_top_k]


def _rank_partidos_api(inten: dict) -> list[dict]:
    deputados = _api_deputados(inten)
    if not deputados:
        return []
    soma: dict = {}
    for d in deputados[:45]:
        try:
            n = len(camara.proposicoes(ano=inten.get("ano"), id_autor=d["id"], itens=100))
        except camara.CamaraError:
            n = 0
        sigla = (d.get("partido") or "").upper()
        if n and sigla:
            soma[sigla] = soma.get(sigla, 0) + n
    out = [{"partido": s, "total": n} for s, n in sorted(soma.items(), key=lambda kv: (-kv[1], kv[0]))]
    return out[: config.rag_top_k]


def _temas_deputado_api(nome: str, ano: int | None, limite_prop: int = 20, limite: int = 3) -> list[dict]:
    """temas mais frequentes de um deputado via API (fallback), limitado."""
    if not nome:
        return []
    try:
        deps = camara.deputados(nome=nome, itens=50)
    except camara.CamaraError:
        return []
    if not deps:
        return []
    dep = deps[0]
    try:
        resumos = camara.proposicoes(ano=ano, id_autor=dep["id"], itens=100)
    except camara.CamaraError:
        return []
    soma: dict = {}
    for r in resumos[:limite_prop]:
        for t in camara.temas_proposicao(r.get("id")):
            if t:
                soma[t] = soma.get(t, 0) + 1
    return [
        {"tema": t, "total": n}
        for t, n in sorted(soma.items(), key=lambda kv: (-kv[1], kv[0]))[:limite]
    ]


def _temas_top_api(inten: dict) -> list[dict]:
    deputados = _api_deputados(inten)
    if not deputados:
        return []
    soma: dict = {}
    vistos = set()
    for d in deputados[:20]:
        try:
            resumos = camara.proposicoes(ano=inten.get("ano"), id_autor=d["id"], itens=100)
        except camara.CamaraError:
            continue
        for r in resumos[:15]:
            pid = r.get("id")
            if not pid or pid in vistos:
                continue
            vistos.add(pid)
            for t in camara.temas_proposicao(pid):
                if t:
                    soma[t] = soma.get(t, 0) + 1
    out = [{"tema": t, "total": n} for t, n in sorted(soma.items(), key=lambda kv: (-kv[1], kv[0]))]
    return out[:10]