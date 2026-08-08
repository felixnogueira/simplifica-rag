"""normalização de proposições em documentos pesquisáveis + formatação determinística."""

from . import camara


def _sigla_prop(p: dict) -> str:
    return p.get("siglaTipo") or p.get("sigla") or p.get("tipo") or ""


def _numero_prop(p: dict) -> str:
    n = p.get("numero")
    return f"{n}" if n or n == 0 else ""


def _ano_prop(p: dict) -> int:
    """ano confiável: usa dataApresentacao quando o campo ano vem zerado/ausente."""
    ano = p.get("ano")
    try:
        if ano:
            return int(ano)
    except (TypeError, ValueError):
        pass
    data = (p.get("dataApresentacao") or "")[:10]
    if len(data) == 10 and data[4] == "-":
        try:
            return int(data[:4])
        except ValueError:
            pass
    return 0


def titulo(p: dict) -> str:
    """'PL 42/2026'."""
    partes = [x for x in (_sigla_prop(p), _numero_prop(p), str(_ano_prop(p) or "")) if x]
    if partes:
        return f"{partes[0]} {partes[1]}/{partes[2]}".rstrip("/ ") if len(partes) >= 3 else " ".join(partes)
    return "Proposição"


def url_ficha(p: dict) -> str:
    pid = p.get("id")
    return camara.url_ficha(pid) if pid else p.get("url") or ""


def _linha_autor(a: dict) -> str:
    nome = a.get("nomeAutor") or a.get("nome") or (a.get("tipo") or "")
    partido = a.get("siglaPartidoAutor") or a.get("partido") or ""
    uf = a.get("siglaUFAutor") or a.get("uf") or ""
    sufixo = f" ({partido}/{uf})" if partido and uf else f" ({uf})" if uf else f" ({partido})" if partido else ""
    return f"{nome}{sufixo}".strip()


def _siglas_autores(autores: list | None) -> str:
    """fragmentos '(SIGLA/UF)' para filtros determinísticos por partido/uf."""
    frags = []
    for a in autores or []:
        partido = a.get("siglaPartidoAutor") or a.get("partido") or ""
        uf = a.get("siglaUFAutor") or a.get("uf") or ""
        if partido and uf:
            frags.append(f"({partido}/{uf})")
        elif partido:
            frags.append(f"({partido}/)")
        elif uf:
            frags.append(f"(/{uf})")
    return " ".join(frags)


def situacao(p: dict) -> str:
    st = p.get("ultimoStatus") or {}
    return p.get("situacao") or st.get("descricaoSituacao") or st.get("descricaoTramitacao") or ""


def _autor_detalhe(a: dict) -> dict:
    """autores normalizados para as tabelas de agregação (nome, partido, uf, ordem)."""
    return {
        "nome": (a.get("nomeAutor") or a.get("nome") or (a.get("tipo") or "") or "").strip(),
        "partido": a.get("siglaPartidoAutor") or a.get("partido") or "",
        "uf": a.get("siglaUFAutor") or a.get("uf") or "",
        "ordem": a.get("ordemAssinatura") or "",
    }


def montar_doc(p: dict, autores: list | None = None, temas: list | None = None) -> dict:
    """constrói o documento textual de uma proposição (uma linha = uma proposição)."""
    temas = temas or p.get("temas") or []
    temas_lista = [t for t in temas if t]
    texto_temas = ", ".join(temas_lista)
    autores = autores if autores is not None else p.get("autores") or []
    autores_detalhe = [_autor_detalhe(a) for a in autores if _autor_detalhe(a).get("nome")]
    texto_autores = "; ".join(_linha_autor(a) for a in autores) if autores else ""

    partes = [
        f"Título: {titulo(p)}",
        f"Tipo: {p.get('descricaoTipo') or _sigla_prop(p)}",
        f"Ementa: {p.get('ementa') or ''}".rstrip(),
    ]
    if p.get("ementaDetalhada"):
        partes.append(f"Ementa detalhada: {p['ementaDetalhada']}".rstrip())
    if p.get("keywords"):
        partes.append(f"Palavras-chave: {p['keywords']}")
    if texto_temas:
        partes.append(f"Temas: {texto_temas}")
    if texto_autores:
        partes.append(f"Autores: {texto_autores}")
    st = situacao(p)
    if st:
        partes.append(f"Situação: {st}")
    partes.append(f"URL: {url_ficha(p)}")

    return {
        "id": p.get("id"),
        "sigla_tipo": _sigla_prop(p),
        "numero": p.get("numero"),
        "ano": _ano_prop(p),
        "ementa": p.get("ementa") or "",
        "temas": texto_temas,
        "temas_lista": temas_lista,
        "autores": texto_autores,
        "autores_detalhe": autores_detalhe,
        "siglas_autores": _siglas_autores(autores),
        "situacao": st,
        "url": url_ficha(p),
        "data": (p.get("dataApresentacao") or "")[:10],
        "doc": "\n".join(partes),
    }


def para_fonte(doc: dict, score: float | None = None) -> dict:
    return {
        "titulo": titulo(doc),
        "url": doc.get("url", ""),
        "tipo": doc.get("sigla_tipo", ""),
        "data": doc.get("data", ""),
        "ano": doc.get("ano"),
        "score": score,
    }


def formato_simples(docs: list[dict]) -> str:
    """resposta determinística (sem llm): lista clara dos projetos recuperados."""
    if not docs:
        return (
            "Não encontrei proposições sobre esse assunto nos dados disponíveis. "
            "Tente perguntar usando outros termos, por um ano específico ou por um tema "
            "como saúde, educação, impostos ou direitos dos trabalhadores."
        )
    linhas = [
        "Encontrei estas proposições relacionadas (com base nos dados da Câmara dos Deputados):",
        "",
    ]
    visto = set()
    for d in docs:
        sigla = f"{d.get('sigla_tipo', '')} {d.get('numero', '')}/{d.get('ano', '')}".strip()
        chave = (d.get("sigla_tipo"), d.get("numero"), d.get("ano"))
        if chave in visto:
            continue
        visto.add(chave)
        ementa = d.get("ementa") or ""
        if len(ementa) > 200:
            ementa = ementa[:200].rstrip() + "…"
        linha = f"• {sigla} — {ementa}"
        if d.get("autores"):
            linha += f" (Autor: {d['autores']})"
        if d.get("situacao"):
            linha += f" | Situação: {d['situacao']}"
        linhas.append(linha)
    linhas.append("")
    linhas.append("Esses dados vêm de fontes oficiais da Câmara dos Deputados.")
    return "\n".join(linhas)