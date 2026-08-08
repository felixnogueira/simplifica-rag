"""cliente da api pública de dados abertos da Câmara dos Deputados (âmbito nacional)."""

from datetime import date, timedelta

import httpx

from .config import config


MAX_PERIODO_DIAS = 93
MAX_AUTORES_RESOLVIDOS = 10
MAX_AUTORES_ENRIQUECIDOS = 6
_cache_deputados: dict[int, dict | None] = {}


class CamaraError(RuntimeError):
    pass


def _id_no_uri(uri: str | None) -> int | None:
    """extrai o id do deputado do último segmento de uma uri de autores."""
    if not uri:
        return None
    fim = uri.rstrip("/").rsplit("/", 1)[-1]
    try:
        return int(fim)
    except (TypeError, ValueError):
        return None


def _partido_deputado(id_deputado: int) -> tuple[str, str]:
    """partido e uf atuais de um deputado; (\"\", \"\") se indisponível."""
    if id_deputado not in _cache_deputados:
        try:
            _cache_deputados[id_deputado] = _get(f"/deputados/{id_deputado}")["dados"]
        except CamaraError:
            _cache_deputados[id_deputado] = None
    d = _cache_deputados[id_deputado]
    if not d:
        return "", ""
    st = d.get("ultimoStatus") or {}
    return (st.get("siglaPartido") or ""), (st.get("siglaUf") or "")


def _get(caminho: str, params: dict | None = None) -> dict:
    try:
        with httpx.Client(timeout=30) as c:
            r = c.get(f"{config.camara_base}{caminho}", params=params)
    except httpx.HTTPError as e:
        raise CamaraError(f"falha de rede na api da câmara: {e}")
    if r.status_code == 404:
        raise CamaraError("nada encontrado na consulta")
    if r.status_code != 200:
        raise CamaraError(f"api da câmara respondeu {r.status_code}")
    return r.json()


def _dados(caminho: str, params: dict | None = None) -> list:
    try:
        return _get(caminho, params)["dados"]
    except (KeyError, TypeError):
        raise CamaraError("resposta inesperada da api da câmara")


def url_ficha(id_proposicao: int) -> str:
    return f"https://www.camara.leg.br/proposicoesWeb/fichadetramitacao?idProposicao={id_proposicao}"


def _resumo_deputado(d: dict) -> dict:
    return {
        "id": d.get("id"),
        "nome": d.get("nome"),
        "partido": d.get("siglaPartido"),
        "uf": d.get("siglaUf"),
        "legislatura": d.get("idLegislatura"),
        "uri": d.get("uri"),
    }


def buscar_deputado(nome: str) -> dict:
    """resolve o deputado pelo nome; prefere igualdade exata."""
    nome = (nome or "").strip()
    if not nome:
        raise CamaraError("nome do deputado vazio")
    res = _dados("/deputados", {"nome": nome, "itens": 50})
    if not res:
        raise CamaraError(f"nenhum deputado encontrado para \"{nome}\"")
    vistos: dict[int, dict] = {}
    for d in res:
        vistos.setdefault(d.get("id"), d)
    alvo = nome.casefold()
    for d in vistos.values():
        if (d.get("nome") or "").casefold() == alvo:
            return _resumo_deputado(d)
    return _resumo_deputado(next(iter(vistos.values())))


def _resumo_prop(p: dict) -> dict:
    return {
        "id": p.get("id"),
        "siglaTipo": p.get("siglaTipo"),
        "numero": p.get("numero"),
        "ano": p.get("ano"),
        "ementa": p.get("ementa"),
        "dataApresentacao": p.get("dataApresentacao"),
        "url": url_ficha(p.get("id")),
    }


def proposicoes(
    id_autor: int | None = None,
    ano: int | None = None,
    sigla_tipo: str | None = None,
    numero: int | None = None,
    keywords: str | None = None,
    data_inicio: str | None = None,
    data_fim: str | None = None,
    sigla_partido_autor: str | None = None,
    sigla_uf_autor: str | None = None,
    itens: int = 100,
    pagina: int = 1,
) -> list[dict]:
    params = {
        "itens": max(1, min(int(itens), 100)),
        "pagina": int(pagina),
    }
    if id_autor:
        params["idDeputadoAutor"] = int(id_autor)
    if ano:
        params["ano"] = int(ano)
    if sigla_tipo:
        params["siglaTipo"] = sigla_tipo.strip().upper()
    if numero:
        params["numero"] = int(numero)
    if keywords:
        params["keywords"] = keywords.strip()
    if sigla_partido_autor:
        params["siglaPartidoAutor"] = sigla_partido_autor.strip().upper()
    if sigla_uf_autor:
        params["siglaUfAutor"] = sigla_uf_autor.strip().upper()
    data_inicio, data_fim = _limitar_periodo(data_inicio, data_fim)
    if data_inicio:
        params["dataApresentacaoInicio"] = data_inicio
    if data_fim:
        params["dataApresentacaoFim"] = data_fim
    return [_resumo_prop(p) for p in _dados("/proposicoes", params)]


def _resumo_autores(autores: list) -> list[dict]:
    """autores com partido/uf resolvido para os primeiros deputados."""
    saida = []
    resolvidos = 0
    for a in autores or []:
        nome = a.get("nome") or a.get("tipo")
        if not nome:
            continue
        item: dict = {"nome": nome, "ordem": a.get("ordemAssinatura")}
        id_dep = _id_no_uri(a.get("uri"))
        if id_dep and resolvidos < MAX_AUTORES_RESOLVIDOS:
            resolvidos += 1
            partido, uf = _partido_deputado(id_dep)
            if partido:
                item["partido"] = partido
                item["uf"] = uf
        saida.append(item)
    return saida


def autores(id_proposicao: int) -> list[dict]:
    """autores (com partido/uf resolvido) de uma proposição."""
    try:
        return _resumo_autores(_dados(f"/proposicoes/{id_proposicao}/autores"))
    except CamaraError:
        return []


def temas_proposicao(id_proposicao: int) -> list[str]:
    """temas oficiais de uma proposição (fallback agregado)."""
    try:
        return [t.get("tema") for t in _dados(f"/proposicoes/{id_proposicao}/temas") if t.get("tema")]
    except CamaraError:
        return []


def _enriquecer_com_autores(dados: list[dict]) -> list[dict]:
    """injeta os autores nos primeiros itens, para o llm nunca citar sem autoria."""
    for p in (dados or [])[:MAX_AUTORES_ENRIQUECIDOS]:
        pid = p.get("id")
        if pid and "autores" not in p:
            p["autores"] = autores(pid)
    return dados or []


def proposicao(id_proposicao: int) -> dict:
    """detalhe completo de uma proposição, com autores, temas e tramitação."""
    try:
        d = _get(f"/proposicoes/{id_proposicao}")["dados"]
    except (KeyError, TypeError):
        raise CamaraError("proposição não encontrada")
    if not d:
        raise CamaraError("proposição não encontrada")

    try:
        autores = _resumo_autores(_dados(f"/proposicoes/{id_proposicao}/autores"))
    except CamaraError:
        autores = []
    try:
        temas = [t.get("tema") for t in _dados(f"/proposicoes/{id_proposicao}/temas") if t.get("tema")]
    except CamaraError:
        temas = []

    st = d.get("statusProposicao") or {}
    d = dict(d)
    d["autores"] = autores
    d["temas"] = temas
    d["situacao"] = st.get("descricaoSituacao") or st.get("descricaoTramitacao")
    d["regime"] = st.get("regime")
    d["orgao"] = st.get("siglaOrgao")
    d["despacho"] = st.get("despacho")
    d["tipo"] = d.get("descricaoTipo") or d.get("siglaTipo")
    d["url"] = url_ficha(id_proposicao)
    return d


def _meses_atras(hoje: date, meses: int) -> str:
    year = hoje.year
    month = hoje.month - meses
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    return date(year, month, min(hoje.day, 28)).isoformat()


def _limitar_periodo(data_inicio: str | None, data_fim: str | None) -> tuple[str | None, str | None]:
    """mantém o período dentro da janela máxima (3 meses) que a api aceita."""
    try:
        fim = date.fromisoformat(data_fim) if data_fim else None
        ini = date.fromisoformat(data_inicio) if data_inicio else None
    except ValueError:
        return data_inicio, data_fim
    if ini and fim and (fim - ini).days > MAX_PERIODO_DIAS:
        ini = fim - timedelta(days=MAX_PERIODO_DIAS)
    return (ini.isoformat() if ini else data_inicio), (fim.isoformat() if fim else data_fim)


def url_evento(id_evento: int | None) -> str:
    if not id_evento:
        return "https://www.camara.leg.br/internet/votacao/"
    return f"https://www.camara.leg.br/evento-legislativo/{id_evento}"


def _resumo_votacao(v: dict) -> dict:
    return {
        "id": v.get("id"),
        "data": (v.get("data") or "")[:10],
        "siglaOrgao": v.get("siglaOrgao"),
        "idEvento": v.get("idEvento"),
        "aprovacao": v.get("aprovacao"),
        "descricao": v.get("descricao"),
        "tipo": "VOTAÇÃO",
        "titulo": v.get("descricao"),
        "url": url_evento(v.get("idEvento")),
    }


def votacoes(
    id_proposicao: int | None = None,
    id_orgao: int | None = None,
    id_evento: int | None = None,
    sigla_orgao: str | None = None,
    data_inicio: str | None = None,
    data_fim: str | None = None,
    itens: int = 50,
    pagina: int = 1,
) -> list[dict]:
    """lista de votações; período máximo de 3 meses."""
    hoje = date.today()
    data_inicio, data_fim = _limitar_periodo(data_inicio, data_fim)
    params = {
        "dataInicio": data_inicio or _meses_atras(hoje, 3),
        "dataFim": data_fim or hoje.isoformat(),
        "itens": max(1, min(int(itens), 100)),
        "pagina": int(pagina),
        "ordem": "DESC",
        "ordenarPor": "dataHoraRegistro",
    }
    if id_proposicao:
        params["idProposicao"] = int(id_proposicao)
    if id_orgao:
        params["idOrgao"] = int(id_orgao)
    if id_evento:
        params["idEvento"] = int(id_evento)
    res = [_resumo_votacao(v) for v in _dados("/votacoes", params)]
    if sigla_orgao:
        alvo = sigla_orgao.strip().casefold()
        res = [v for v in res if (v.get("siglaOrgao") or "").casefold() == alvo]
    return res


def _resumo_voto(v: dict) -> dict:
    dep = v.get("deputado_") or {}
    return {
        "voto": v.get("tipoVoto"),
        "nome": dep.get("nome"),
        "partido": dep.get("siglaPartido"),
        "uf": dep.get("siglaUf"),
    }


def _resumo_orientacao(o: dict) -> dict:
    return {
        "partido_bloco": o.get("siglaPartidoBloco"),
        "orientacao": o.get("orientacaoVoto"),
    }


def _resumo_afetada(p: dict) -> dict:
    return {
        "id": p.get("id"),
        "sigla": p.get("sigla"),
        "siglaTipo": p.get("siglaTipo"),
        "numero": p.get("numero"),
        "ano": p.get("ano"),
        "ementa": p.get("ementa"),
        "url": url_ficha(p.get("id")) if p.get("id") else (p.get("uri") or ""),
    }


def votacao(id_votacao: str, nome_deputado: str | None = None) -> dict:
    try:
        d = _get(f"/votacoes/{id_votacao}")["dados"]
    except (KeyError, TypeError):
        raise CamaraError("votação não encontrada")
    if not d:
        raise CamaraError("votação não encontrada")

    votos = [_resumo_voto(v) for v in _dados(f"/votacoes/{id_votacao}/votos")]
    afetadas = [_resumo_afetada(p) for p in d.get("proposicoesAfetadas") or []]
    afetadas = _enriquecer_com_autores(afetadas)
    base = {
        "id": d.get("id"),
        "data": (d.get("data") or "")[:10],
        "siglaOrgao": d.get("siglaOrgao"),
        "idEvento": d.get("idEvento"),
        "descricao": d.get("descricao"),
        "aprovacao": d.get("aprovacao"),
        "proposicoesAfetadas": afetadas,
        "tipo": "VOTAÇÃO",
        "titulo": d.get("descricao"),
        "url": url_evento(d.get("idEvento")),
    }

    if nome_deputado:
        alvo = (nome_deputado or "").casefold()
        achou = [v for v in votos if alvo in (v.get("nome") or "").casefold()]
        return {**base, "votos_por_nome": achou}

    contagem: dict[str, int] = {}
    por_partido: dict[str, dict[str, int]] = {}
    for v in votos:
        tv = v.get("voto") or "?"
        partido = v.get("partido") or "?"
        contagem[tv] = contagem.get(tv, 0) + 1
        por_partido.setdefault(partido, {})
        por_partido[partido][tv] = por_partido[partido].get(tv, 0) + 1

    try:
        orientacoes = [_resumo_orientacao(o) for o in _dados(f"/votacoes/{id_votacao}/orientacoes")]
    except CamaraError:
        orientacoes = []

    return {
        **base,
        "total_votos_registrados": len(votos),
        "contagem_por_voto": contagem,
        "votos_por_partido": por_partido,
        "orientacoes": orientacoes,
    }


def siglas_tipos() -> list[dict]:
    """tipos de proposição (PL, PEC, MPV...) para filtros."""
    return [
        {"sigla": x.get("sigla"), "nome": x.get("nome")}
        for x in _dados("/referencias/proposicoes/siglaTipo")
        if x.get("sigla")
    ]


def situacoes_proposicao() -> list[dict]:
    """situações possíveis de tramitação para filtros."""
    return [
        {"cod": x.get("cod"), "nome": x.get("nome")}
        for x in _dados("/referencias/proposicoes/codSituacao")
        if x.get("nome")
    ]


# ---- deputados (listagem com filtros, despesas, discursos, órgãos) ----

def deputados(
    nome: str | None = None,
    sigla_partido: str | None = None,
    sigla_uf: str | None = None,
    sexo: str | None = None,
    ids: list[int] | None = None,
    id_legislatura: int | None = None,
    data_inicio: str | None = None,
    data_fim: str | None = None,
    itens: int = 100,
) -> list[dict]:
    params = {"itens": max(1, min(int(itens), 100)), "ordem": "ASC", "ordenarPor": "nome"}
    if nome:
        params["nome"] = nome.strip()
    if sigla_partido:
        params["siglaPartido"] = sigla_partido.strip().upper()
    if sigla_uf:
        params["siglaUf"] = sigla_uf.strip().upper()
    if sexo:
        params["siglaSexo"] = sexo.strip().upper()
    if ids:
        params["id"] = ",".join(str(i) for i in ids[:100])
    if id_legislatura:
        params["idLegislatura"] = int(id_legislatura)
    if data_inicio:
        params["dataInicio"] = data_inicio
    if data_fim:
        params["dataFim"] = data_fim
    return [_resumo_deputado(d) for d in _dados("/deputados", params)]


def _resumo_despesa(d: dict) -> dict:
    return {
        "data": (d.get("dataDocumento") or "")[:10],
        "tipo": d.get("tipoDespesa"),
        "valor": d.get("valorLiquido") or d.get("valorDocumento"),
        "fornecedor": d.get("nomeFornecedor"),
        "url": d.get("urlDocumento") or "",
    }


def despesas_deputado(
    id_deputado: int,
    ano: int | None = None,
    mes: int | None = None,
    cnpj_fornecedor: str | None = None,
    itens: int = 50,
) -> list[dict]:
    params = {"itens": max(1, min(int(itens), 100)), "ordem": "DESC", "ordenarPor": "dataDocumento"}
    if ano:
        params["ano"] = int(ano)
    if mes:
        params["mes"] = int(mes)
    if cnpj_fornecedor:
        params["cnpjCpfFornecedor"] = cnpj_fornecedor.strip()
    return [_resumo_despesa(d) for d in _dados(f"/deputados/{id_deputado}/despesas", params)]


def discursos_deputado(
    id_deputado: int,
    data_inicio: str | None = None,
    data_fim: str | None = None,
    id_legislatura: int | None = None,
    itens: int = 30,
) -> list[dict]:
    params = {"itens": max(1, min(int(itens), 100))}
    if data_inicio:
        params["dataInicio"] = data_inicio
    if data_fim:
        params["dataFim"] = data_fim
    if id_legislatura:
        params["idLegislatura"] = int(id_legislatura)
    return [
        {
            "data": (d.get("dataHoraInicio") or "")[:10],
            "evento": d.get("descricaoTipo") or d.get("tipoEvento"),
            "resumo": (d.get("resumo") or d.get("sumarioEvento") or "")[:300],
            "url": d.get("urlTexto") or "",
        }
        for d in _dados(f"/deputados/{id_deputado}/discursos", params)
    ]


def orgaos_deputado(id_deputado: int) -> list[dict]:
    try:
        dados = _dados(f"/deputados/{id_deputado}/orgaos", {"itens": 100})
    except CamaraError:
        return []
    return [
        {
            "sigla": o.get("siglaOrgao") or o.get("sigla"),
            "nome": o.get("nomeOrgao") or o.get("nome"),
            "cargo": o.get("cargo") or "",
        }
        for o in dados
        if o.get("siglaOrgao") or o.get("sigla")
    ]


# ---- partidos, frentes, órgãos, eventos ----

def partidos(
    sigla: str | None = None,
    data_inicio: str | None = None,
    data_fim: str | None = None,
    itens: int = 100,
) -> list[dict]:
    params = {"itens": max(1, min(int(itens), 100))}
    if sigla:
        params["sigla"] = sigla.strip().upper()
    if data_inicio:
        params["dataInicio"] = data_inicio
    if data_fim:
        params["dataFim"] = data_fim
    return [
        {"id": p.get("id"), "sigla": p.get("sigla"), "nome": p.get("nome")}
        for p in _dados("/partidos", params)
        if p.get("sigla")
    ]


def membros_partido(
    id_partido: int,
    id_legislatura: int | None = None,
    data_inicio: str | None = None,
    data_fim: str | None = None,
    itens: int = 100,
) -> list[dict]:
    params: dict = {"itens": max(1, min(int(itens), 100))}
    if id_legislatura:
        params["idLegislatura"] = int(id_legislatura)
    if data_inicio:
        params["dataInicio"] = data_inicio
    if data_fim:
        params["dataFim"] = data_fim
    return [_resumo_deputado(d) for d in _dados(f"/partidos/{id_partido}/membros", params)]


def lideres_partido(id_partido: int) -> list[dict]:
    try:
        dados = _dados(f"/partidos/{id_partido}/lideres", {"itens": 100})
    except CamaraError:
        return []
    saida = []
    for l in dados:
        dep = l.get("deputado_") or l.get("deputado") or {}
        nome = l.get("nome") or dep.get("nome")
        cargo = l.get("tipoLideranca") or l.get("descricaoLideranca") or ""
        if nome:
            saida.append({"nome": nome, "cargo": cargo, "partido": dep.get("siglaPartido")})
    return saida


def frentes(id_legislatura: int | None = None, itens: int = 100) -> list[dict]:
    params = {"itens": max(1, min(int(itens), 100))}
    if id_legislatura:
        params["idLegislatura"] = int(id_legislatura)
    return [
        {
            "id": f.get("id"),
            "titulo": f.get("titulo"),
            "legislatura": f.get("idLegislatura"),
            "url": f"https://www.camara.leg.br/frente-parlamentar/{f.get('id')}" if f.get("id") else "",
        }
        for f in _dados("/frentes", params)
        if f.get("titulo")
    ]


def membros_frente(id_frente: int, id_legislatura: int | None = None, itens: int = 100) -> list[dict]:
    params: dict = {"itens": max(1, min(int(itens), 100))}
    if id_legislatura:
        params["idLegislatura"] = int(id_legislatura)
    try:
        dados = _dados(f"/frentes/{id_frente}/membros", params)
    except CamaraError:
        return []
    saida = []
    for m in dados:
        if "deputado_" in m:
            saida.append(_resumo_deputado(m["deputado_"]))
        elif m.get("nome"):
            saida.append(_resumo_deputado(m))
    return saida


def orgaos(nome: str | None = None, sigla: str | None = None, itens: int = 50) -> list[dict]:
    params = {"itens": max(1, min(int(itens), 100))}
    if nome:
        params["nome"] = nome.strip()
    if sigla:
        params["sigla"] = sigla.strip().upper()
    return [
        {"id": o.get("id"), "sigla": o.get("sigla"), "nome": o.get("nome"), "tipo": o.get("tipoOrgao")}
        for o in _dados("/orgaos", params)
        if o.get("id")
    ]


def eventos(
    data_inicio: str | None = None,
    data_fim: str | None = None,
    id_orgao: int | None = None,
    id_tipo: int | None = None,
    itens: int = 30,
) -> list[dict]:
    hoje = date.today()
    params = {
        "dataInicio": data_inicio or _meses_atras(hoje, 1),
        "dataFim": data_fim or hoje.isoformat(),
        "itens": max(1, min(int(itens), 100)),
    }
    if id_orgao:
        params["idOrgao"] = id_orgao
    if id_tipo:
        params["idTipoEvento"] = id_tipo
    dados = _dados("/eventos", params)
    return [
        {
            "id": e.get("id"),
            "data": (e.get("dataHoraInicio") or "")[:10],
            "tipo": e.get("descricaoTipo"),
            "situacao": e.get("situacao"),
            "titulo": e.get("descricao"),
            "url": url_evento(e.get("id")),
        }
        for e in dados
        if e.get("id")
    ]


def pauta_evento(id_evento: int) -> list[dict]:
    try:
        dados = _dados(f"/eventos/{id_evento}/pauta", {"itens": 100})
    except CamaraError:
        return []
    saida = []
    for p in dados:
        pid = p.get("id") or p.get("idProposicao")
        if not pid:
            continue
        sigla = p.get("sigla") or p.get("siglaTipo")
        saida.append(
            {
                "id": pid,
                "sigla": sigla,
                "siglaTipo": sigla,
                "numero": p.get("numero"),
                "ano": p.get("ano"),
                "ementa": p.get("ementa"),
                "url": url_ficha(pid),
            }
        )
    return _enriquecer_com_autores(saida)