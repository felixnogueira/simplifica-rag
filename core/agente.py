"""agente: responde perguntas com camada determinística.

Ordem de resolução (nunca inventa dados; a LLM só redige):
1. intenções estruturadas via API da Câmara: votação de proposição, proposição específica;
2. agregações determinísticas (contagem/ranking/temas) sobre o índice ou a API;
3. busca no índice vetorial/textual (RAG) com filtros (partido/uf/ano/tipo);
4. busca estruturada na API com filtros (partido/uf/autor/keywords) — todos os anos;
5. resposta determinística de "não encontrei" (sem alucinação)."""

import json
import re
from datetime import date

from . import agregacao, camara, rag
from .config import config
from .documento import formato_simples, montar_doc, para_fonte
from .llm import LlmError, lm

_UFS = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
}

_UF_PALAVRAS = {
    "SÃO PAULO": "SP", "SAO PAULO": "SP", "PAULISTA": "SP", "PAULISTAS": "SP",
    "RIO DE JANEIRO": "RJ", "CARIOCA": "RJ", "CARIOCAS": "RJ", "FLUMINENSE": "RJ",
    "MINAS GERAIS": "MG",
    "SANTA CATARINA": "SC", "CATARINENSE": "SC",
    "RIO GRANDE DO SUL": "RS", "GAÚCHO": "RS", "GAUCHO": "RS", "GAÚCHA": "RS",
    "RIO GRANDE DO NORTE": "RN", "POTIGUAR": "RN",
    "PARANÁ": "PR", "PARANA": "PR", "PARANAENSE": "PR", "PARANAENSES": "PR",
    "BAHIA": "BA", "BAIANO": "BA", "BAIANA": "BA",
    "DISTRITO FEDERAL": "DF", "BRASILIENSE": "DF",
    "CEARÁ": "CE", "CEARA": "CE", "CEARENSE": "CE", "CEARAENSE": "CE",
    "PERNAMBUCO": "PE", "PERNAMBUCANO": "PE", "PERNAMBUCANA": "PE",
    "GOIÁS": "GO", "GOIAS": "GO", "GOIANO": "GO", "GOIANA": "GO",
    "AMAZONAS": "AM", "AMAZONENSE": "AM",
    "PARÁ": "PA", "PARAENSE": "PA",
    "MARANHÃO": "MA", "MARANHAO": "MA", "MARANHENSE": "MA",
    "ESPÍRITO SANTO": "ES", "ESPIRITO SANTO": "ES", "CAPIXABA": "ES",
    "ALAGOAS": "AL", "ALAGOANO": "AL",
    "SERGIPE": "SE", "SERGIPANO": "SE",
    "MATO GROSSO DO SUL": "MS", "MATO GROSSO": "MT",
}

_PARTIDOS = {
    "PL", "PT", "PSDB", "MDB", "PSOL", "PSD", "PP", "REPUBLICANOS", "PDT",
    "PODEMOS", "PSB", "PCdoB", "NOVO", "SOLIDARIEDADE", "UNIÃO", "UNIAO",
    "AVANTE", "CIDADANIA", "DC", "DEM", "PATRIOTA", "PROS", "PSL", "PSC",
    "PR", "PRB", "PRTB", "PHS", "PPS", "PMB", "PMN", "PTB", "PV", "REDE",
    "PSTU", "PCB", "PCO", "UP",
}

_SIGLAS_TIPO = [
    "PEC", "MPV", "PDL", "PLP", "PLS", "PL", "REQ", "PRC", "EMC", "RCP",
    "REC", "INC", "SUB", "OF", "RIC", "MSC", "PDC",
]

_SISTEMA = """Você é o Simplifica Legislativo, um assistente que explica dados oficiais da Câmara dos Deputados em linguagem simples, em português brasileiro, para pessoas leigas.

REGRAS OBRIGATÓRIAS (nunca as quebre):

1. Responda APENAS com base nos DADOS OFICIAIS fornecidos no contexto. Nunca cite, descreva ou invente proposição, número, autor, voto ou número de projeto que não esteja no contexto.

2. Cada proposição citada deve vir acompanhada de uma descrição curta e simples do que propõe (com base na ementa). Formato: "PL 42/2026 — inclui a vacina contra Herpes Zoster no PNI".

3. Sempre que citar uma proposição, informe o autor quando o contexto o trouxer. Formato: "Autor: Dep. FULANO (SIGLA/UF)". Se não houver autor, "Autor: Poder Executivo".

4. Ao responder sobre VOTAÇÃO, descreva o resultado oficial (dados do contexto), a contagem dos votos e, se o contexto trouxer, como a pessoa/partido votou. Nunca invente votos.

5. Quando houver muitas proposições no contexto, agrupe por tema e apresente as mais representativas (cerca de 6 a 10), cada uma com descrição curta: "Sigla Número/Ano — o que propõe (Autor)". Sinalize limites quando o total real for maior que o listado.

6. Se o contexto não contiver o que foi pedido, diga claramente que não encontrou dados e sugira perguntas parecidas. Nunca preencha lacunas com dados ilustrativos ou inventados.

7. Não cite URLs no texto; use apenas "Sigla Número/Ano" (cada citação aparece automaticamente como botão). Responda apenas em português brasileiro, com frases curtas.

8. O histórico abaixo contém perguntas e respostas anteriores da mesma conversa. Use-o só para entender referências como "essas proposições", "os projetos citados", "e o autor?". Perguntas de contexto (pergunta atual) têm prioridade sobre o histórico; nunca invente dados que não estejam no contexto ou no histórico."""


def _mensagens(sistema: str, pergunta: str, historico: list[dict] | None) -> list[dict]:
    """mensagens para a llm: sistema, último histórico válido e a pergunta atual."""
    mensagens = [{"role": "system", "content": sistema}]
    turnos = {"usuario": "user", "assistente": "assistant"}
    for item in (historico or [])[-8:]:
        papel = (item.get("papel") or "").strip()
        conteudo = (item.get("conteudo") or "").strip()
        if papel not in turnos or not conteudo:
            continue
        mensagens.append({"role": turnos[papel], "content": conteudo})
    mensagens.append({"role": "user", "content": pergunta})
    return mensagens


def _extrair_filtros(pergunta: str) -> dict:
    """filtros determinísticos (regex): anos, sigla do tipo, partido e uf."""
    texto = pergunta.upper()
    filtros: dict = {"anos": [], "sigla_tipo": None, "partido": None, "uf": None}

    rng = re.search(
        r"\b(?:ENTRE|DE)\s+(19|20)\d{2}\b\s+(?:A|E|ATÉ)\s+(19|20)\d{2}\b", texto
    )
    if rng:
        ini, fim = (int(x) for x in rng.group(0).split() if x.isdigit())
        filtros["anos"] = list(range(ini, fim + 1))
    anos = re.findall(r"\b(?:19|20)\d{2}\b", texto)
    for a in anos:
        if int(a) not in filtros["anos"]:
            filtros["anos"].append(int(a))
    filtros["anos"] = sorted(filtros["anos"]) or None

    sigla = re.search(r"\b(" + "|".join(_SIGLAS_TIPO) + r")\s*\d{1,6}(?:\s*/\s*\d{4})?\b", texto)
    if sigla:
        filtros["sigla_tipo"] = sigla.group(1)

    marks = set(sigla.group(1) for sigla in sigla and [sigla] or [])
    for partido in _PARTIDOS:
        if partido in marks:
            continue
        if re.search(r"\b" + re.escape(partido) + r"\b(?!\s*\d)", texto):
            filtros["partido"] = partido
            break

    for uf in _UFS:
        if re.search(r"\b" + uf + r"\b", texto):
            filtros["uf"] = uf
            break
    if not filtros["uf"]:
        for frase in sorted(_UF_PALAVRAS, key=len, reverse=True):
            if re.search(r"\b" + re.escape(frase) + r"\b", texto):
                filtros["uf"] = _UF_PALAVRAS[frase]
                break
    if not filtros["anos"] and re.search(
        r"\b(esse|este|nesse|neste|no|em)\s+ano\b|\bano\s*(atual|vigente|corrente)\b"
        r"|\bhoje\b|\batualmente\b",
        pergunta, re.I,
    ):
        filtros["anos"] = [date.today().year]
    return filtros


# ---- intenções estruturadas -------------------------------------------------

_INTENT_VOTO = re.compile(r"\bvot\w*", re.I)
_INTENT_VOTOS_RECENTES = re.compile(
    r"foi votado|foram votad|o que votaram|o que foi votad|o que está sendo votado|"
    r"ultima[s]? votação|ultima[s]? votacao|últimas? votad|votações\s+(de|da|no|na|do) (hoje|ontem|semana|dia|recorde|dia de hoje)|"
    r"sessão de hoje|sessao de hoje|no plenário (hoje|essa semana|esta semana)|roll call",
    re.I,
)
_VERBOS_PROPOSTA = re.compile(
    r"o que é|o que e|o que qr|o que diz|conte|me fale|fala sobre|detalhe|explica"
    r"|situação|situacao|status|andamento|tramitação|tramitacao"
    r"|foi aprovad|foi rejeitad|foi sancionad|foi arquivad|foi retirad|foi votad"
    r"|vai virar|viro|passou|pra onde foi|foi para|foi pra",
    re.I,
)
_MENCAO_PROP = re.compile(
    r"\b(PEC|MPV|PDL|PLP|PLS|PL|REQ|PRC|EMC|RCP|REC|INC|SUB|OF|RIC|MSC|PDC)"
    r"\s*(\d{1,6})(?:\s*/\s*(\d{4}))?\b",
    re.I,
)


def _menção_proposicao(pergunta: str) -> tuple[str, int, int | None] | None:
    m = _MENCAO_PROP.search(pergunta)
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2)), int(m.group(3)) if m.group(3) else None


def _nome_autor(pergunta: str) -> str:
    """nome de deputado(a) citado na pergunta ("deputada Erica Hilton", "projetos da Erica Hilton")."""
    m = agregacao._nome_deputado(pergunta)
    if m:
        return m
    if not re.search(r"projet|propost|proposiç|proposica|apresent", pergunta, re.I):
        return ""
    for pat in [
        r"\b(?:dos|das|do|da|de)\s+([A-ZÀ-Ú][A-Za-zÀ-ú'’-]{2,}(?:\s+[A-Z][A-Za-zÀ-ú'’-]{2,}){0,3})\b",
    ]:
        m = re.search(pat, pergunta)
        if not m:
            continue
        nome = m.group(1).strip()
        if nome.split()[0].upper() in _PARTIDOS or nome.split()[0].upper() in _UFS:
            return ""
        return nome
    return ""


def _id_deputado(nome: str) -> dict | None:
    """resolver nome de deputado na API e devolver id/partido/uf do primeiro casamento."""
    try:
        deps = camara.deputados(nome=nome, itens=50)
    except camara.CamaraError:
        return None
    if not deps:
        return None
    alvo = nome.casefold()
    melhor = None
    for d in deps:
        txt = (d.get("nome") or "").casefold()
        if alvo in txt:
            if melhor is None or len(txt) < len(melhor.get("nome") or txt):
                melhor = d
    dep = melhor or deps[0]
    return {"id": dep.get("id"), "nome": dep.get("nome"),
            "partido": dep.get("partido") or "", "uf": dep.get("uf") or ""}


def _detectar_intencao(pergunta: str, filtros: dict) -> dict | None:
    """intenções estruturadas: votação de proposição, votações recentes, proposição específica."""
    men = _menção_proposicao(pergunta)
    if men:
        sigla, numero, ano = men
        if _INTENT_VOTO.search(pergunta):
            return {
                "tipo": "votacao", "sigla": sigla, "numero": numero, "ano": ano,
                "nome": _nome_autor(pergunta),
            }
        if _VERBOS_PROPOSTA.search(pergunta) or len(pergunta.split()) <= 8:
            return {"tipo": "proposicao", "sigla": sigla, "numero": numero, "ano": ano}
        return None
    if _INTENT_VOTOS_RECENTES.search(pergunta) or (
        _INTENT_VOTO.search(pergunta) and re.search(r"\b(hoje|ontem|semana|agora|esse\s+ano)\b", pergunta, re.I)
    ):
        return {"tipo": "votacoes_recentes"}
    return None


# ---- execução de intenções estruturadas ------------------------------------

def _rotulo_proposta(sigla: str, numero: int, ano: int | None) -> str:
    return f"{sigla} {numero}/{ano}" if ano else f"{sigla} {numero}"


def _achar_proposicao(sigla: str, numero: int, ano: int | None) -> dict | None:
    try:
        resumos = camara.proposicoes(sigla_tipo=sigla, numero=numero, ano=ano, itens=20)
    except camara.CamaraError:
        return None
    if not resumos and ano:
        try:
            resumos = camara.proposicoes(sigla_tipo=sigla, numero=numero, itens=20)
        except camara.CamaraError:
            resumos = []
    if not resumos:
        return None
    resumos.sort(key=lambda r: r.get("dataApresentacao") or "", reverse=True)
    return resumos[0]


def _voto_leigo(voto: str | None) -> str:
    return {
        "Sim": "a favor", "Não": "contra", "Abstenção": "abstenção",
        "Obstrução": "abstenção", "Art. 17": "abstém-se (art. 17)",
    }.get(voto, (voto if voto else "não informado"))


def _executar_votacao(inten: dict, historico: list[dict] | None = None) -> dict | None:
    prop = _achar_proposicao(inten["sigla"], inten["numero"], inten.get("ano"))
    if not prop:
        return None
    try:
        vots = camara.votacoes(id_proposicao=prop["id"], itens=20)
    except camara.CamaraError:
        vots = []
    rotulo = _rotulo_proposta(inten["sigla"], inten["numero"], inten.get("ano") or prop.get("ano"))
    if not vots:
        fonte = [{"titulo": rotulo, "url": prop.get("url"), "tipo": inten["sigla"],
                  "data": (prop.get("dataApresentacao") or "")[:10], "ano": prop.get("ano"), "score": None}]
        return {
            "resposta": (
                f"Não encontrei registro de votação em plenário para a {rotulo} nas fontes oficiais. "
                "A proposição pode ainda estar em comissões (que não disponibilizam votos nominais na API)."
            ),
            "fontes": fonte,
        }
    v = vots[0]
    try:
        detalhe = camara.votacao(v["id"])
    except camara.CamaraError:
        detalhe = {}

    contagem = detalhe.get("contagem") or {}
    descricao = detalhe.get("descricao") or ""
    if not contagem:
        m = re.search(r"Sim:\s*(\d+)[^.]*?Não:\s*(\d+)", descricao, re.I)
        if m:
            contagem = {"Sim": int(m.group(1)), "Não": int(m.group(2))}
    aprovacao = detalhe.get("aprovacao")
    if aprovacao in (True, 1, "true", "True", "aprovado"):
        resultado = "Aprovado"
    elif aprovacao in (False, 0, "false", "False", "rejeitado"):
        resultado = "Rejeitado"
    else:
        resultado = None
    partes = [f"{rotulo} — {descricao} ({detalhe.get('data') or ''})."]
    if resultado:
        partes.append(f"Resultado: {resultado}.")
    if contagem:
        resumo = ", ".join(f"{k}: {n}" for k, n in sorted(contagem.items()))
        partes.append(f"Votos: {resumo}.")
    orientacoes = [o for o in (detalhe.get("orientacoes") or []) if o.get("orientacao")]
    if orientacoes:
        linhas = [f"{o.get('partido_bloco')}: {_voto_leigo(o.get('orientacao'))}" for o in orientacoes[:8]]
        partes.append("Orientação das bancadas: " + "; ".join(linhas) + ".")

    nome = (inten.get("nome") or "").strip()
    dados = {
        "proposicao": rotulo + (" — " + (prop.get("ementa") or "")[:180] if prop.get("ementa") else ""),
        "votacao": {"data": detalhe.get("data"), "descricao": descricao,
                     "resultado": resultado, "contagem": contagem},
        "orientacoes": orientacoes[:8],
    }
    if nome:
        try:
            pessoal = camara.votacao(v["id"], nome)
        except camara.CamaraError:
            pessoal = {}
        achou = pessoal.get("votos_por_nome") or []
        if achou:
            voto = achou[0]
            dados["deputado"] = {"nome": voto.get("nome"), "partido": voto.get("partido"),
                                  "uf": voto.get("uf"), "voto": voto.get("voto")}
            partes.append(
                f"{voto.get('nome') or nome} ({voto.get('partido') or ''}/{voto.get('uf') or ''}) "
                f"votou {_voto_leigo(voto.get('voto'))}."
            )
        else:
            partes.append(f"Não encontrei o voto nominal de {nome} nesta votação.")
    partes.append("Dados oficiais: Portal da Câmara dos Deputados (Dados Abertos).")
    fontes = [
        {"titulo": rotulo, "url": prop.get("url"), "tipo": inten["sigla"],
         "data": (prop.get("dataApresentacao") or "")[:10], "ano": prop.get("ano"), "score": None},
    ]
    if v.get("url"):
        fontes.append({"titulo": f"Registro da votação ({v.get('data') or ''})", "url": v["url"],
                       "tipo": "VOTAÇÃO", "data": v.get("data") or "", "ano": None, "score": None})
    resposta = _redigir_intencao(inten.get("_pergunta", rotulo), dados, "\n".join(partes), historico)
    return {"resposta": resposta, "fontes": fontes}


def _executar_proposicao(inten: dict, historico: list[dict] | None = None) -> dict | None:
    prop = _achar_proposicao(inten["sigla"], inten["numero"], inten.get("ano"))
    if not prop:
        return None
    rotulo = _rotulo_proposta(inten["sigla"], inten["numero"], inten.get("ano") or prop.get("ano"))
    try:
        d = camara.proposicao(prop["id"])
    except camara.CamaraError:
        d = {}
    partes = [f"{rotulo}."]
    ementa = d.get("ementa") or prop.get("ementa") or ""
    if ementa:
        partes.append(f"O que propõe: {ementa}")
    autores = d.get("autores") or []
    if autores:
        lista = ", ".join(f"{a.get('nome') or ''}" + (f" ({a.get('partido') or ''}/{a.get('uf') or ''})" if a.get('partido') else "") for a in autores[:6])
        partes.append(f"Autores: {lista}.")
        if len(autores) > 6:
            partes[-1] = partes[-1][:-1] + f" e outros {len(autores) - 6}."
    if d.get("temas"):
        partes.append("Temas: " + ", ".join(d["temas"][:6]) + ".")
    if d.get("situacao"):
        partes.append(f"Situação atual: {d['situacao']}.")
    if d.get("regime"):
        partes.append(f"Regime de tramitação: {d['regime']}.")
    if d.get("orgao"):
        partes.append(f"Órgão: {d['orgao']}.")
    partes.append("Dados oficiais: Portal da Câmara dos Deputados (Dados Abertos).")
    dados = {
        "proposicao": rotulo,
        "ementa": ementa,
        "autores": autores[:6],
        "temas": d.get("temas") or [],
        "situacao": d.get("situacao"),
        "regime": d.get("regime"),
        "orgao": d.get("orgao"),
    }
    resposta = _redigir_intencao(inten.get("_pergunta", rotulo), dados, "\n".join(partes), historico)
    fonte = [{"titulo": rotulo, "url": prop.get("url"), "tipo": inten["sigla"],
              "data": (prop.get("dataApresentacao") or "")[:10], "ano": prop.get("ano"), "score": None}]
    return {"resposta": resposta, "fontes": fonte}


def _redigir_intencao(pergunta: str, dados: dict, fallback: str,
                      historico: list[dict] | None = None) -> str:
    """se a llm estiver disponível, redige a partir dos dados oficiais; senão usa o template."""
    if not config.tem_chaves:
        return fallback
    mensagens = _mensagens(_SISTEMA, (
        f"Pergunta:\n{pergunta}\n\nDADOS OFICIAIS (JSON, use somente estes dados):\n"
        f"{json.dumps(dados, ensure_ascii=False)}\n"
        "Responda em linguagem simples, sem inventar nada que não esteja nos dados."
    ), historico)
    try:
        msg = lm.completar(mensagens, tools=None)
        conteudo = (msg.get("content") or "").strip()
        return conteudo or fallback
    except LlmError:
        return fallback


# ---- busca (RAG + API estruturada) ------------------------------------------

def _recuperar(pergunta: str, filtros: dict) -> list[dict]:
    """prioriza o índice vetorial (RAG) quando configurado, com fallback para API estruturada."""
    if filtros.get("id_autor"):
        return _buscar_api(pergunta, filtros)
    if config.tem_rag:
        try:
            docs = rag.buscar(pergunta or "", filtros)
        except Exception:
            docs = []
        if docs:
            return docs
        try:
            docs = rag.busca_texto(filtros, pergunta or "")
        except Exception:
            docs = []
        if docs:
            return docs
    return _buscar_api(pergunta, filtros)


_STOPWORDS = {
    "projetos", "projeto", "proposta", "proposição", "proposicoes", "proposicao",
    "quero", "saber", "como", "qual", "quais", "quantos", "quantas", "existe",
    "existem", "sobre", "para", "com", "até", "ate", "por", "uma", "um", "dos",
    "das", "do", "da", "das", "desde", "entre", "depois", "antes", "quando",
    "quem", "onde", "porque", "pode", "tem", "ter", "ser", "está", "esta",
    "foi", "são", "sao", "podem", "relativos", "relativas", "não", "nao",
    "deputado", "deputada", "deputados", "deputadas", "parlamentar",
    "esse", "este", "essa", "esta", "autorou", "autora", "autoraram", "ano",
}


def _termos_chave(texto: str) -> str:
    """extrai termos para busca textual (API/keywords): palavras curtas de conteúdo."""
    encontradas = [t for t in re.findall(r"[A-Za-zÀ-ú]{4,}", texto) if t.casefold() not in _STOPWORDS]
    if not encontradas:
        return texto.strip()
    return " ".join(encontradas[:8])


def _termos_conteudo(pergunta: str, filtros: dict) -> str:
    """termos de conteúdo (ignora partido, uf, tipo, ano, nomes e artigos) para keywords."""
    palavras = re.findall(r"[A-Za-zÀ-ú]{4,}", pergunta)
    conteudo = []
    for t in palavras:
        up = t.upper()
        if up in _PARTIDOS or up in _UFS or up in _UF_PALAVRAS:
            continue
        if t.casefold() in _STOPWORDS:
            continue
        if up.startswith("DEPUTAD") or up in {s.upper() for s in _SIGLAS_TIPO}:
            continue
        conteudo.append(t.casefold())
    return " ".join(conteudo[:8])


def _buscar_api(pergunta: str, filtros: dict) -> list[dict]:
    """consulta determinística à API de dados abertos — todos os anos, com filtros de
    partido/uf/autor/tipo. Nunca depende de embeddings nem da LLM."""
    anos = filtros.get("anos") or []
    ano = anos[0] if len(anos) == 1 else None
    autor = filtros.get("id_autor")
    tem_filtro_estruturado = bool(autor or filtros.get("partido") or filtros.get("uf")
                                  or filtros.get("sigla_tipo") or ano or filtros.get("numero"))

    def consultar(terms: str | None) -> list[dict]:
        try:
            resumos = camara.proposicoes(
                ano=ano,
                sigla_tipo=filtros.get("sigla_tipo"),
                numero=filtros.get("numero"),
                keywords=terms,
                id_autor=autor,
                sigla_partido_autor=filtros.get("partido"),
                sigla_uf_autor=filtros.get("uf"),
                itens=100,
            )
        except camara.CamaraError:
            return []
        resumos.sort(key=lambda r: r.get("dataApresentacao") or "", reverse=True)
        docs = []
        for r in resumos[: config.rag_top_k]:
            try:
                doc_autores = camara.autores(r.get("id")) if r.get("id") else []
            except camara.CamaraError:
                doc_autores = []
            docs.append(montar_doc(r, autores=doc_autores))
        return docs

    termos = _termos_conteudo(pergunta, filtros)
    if tem_filtro_estruturado:
        docs = consultar(termos or None)
    else:
        docs = consultar(termos or _termos_chave(pergunta) or None)
    if not docs:
        docs = consultar(_termos_chave(pergunta) or None)
    return docs[: config.rag_top_k]


def _redigir(pergunta: str, docs: list[dict],
             historico: list[dict] | None = None) -> str:
    """LLM redige a partir do contexto; sem llm ou em falha, resposta determinística."""
    if not docs:
        return formato_simples(docs)
    if not config.tem_chaves:
        return formato_simples(docs)
    contexto = "\n\n".join(f"{i}. {d['doc']}" for i, d in enumerate(docs, 1))
    mensagens = _mensagens(_SISTEMA, f"Pergunta:\n{pergunta}\n\nCONTEXTO (dados oficiais da Câmara):\n{contexto}", historico)
    try:
        msg = lm.completar(mensagens, tools=None)
        conteudo = (msg.get("content") or "").strip()
        return conteudo or formato_simples(docs)
    except LlmError:
        return formato_simples(docs)


# ---- porta de entrada -------------------------------------------------------

def responder(pergunta: str, historico: list[dict] | None = None) -> dict:
    pergunta = (pergunta or "").strip()
    if not pergunta:
        return {"resposta": "Não consegui processar essa consulta.", "fontes": [],
                "raciocinio": "", "aviso": None}
    ano_vigente = date.today().year
    filtros = _extrair_filtros(pergunta)

    inten = _detectar_intencao(pergunta, filtros)
    if inten:
        inten["_pergunta"] = pergunta
        if inten["tipo"] == "votacao":
            out = _executar_votacao(inten, historico)
        else:
            out = _executar_proposicao(inten, historico)
        if out:
            return {**out, "raciocinio": "", "aviso": None}

    autor = _nome_autor(pergunta)
    if autor:
        dep = _id_deputado(autor)
        if dep:
            filtros["id_autor"] = dep["id"]

    ag = agregacao.detectar(pergunta, filtros)
    if ag:
        out = agregacao.executar(ag)
        if out:
            return {**out, "raciocinio": "", "aviso": None}

    docs = _recuperar(pergunta, filtros)
    fontes = [para_fonte(d, d.get("score")) for d in docs]
    return {"resposta": _redigir(pergunta, docs, historico), "fontes": fontes,
            "raciocinio": "", "aviso": None, "ano_vigente": ano_vigente}