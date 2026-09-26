"""guia estático do processo legislativo: responde perguntas gerais de tramitação
sem depender de busca, com base na página oficial da Câmara dos Deputados.

Perguntas como "como funciona o processo legislativo?" não têm resposta boa no
índice vetorial (que só guarda proposições específicas) — este módulo injeta um
guia confiável no contexto quando a pergunta é genérica sobre o processo."""

import re
import unicodedata

FONTE = {
    "titulo": "Câmara dos Deputados — Entenda o processo legislativo",
    "url": "https://www.camara.leg.br/entenda-o-processo-legislativo/",
    "tipo": "GUIA",
    "data": "",
    "ano": None,
    "score": None,
}

_GUIA = """Um projeto de lei ordinária na Câmara dos Deputados segue este caminho (fonte: Câmara dos Deputados — "Entenda o processo legislativo"):

1. Publicação — quem pode propor: qualquer deputado ou senador, qualquer comissão da Câmara, do Senado ou do Congresso Nacional, o presidente da República, o Supremo Tribunal Federal, os tribunais superiores, o procurador-geral da República e os cidadãos (iniciativa popular). A proposta recebe uma numeração (por exemplo PL 1234/2025). Todos os projetos de lei começam a tramitar pela Câmara dos Deputados, exceto os apresentados por senador ou comissão do Senado (esses começam pelo Senado).

2. Análise de mérito nas comissões temáticas: o presidente da Câmara distribui a proposta para as comissões temáticas ligadas ao assunto, até 3 no máximo. Em cada comissão, um relator recebe e analisa as sugestões (emendas) dos deputados e emite parecer sobre o mérito da proposta. Se o mérito for analisado por muitas comissões, a Câmara pode criar uma comissão especial, para que a tramitação não fique muito longa.

3. Análise de admissibilidade pela CFT e pela CCJC: as comissões de Finanças e Tributação (CFT) e de Constituição, Justiça e de Cidadania (CCJC) são as últimas a analisar os projetos. As propostas que criam gastos ou tratam de finanças públicas passam pela CFT, que avalia se estão adequadas ao Orçamento federal. Todas as propostas passam por último pela CCJC, que avalia se estão de acordo com a Constituição. Se a CFT ou a CCJC considerarem que a proposta não pode ser admitida, por não estar adequada ao Orçamento ou por ser inconstitucional, ela é arquivada.

4. Tramitação conclusiva ou votação no Plenário: a maioria dos projetos em tramitação na Câmara só precisa passar pelas comissões (caráter conclusivo). Se forem aprovados por todas, vão direto para o Senado — ou para sanção presidencial, se já tiverem passado pelo Senado. Se aprovados por algumas e rejeitados por outras, vão para o Plenário. Precisam ser votados no Plenário, entre outros: projetos de lei complementar; de código; de iniciativa popular; de comissão; aprovados pelo Plenário do Senado; em regime de urgência; e os que receberam pareceres divergentes ou recurso. No Plenário, o quórum mínimo é de 257 deputados (maioria absoluta) e, para aprovar um projeto de lei ordinária, basta a maioria simples dos votos, em turno único.

5. Depois do Plenário: se a tramitação começou na Câmara, o projeto segue para o Senado, que analisa e vota a proposta, podendo alterá-la. Se alterado, volta para a Câmara, que analisa apenas as alterações, podendo mantê-las ou recuperar o texto original. Se a proposta veio do Senado e for aprovada na Câmara sem alterações, segue direto para a sanção.

6. Sanção e veto: aprovado nas duas casas (ou vindo do Senado sem alterações), o projeto vai ao presidente da República, que tem o prazo de 15 dias úteis para sancionar ou vetar, no todo ou em partes. Se sancionado, o projeto se torna lei e é publicado no Diário Oficial da União. Se vetado, os vetos voltam para análise do Congresso Nacional, em sessão conjunta da Câmara e do Senado: se mantidos, a lei fica como está; se derrubados, os trechos antes vetados passam a integrar a lei.

Referência: Câmara dos Deputados — "Entenda o processo legislativo"."""


def guia() -> str:
    """texto completo do guia, usado como contexto para a llm."""
    return _GUIA


def explicacao_simples() -> str:
    """resposta determinística em linguagem simples (sem llm)."""
    return (
        "Um projeto de lei ordinária passa por estas etapas na Câmara dos Deputados, em "
        "resumo: é apresentado e publicado (podem propor deputados, senadores, comissões, o "
        "presidente da República, tribunais superiores, o procurador-geral da República ou "
        "cidadãos, via iniciativa popular); o mérito é analisado em comissões temáticas, até 3, "
        "com parecer de um relator e emendas dos deputados; por último passam pelas comissões "
        "de Finanças e Tributação e de Constituição e Justiça, que avaliam a adequação ao "
        "Orçamento e à Constituição. A maioria dos projetos só passa pelas comissões e segue "
        "direto para o Senado ou para a sanção; os demais são votados no Plenário, com quórum "
        "de 257 deputados e aprovação pela maioria dos votos. Ao final, o presidente da "
        "República tem 15 dias úteis para sancionar (transformando o projeto em lei, publicada "
        "no Diário Oficial da União) ou vetar; os vetos voltam ao Congresso Nacional, que decide "
        "se os mantém ou derruba. Fonte: Câmara dos Deputados."
    )


def _norm(texto: str) -> str:
    s = unicodedata.normalize("NFD", texto or "")
    return re.sub(r"\s+", " ", "".join(c for c in s if not unicodedata.combining(c))).casefold()


_POS_PROCESSUAL = [
    r"processo legislativ",
    r"etapas? .{0,40}(processo legislativ|tramita)",
    r"fases? .{0,40}(processo legislativ|tramita)",
    r"(etapas?|fases?|passos) .{0,50}(um |o |a )?(projeto|proposta|proposi|lei) .{0,30}(vira|virar|vire|ate ser aprovad)",
    r"como funciona o processo",
    r"como .{0,60}(um |o |a )?(projeto|proposta|proposi|lei) .{0,40}(vira|virar|se torna|ate virar|e arquivad|arquivament)",
    r"ate virar lei",
    r"passos (de|do|para) (processo|tramita)",
    r"o que (e|significa) (o )?(processo legislativo|tramitacao|sancao|veto|arquivamento)",
    r"o que significa .{0,40}(aguardando apreciacao|senado|comissao especial)",
    r"o que acontece .{0,40}(projeto|proposta|aprova|senad|vet)",
    r"depois de aprovad",
]

_NEG_RECUPERAR = [
    r"\bquais?\s+(sao\s+)?(os|as)?\s*(projetos|propostas|proposi)",
    r"\bquem\b",
    r"\bquando\s+(foi|vai|ser|votad|aprovad|sancionad|arquivad)",
    r"\bquant",
    r"\b(19|20)\d{2}\b",
    r"deputad",
    r"(foi|foram)\s+votad",
]


def detectar(pergunta: str) -> bool:
    """True quando a pergunta pede explicação genérica do processo legislativo."""
    texto = _norm(pergunta)
    if not texto:
        return False
    if re.search(r"processo legislativ", texto):
        return True
    if not any(re.search(p, texto) for p in _POS_PROCESSUAL):
        return False
    return not any(re.search(p, texto) for p in _NEG_RECUPERAR)