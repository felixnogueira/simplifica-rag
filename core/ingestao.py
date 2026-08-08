"""Ingestão RAG: baixa os arquivos oficiais da Câmara, monta um documento por
proposição, gera embeddings (Gemini, camada gratuita) e grava no PostgreSQL+pgvector.

Rode localmente (fora da Vercel), exemplo:

    python -m core.ingestao --anos 2016-2026
    python -m core.ingestao --anos 2001-2015      # anos antigos, se quiser
    python -m core.ingestao --anos 2020 --amostra 50   # teste rápido

Os arquivos baixados ficam em cache na pasta `dados/` (gitignored) para não
precisar baixar de novo a cada execução.
"""

import argparse
import json
import os
import sys
from datetime import date
from urllib.parse import urlsplit

import httpx

from .config import config
from . import embeddings
from . import rag
from .documento import montar_doc

_BASE = "https://dadosabertos.camara.leg.br/arquivos"

_CONJUNTOS = (
    ("proposicoes", "proposicoes-{ano}.json"),
    ("proposicoesAutores", "proposicoesAutores-{ano}.json"),
    ("proposicoesTemas", "proposicoesTemas-{ano}.json"),
)


def _baixar(cache: str, conjunto: str, ano: int) -> dict | None:
    """baixa (ou usa cache) e devolve o json {"dados": [...]}; None se 404."""
    os.makedirs(cache, exist_ok=True)
    caminho = os.path.join(cache, f"{conjunto}-{ano}.json")
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as f:
            return json.load(f)
    url = f"{_BASE}/{conjunto}/json/{conjunto}-{ano}.json"
    try:
        with httpx.Client(timeout=120) as c:
            r = c.get(url)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        dados = r.json()
    except httpx.HTTPError as e:
        print(f"  ! falha ao baixar {url}: {e}", file=sys.stderr)
        return None
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f)
    return dados


def _autores_da_proposicao(linhas: list[dict]) -> list[dict]:
    return sorted(linhas, key=lambda a: a.get("ordemAssinatura") or "999")


def _temas_da_proposicao(linhas: list[dict]) -> list[str]:
    vistos = {}
    for t in linhas:
        nome = t.get("tema")
        if nome:
            vistos.setdefault(nome, True)
    return list(vistos)


def _parse_anos(lista: list[str]) -> list[int]:
    anos: set[int] = set()
    for item in lista:
        if "-" in item:
            ini, fim = item.split("-", 1)
            anos.update(range(int(ini), int(fim) + 1))
        else:
            anos.add(int(item))
    return sorted(anos)


def _embeddings_de(textos: list[str], progresso: callable | None = None) -> list:
    return embeddings.gerar_embeddings_lote(textos, progresso=progresso)


def _ingestao(anos: list[int], cache: str, amostra: int | None, usar_conectado: bool, lote: int = 500) -> None:
    ano_corrente = date.today().year
    if not anos:
        anos = list(range(config.rag_desde, ano_corrente + 1))
    print(f"Ingestão de {len(anos)} ano(s): {anos[0]}-{anos[-1]}")
    rag.garantir_esquema()
    total_docs = total_gravados = 0
    for ano in anos:
        print(f"\n[{ano}]")
        dados = {nome: _baixar(cache, nome, ano) for nome, _ in _CONJUNTOS}
        props = (dados.get("proposicoes") or {}).get("dados") or []
        aut = (dados.get("proposicoesAutores") or {}).get("dados") or []
        tem = (dados.get("proposicoesTemas") or {}).get("dados") or []
        if not props:
            print("  sem proposições cadastradas — pulando")
            continue
        autores_por_id: dict[int, list[dict]] = {}
        for a in aut:
            autores_por_id.setdefault(a.get("idProposicao"), []).append(a)
        temas_por_id: dict[int, list[str]] = {}
        for t in tem:
            temas_por_id.setdefault(_id_de_uri(t.get("uriProposicao")), []).append(t.get("tema") or "")

        docs = []
        for p in props:
            if amostra and len(docs) >= amostra:
                break
            pid = p.get("id")
            if pid is None:
                continue
            docs.append(
                montar_doc(
                    p,
                    autores=_autores_da_proposicao(autores_por_id.get(pid) or []),
                    temas=temas_por_id.get(pid) or [],
                )
            )
        if not docs:
            print("  nada a indexar")
            continue

        if usar_conectado:
            existentes = rag.registrados([d["id"] for d in docs])
            docs = [d for d in docs if d["id"] not in existentes]
            if not docs:
                print("  todos já indexados — pulando")
                continue

        textos = [d["doc"] for d in docs]
        print(f"  gerando embeddings para {len(textos)} proposições...")
        n_emb = embeddings.quantidade_ok()
        gravados = 0
        sem_emb = 0
        for ini in range(0, len(textos), lote):
            fim = min(ini + lote, len(textos))
            print(f"  lote {ini + 1}-{fim}: embeddings...", end=" ", flush=True)
            emb = _embeddings_de(textos[ini:fim])
            batch = []
            for d, e in zip(docs[ini:fim], emb):
                if e is None:
                    sem_emb += 1
                    continue
                d["embedding"] = e
                batch.append(d)
            if batch:
                gravados += rag.upsert_docs(batch)
            print(f"{len(batch)} com embedding | {gravados} gravadas até agora", flush=True)
            print(
                f"  checkpoint: lote {ini // lote + 1}/{max(1, (len(textos) + lote - 1) // lote)}"
                f" — {fim}/{len(textos)} — {gravados} gravadas neste ano",
                file=sys.stderr,
            )
        if sem_emb:
            print(f"  ! {sem_emb} proposições ficaram sem embedding (ignoradas)", file=sys.stderr)
            if sem_emb == len(docs):
                print(
                    "  ! nenhum embedding gerado neste ano — confira a cota/estado dos "
                    "provedores (GEMINI_API_KEY, OLLAMA_URL/GROQ_API_KEY) antes de reprocessar",
                    file=sys.stderr,
                )
        else:
            print(f"  provedor(es) usados: {n_emb}")
        total_docs += len(docs)
        print(f"  ano: {gravados} gravadas (de {len(docs)} montadas)")
        total_gravados += gravados
    print(f"\nConcluído: {total_gravados} proposições gravadas.")
    print(rag.status())


def _id_de_uri(uri: str | None) -> int | None:
    if not uri:
        return None
    try:
        return int(urlsplit(uri.rstrip("/")).path.rsplit("/", 1)[-1])
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestão RAG da Câmara dos Deputados")
    parser.add_argument("--anos", nargs="*", default=[],
                        help="anos ou intervalos (2016-2026); padrão: últimos anos configurados")
    parser.add_argument("--cache", default=os.path.join(".", "dados"),
                        help="pasta para cachear os arquivos baixados")
    parser.add_argument("--amostra", type=int, default=None,
                        help="limite de proposições por ano (teste)")
    parser.add_argument("--forcar", action="store_true",
                        help="reindexa mesmo os anos que já estão no banco")
    parser.add_argument("--lote", type=int, default=500,
                        help="quantas proposições gravar por commit (pausa/resumo perde no máximo isso)")
    parser.add_argument("--limpar", action="store_true",
                        help="apaga o que existir no índice e começa do zero")
    args = parser.parse_args()
    if args.limpar:
        print("Apagando o índice atual...")
        rag.recriar_esquema()
    try:
        _ingestao(_parse_anos(args.anos) if args.anos else [], args.cache,
                  args.amostra, usar_conectado=not args.forcar, lote=args.lote)
    except KeyboardInterrupt:
        print("\n  pausado — rode o mesmo comando de novo para continuar "
              "(sem --limpar), ele pula o que já gravou.")


if __name__ == "__main__":
    main()