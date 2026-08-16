"""busca vetorial RAG sobre proposições da Câmara (PostgreSQL + pgvector)."""

import psycopg
from pgvector.psycopg import Vector, register_vector

from .config import config

_TABELA = "proposicoes_rag"
_TABELA_AUTORES = "proposicao_autores"
_TABELA_TEMAS = "proposicao_temas"

_SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {_TABELA} (
    id BIGINT PRIMARY KEY,
    sigla_tipo TEXT NOT NULL DEFAULT '',
    numero TEXT NOT NULL DEFAULT '',
    ano INTEGER,
    ementa TEXT NOT NULL DEFAULT '',
    temas TEXT NOT NULL DEFAULT '',
    autores TEXT NOT NULL DEFAULT '',
    siglas_autores TEXT NOT NULL DEFAULT '',
    situacao TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    data TEXT NOT NULL DEFAULT '',
    doc TEXT NOT NULL DEFAULT '',
    embedding vector({config.embedding_dim})
);
CREATE TABLE IF NOT EXISTS {_TABELA_AUTORES} (
    id_proposicao BIGINT NOT NULL REFERENCES {_TABELA}(id) ON DELETE CASCADE,
    nome TEXT NOT NULL DEFAULT '',
    partido TEXT NOT NULL DEFAULT '',
    uf TEXT NOT NULL DEFAULT '',
    ordem TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (id_proposicao, nome, ordem)
);
CREATE TABLE IF NOT EXISTS {_TABELA_TEMAS} (
    id_proposicao BIGINT NOT NULL REFERENCES {_TABELA}(id) ON DELETE CASCADE,
    tema TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (id_proposicao, tema)
);
"""


class RAGError(RuntimeError):
    pass


def conectar():
    if not config.database_url:
        raise RAGError("DATABASE_URL ausente — configure o banco PostgreSQL+pgvector")
    try:
        conn = psycopg.connect(config.database_url, autocommit=True, connect_timeout=15)
    except psycopg.Error as e:
        raise RAGError(f"falha ao conectar no banco: {e}")
    try:
        register_vector(conn)
    except Exception:
        pass
    return conn


def garantir_esquema() -> None:
    """cria a extensão, a tabela e o índice hnsw (idempotente)."""
    try:
        with conectar() as conn:
            conn.execute(_SCHEMA)
            conn.execute(
                f"""CREATE INDEX IF NOT EXISTS {_TABELA}_embedding_hnsw
                    ON {_TABELA} USING hnsw (embedding vector_cosine_ops)"""
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {_TABELA_AUTORES}_nome_idx ON {_TABELA_AUTORES}(nome)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {_TABELA_AUTORES}_prop_idx ON {_TABELA_AUTORES}(id_proposicao)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {_TABELA_TEMAS}_prop_idx ON {_TABELA_TEMAS}(id_proposicao)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {_TABELA}_ano_idx ON {_TABELA}(ano)"
            )
    except RAGError:
        raise
    except psycopg.Error as e:
        raise RAGError(f"não foi possível criar o esquema do banco (pgvector?): {e}")


def recriar_esquema() -> None:
    """dropa as tabelas do índice e recria vazias (para trocar de escopo/dimensão)."""
    with conectar() as conn:
        for tabela in (_TABELA_TEMAS, _TABELA_AUTORES, _TABELA):
            conn.execute(f"DROP TABLE IF EXISTS {tabela} CASCADE")
    garantir_esquema()


def _gravados_ids(conn, ids: list[int]) -> set[int]:
    if not ids:
        return set()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id FROM {_TABELA} WHERE id = ANY(%s)",
                (ids,),
            )
            return {r[0] for r in cur.fetchall()}
    except psycopg.Error:
        return set()


def upsert_docs(docs: list[dict]) -> int:
    """insere/atualiza documentos já com 'embedding' (e os autores/temas normalizados)."""
    if not docs:
        return 0
    with conectar() as conn:
        garantidos = [d for d in docs if d.get("embedding")]
        if not garantidos:
            return 0
        with conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {_TABELA} (
                        id, sigla_tipo, numero, ano, ementa, temas, autores,
                        siglas_autores, situacao, url, data, doc, embedding
                    ) VALUES (%(id)s, %(sigla_tipo)s, %(numero)s, %(ano)s, %(ementa)s,
                        %(temas)s, %(autores)s, %(siglas_autores)s, %(situacao)s,
                        %(url)s, %(data)s, %(doc)s, %(embedding)s)
                    ON CONFLICT (id) DO UPDATE SET
                        sigla_tipo = EXCLUDED.sigla_tipo,
                        numero = EXCLUDED.numero,
                        ano = EXCLUDED.ano,
                        ementa = EXCLUDED.ementa,
                        temas = EXCLUDED.temas,
                        autores = EXCLUDED.autores,
                        siglas_autores = EXCLUDED.siglas_autores,
                        situacao = EXCLUDED.situacao,
                        url = EXCLUDED.url,
                        data = EXCLUDED.data,
                        doc = EXCLUDED.doc,
                        embedding = EXCLUDED.embedding""",
                [{**d, "embedding": Vector(d["embedding"])} for d in garantidos],
            )
            _upsert_autores(cur, [d for d in garantidos if d.get("autores_detalhe")])
            _upsert_temas(cur, [d for d in garantidos if d.get("temas_lista")])
        return len(garantidos)


def _upsert_autores(cur, docs: list[dict]) -> None:
    linhas = []
    for d in docs:
        pid = d.get("id")
        for a in d["autores_detalhe"]:
            if a.get("nome"):
                linhas.append(
                    {
                        "id_proposicao": pid,
                        "nome": a["nome"],
                        "partido": a.get("partido") or "",
                        "uf": a.get("uf") or "",
                        "ordem": a.get("ordem") or "",
                    }
                )
    if not linhas:
        return
    cur.executemany(
        f"""INSERT INTO {_TABELA_AUTORES} (id_proposicao, nome, partido, uf, ordem)
            VALUES (%(id_proposicao)s, %(nome)s, %(partido)s, %(uf)s, %(ordem)s)
            ON CONFLICT (id_proposicao, nome, ordem) DO UPDATE SET
                partido = EXCLUDED.partido, uf = EXCLUDED.uf""",
        linhas,
    )


def _upsert_temas(cur, docs: list[dict]) -> None:
    linhas = []
    for d in docs:
        pid = d.get("id")
        for t in d["temas_lista"]:
            if t:
                linhas.append({"id_proposicao": pid, "tema": t})
    if not linhas:
        return
    cur.executemany(
        f"""INSERT INTO {_TABELA_TEMAS} (id_proposicao, tema)
            VALUES (%(id_proposicao)s, %(tema)s)
            ON CONFLICT (id_proposicao, tema) DO NOTHING""",
        linhas,
    )


def registrados(ids: list[int]) -> set[int]:
    """ids já presentes no banco (para ingestão só do que falta)."""
    if not ids:
        return set()
    with conectar() as conn:
        return _gravados_ids(conn, list(ids))


def _filtros(filtros: dict) -> tuple[str, list]:
    """WHERE determinístico a partir de filtros (anos, sigla_tipo, partido, uf)."""
    clausulas: list[str] = []
    params: list = []
    anos = list(filtros.get("anos") or [])
    if anos:
        clausulas.append("ano = ANY(%s)")
        params.append([int(a) for a in anos])
    sigla = (filtros.get("sigla_tipo") or "").strip().upper()
    if sigla:
        clausulas.append("sigla_tipo = %s")
        params.append(sigla)
    partido = (filtros.get("partido") or "").strip().upper()
    if partido:
        clausulas.append("siglas_autores ILIKE %s")
        params.append(f"%({partido}/%")
    uf = (filtros.get("uf") or "").strip().upper()
    if uf:
        clausulas.append("siglas_autores ILIKE %s")
        params.append(f"%/{uf})%")
    where = ("WHERE " + " AND ".join(clausulas)) if clausulas else ""
    return where, params


def buscar(pergunta: str, filtros: dict | None = None, top_k: int | None = None) -> list[dict]:
    """busca vetorial; retorna documentos rankeados por similaridade à pergunta."""
    from . import embeddings

    top_k = top_k or config.rag_top_k
    where, params = _filtros(filtros or {})
    emb = embeddings.gerar_embedding(pergunta or "")
    with conectar() as conn:
        sql = f"""SELECT id, sigla_tipo, numero, ano, ementa, temas, autores,
                         situacao, url, data, doc,
                         1 - (embedding <=> %s) AS score
                  FROM {_TABELA} {where}
                  ORDER BY embedding <=> %s
                  LIMIT %s"""
        args = [Vector(emb)] + params + [Vector(emb), top_k]
        with conn.cursor() as cur:
            cur.execute(sql, args)
            linhas = cur.fetchall()
    colunas = ("id", "sigla_tipo", "numero", "ano", "ementa", "temas", "autores",
               "situacao", "url", "data", "doc", "score")
    return [
        {c: (float(v) if c == "score" and v is not None else v) for c, v in zip(colunas, linha)}
        for linha in linhas
    ]


def busca_texto(filtros: dict | None = None, termos: str = "", top_k: int | None = None) -> list[dict]:
    """fallback sem embeddings: substring sobre doc/ementa."""
    top_k = top_k or config.rag_top_k
    where, params = _filtros(filtros or {})
    padroes = [f"%{t}%" for t in (termos or "").lower().split() if len(t) > 3]
    if padroes:
        clausula_pats = ["doc ILIKE ANY(%s)", "ementa ILIKE ANY(%s)"]
        if where:
            where += " AND (" + " OR ".join(clausula_pats) + ")"
        else:
            where = "WHERE (" + " OR ".join(clausula_pats) + ")"
        params.append(padroes)
        params.append(padroes)
    if not where:
        return []
    with conectar() as conn:
        sql = f"""SELECT id, sigla_tipo, numero, ano, ementa, temas, autores,
                         situacao, url, data, doc
                  FROM {_TABELA} {where}
                  ORDER BY ano DESC, id DESC
                  LIMIT %s"""
        with conn.cursor() as cur:
            cur.execute(sql, params + [top_k])
            linhas = cur.fetchall()
    colunas = ("id", "sigla_tipo", "numero", "ano", "ementa", "temas", "autores",
               "situacao", "url", "data", "doc")
    return [{c: v for c, v in zip(colunas, linha)} for linha in linhas]


def _clausulas_autor(uf: str | None, partido: str | None, ano: int | None) -> tuple[str, list]:
    clausulas = [f"p.id = a.id_proposicao"]
    params: list = []
    if uf:
        clausulas.append("a.uf = %s")
        params.append(uf.upper())
    if partido:
        clausulas.append("a.partido = %s")
        params.append(partido.upper())
    if ano:
        clausulas.append("p.ano = %s")
        params.append(int(ano))
    return " AND ".join(clausulas), params


def rank_deputados(uf: str | None = None, partido: str | None = None,
                   ano: int | None = None, limite: int = 10) -> list[dict]:
    """deputados (com partido/uf e total de proposições autoradas)."""
    clausulas, params = _clausulas_autor(uf, partido, ano)
    sql = f"""SELECT a.nome, MAX(a.partido) AS partido, MAX(a.uf) AS uf,
                COUNT(DISTINCT a.id_proposicao) AS total
            FROM {_TABELA_AUTORES} a
            JOIN {_TABELA} p ON {clausulas}
            WHERE a.nome <> ''
            GROUP BY a.nome
            ORDER BY total DESC, a.nome ASC
            LIMIT %s"""
    try:
        with conectar() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params + [int(limite)])
                return [
                    {"nome": r[0], "partido": r[1], "uf": r[2], "total": r[3]}
                    for r in cur.fetchall()
                ]
    except (RAGError, psycopg.Error):
        return []


def count_partido(partido: str, uf: str | None = None, ano: int | None = None) -> int | None:
    """total de proposições cujos autores (deputados) são do partido — exato no índice."""
    clausulas = ["a.id_proposicao = p.id", "a.partido = %s"]
    params: list = [partido.upper()]
    if uf:
        clausulas.append("a.uf = %s")
        params.append(uf.upper())
    if ano:
        clausulas.append("p.ano = %s")
        params.append(int(ano))
    sql = f"""SELECT COUNT(DISTINCT a.id_proposicao)
            FROM {_TABELA_AUTORES} a JOIN {_TABELA} p ON {" AND ".join(clausulas)}"""
    try:
        with conectar() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()[0]
    except (RAGError, psycopg.Error):
        return None


def rank_partidos(uf: str | None = None, ano: int | None = None, limite: int = 10) -> list[dict]:
    """partidos (sigla e total de proposições autoradas por seus deputados)."""
    clausulas, params = _clausulas_autor(uf, None, ano)
    sql = f"""SELECT a.partido, COUNT(DISTINCT a.id_proposicao) AS total
            FROM {_TABELA_AUTORES} a
            JOIN {_TABELA} p ON {clausulas}
            WHERE a.partido <> ''
            GROUP BY a.partido
            ORDER BY total DESC, a.partido ASC
            LIMIT %s"""
    try:
        with conectar() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params + [int(limite)])
                return [{"partido": r[0], "total": r[1]} for r in cur.fetchall()]
    except (RAGError, psycopg.Error):
        return []


def dados_deputado(nome: str) -> dict | None:
    """partido/uf atual (mais frequente) de um deputado no índice."""
    try:
        with conectar() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT a.nome, MAX(a.partido), MAX(a.uf), COUNT(DISTINCT a.id_proposicao)
                        FROM {_TABELA_AUTORES} a WHERE a.nome ILIKE %s
                        GROUP BY a.nome ORDER BY 4 DESC LIMIT 1""",
                    (f"%{nome}%",),
                )
                linha = cur.fetchone()
        if not linha:
            return None
        return {"nome": linha[0], "partido": linha[1] or "", "uf": linha[2] or "", "total": linha[3]}
    except (RAGError, psycopg.Error):
        return None


def temas_top(uf: str | None = None, partido: str | None = None,
              ano: int | None = None, limite: int = 10) -> list[dict]:
    """temas mais frequentes num grupo de proposições do índice."""
    clausulas, params = _clausulas_autor(uf, partido, ano)
    sql = f"""SELECT t.tema, COUNT(DISTINCT t.id_proposicao) AS total
            FROM {_TABELA_TEMAS} t
            JOIN {_TABELA_AUTORES} a ON a.id_proposicao = t.id_proposicao
            JOIN {_TABELA} p ON {clausulas}
            GROUP BY t.tema
            ORDER BY total DESC, t.tema ASC
            LIMIT %s"""
    try:
        with conectar() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params + [int(limite)])
                return [{"tema": r[0], "total": r[1]} for r in cur.fetchall()]
    except (RAGError, psycopg.Error):
        return []


def temas_de_deputado(nome: str, ano: int | None = None, limite: int = 3) -> list[dict]:
    """temas mais frequentes nas proposições de um deputado."""
    clausulas = ["a.id_proposicao = t.id_proposicao", "a.nome = %s"]
    params: list = [nome]
    if ano:
        clausulas.append("a.id_proposicao IN (SELECT id FROM {p} WHERE ano = %s)".replace("{p}", _TABELA))
        params.append(int(ano))
    sql = f"""SELECT t.tema, COUNT(DISTINCT t.id_proposicao) AS total
            FROM {_TABELA_TEMAS} t
            JOIN {_TABELA_AUTORES} a ON {" AND ".join(clausulas)}
            GROUP BY t.tema
            ORDER BY total DESC, t.tema ASC
            LIMIT %s"""
    try:
        with conectar() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params + [int(limite)])
                return [{"tema": r[0], "total": r[1]} for r in cur.fetchall()]
    except (RAGError, psycopg.Error):
        return []


def proposicoes_de_deputado(nome: str, ano: int | None = None, limite: int = 6) -> list[dict]:
    """proposições de um deputado (para fontes/botões)."""
    clausulas = ["a.id_proposicao = p.id", "a.nome = %s"]
    params: list = [nome]
    if ano:
        clausulas.append("p.ano = %s")
        params.append(int(ano))
    sql = f"""SELECT p.id, p.sigla_tipo, p.numero, p.ano, p.ementa, p.url, p.data
            FROM {_TABELA} p JOIN {_TABELA_AUTORES} a ON {" AND ".join(clausulas)}
            ORDER BY p.data DESC NULLS LAST, p.id DESC
            LIMIT %s"""
    try:
        with conectar() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params + [int(limite)])
                return [
                    {"id": r[0], "sigla_tipo": r[1], "numero": r[2],
                     "ano": r[3], "ementa": r[4], "url": r[5], "data": r[6]}
                    for r in cur.fetchall()
                ]
    except (RAGError, psycopg.Error):
        return []


def status() -> dict:
    try:
        with conectar() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*), MIN(ano), MAX(ano) FROM {_TABELA}")
                total, ano_min, ano_max = cur.fetchone()
        return {"total": total, "ano_minimo": ano_min, "ano_maximo": ano_max}
    except (RAGError, psycopg.Error, psycopg.OperationalError):
        return {"total": 0, "ano_minimo": None, "ano_maximo": None}