"""porta de entrada do RAG: usa Qdrant Cloud quando configurado, senão PostgreSQL+pgvector."""

from .config import config

if config.tem_qdrant:
    from .qdrant import (
        RAGError,
        conectar,
        garantir_esquema,
        recriar_esquema,
        upsert_docs,
        registrados,
        buscar,
        busca_texto,
        rank_deputados,
        count_partido,
        rank_partidos,
        dados_deputado,
        temas_top,
        temas_de_deputado,
        proposicoes_de_deputado,
        status,
    )
else:
    from .rag_pg import (
        RAGError,
        conectar,
        garantir_esquema,
        recriar_esquema,
        upsert_docs,
        registrados,
        buscar,
        busca_texto,
        rank_deputados,
        count_partido,
        rank_partidos,
        dados_deputado,
        temas_top,
        temas_de_deputado,
        proposicoes_de_deputado,
        status,
    )