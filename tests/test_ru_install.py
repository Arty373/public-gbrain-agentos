"""ru-install patches: OR full-text mode, embedding helpers."""
import asyncio

from services.recall_mcp.search import _fts_search
from services.shared.embedding import as_passage, as_query


class _Pool:
    def __init__(self):
        self.params = None

    async def fetch(self, query, *params):
        self.params = params
        return []


def test_fts_or_mode(monkeypatch):
    monkeypatch.setenv("GBRAIN_FTS_MODE", "or")
    pool = _Pool()
    asyncio.run(_fts_search(pool, "Что с конвекторами по 82.6?", "", []))
    assert pool.params[0] == "Что or конвекторами or по or 82.6"


def test_fts_default_mode_unchanged(monkeypatch):
    monkeypatch.delenv("GBRAIN_FTS_MODE", raising=False)
    pool = _Pool()
    asyncio.run(_fts_search(pool, "доступ к 1С", "", []))
    assert pool.params[0] == "доступ к 1С"


def test_e5_prefixes():
    assert as_query("intfloat/multilingual-e5-base", "q") == "query: q"
    assert as_passage("intfloat/multilingual-e5-base", "p") == "passage: p"
    assert as_query("sentence-transformers/x", "q") == "q"
