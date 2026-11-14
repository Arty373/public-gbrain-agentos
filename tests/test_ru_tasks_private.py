"""ru-install: private agent area (create_private_note, GBRAIN_PRIVATE_AREAS) and task-board guards."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from services.memory_mcp import path_guard
from services.memory_mcp.tools import _REQUEST_AUTH, register_tools
from services.shared.auth import AgentContext, restrict_read_scopes
from services.shared.tool_gating import should_register_tool
from services.task_mcp import server as task_server
from services.task_mcp import store as task_store
from tests.test_memory_mcp import ToolRecorder
from tests.test_note_paths import DocStorePool


# --- private area: config + path guard ---------------------------------------

def test_private_areas_parse(monkeypatch):
    assert path_guard.private_areas("anna:40-anna") == {"anna": "40-anna"}
    assert path_guard.private_areas("") == {}
    for bad in ("anna:30-decisions", "anna:40-a/b", "anna:"):
        with pytest.raises(RuntimeError):
            path_guard.private_areas(bad)


def test_private_area_scope_is_allowed_only_when_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("GBRAIN_PRIVATE_AREAS", raising=False)
    with pytest.raises(ValueError):
        path_guard.validate_path("40-anna/x.md", str(tmp_path))
    monkeypatch.setenv("GBRAIN_PRIVATE_AREAS", "anna:40-anna")
    assert path_guard.validate_path("40-anna/x.md", str(tmp_path)).name == "x.md"


def test_create_private_note_is_core():
    assert should_register_tool("memory_mcp", "create_private_note", "core")


def _tools(tmp_path):
    pool = DocStorePool()
    rec = ToolRecorder()

    async def pool_fn():
        return pool

    register_tools(rec, str(tmp_path), pool_fn, tool_set="core")
    return pool, rec


async def _call(rec, ctx, **kw):
    with patch("services.memory_mcp.tools.authenticate", new=AsyncMock(return_value=ctx)):
        t = _REQUEST_AUTH.set("Bearer token")
        try:
            return await rec.tools["create_private_note"](**kw)
        finally:
            _REQUEST_AUTH.reset(t)


@pytest.mark.asyncio
async def test_private_note_written_to_own_area(monkeypatch, tmp_path):
    monkeypatch.setenv("GBRAIN_PRIVATE_AREAS", "anna:40-anna")
    pool, rec = _tools(tmp_path)
    anna = AgentContext(agent="anna", write_scopes=["40-anna"], read_scopes=["40-anna"])
    out = await _call(rec, anna, title="Выверка", body="текст", tags=["x"])
    assert out.startswith("created: 40-anna/")
    assert list(pool.docs)[0].startswith("40-anna/")
    assert (tmp_path / out.split(": ", 1)[1]).is_file()


@pytest.mark.asyncio
async def test_private_note_denied_without_area_or_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("GBRAIN_PRIVATE_AREAS", "anna:40-anna")
    _, rec = _tools(tmp_path)
    eva = AgentContext(agent="eva", write_scopes=["*"], read_scopes=["*"])
    with pytest.raises(PermissionError):
        await _call(rec, eva, title="t", body="b")
    anna_no_scope = AgentContext(agent="anna", write_scopes=["*"], read_scopes=[])
    with pytest.raises(PermissionError):  # wildcard does not count as owning the area
        await _call(rec, anna_no_scope, title="t", body="b")


def test_explicit_read_scopes_exclude_private_area():
    eva = AgentContext(agent="eva", write_scopes=[], read_scopes=["10-system", "30-decisions", "50-knowledge"])
    assert "40-anna" not in restrict_read_scopes(eva, None)
    with pytest.raises(PermissionError):
        restrict_read_scopes(eva, ["40-anna"])


# --- task board guards --------------------------------------------------------

def test_assignee_validation(monkeypatch):
    monkeypatch.setenv("GBRAIN_TASK_ASSIGNEES", "eva,bob,anna")
    task_server._validate_assignee("anna")
    task_server._validate_assignee(None)
    with pytest.raises(ValueError):
        task_server._validate_assignee("mallory")
    monkeypatch.delenv("GBRAIN_TASK_ASSIGNEES")
    task_server._validate_assignee("anyone")


def _task(created_by="eva", assignee="anna"):
    now = datetime.now(timezone.utc).isoformat()
    return {"id": 1, "title": "t", "description": "", "status": "new", "assignee": assignee,
            "priority": "medium", "created_by": created_by, "created_at": now, "updated_at": now, "metadata": {}}


@pytest.mark.asyncio
async def test_only_creator_or_assignee_may_change(monkeypatch):
    monkeypatch.delenv("GBRAIN_TASK_PARTY_ONLY", raising=False)
    monkeypatch.setattr(task_store, "get_task", AsyncMock(return_value=_task()))
    for agent in ("eva", "anna"):
        await task_server._require_party(None, 1, AgentContext(agent, ["10-tasks"], []))
    with pytest.raises(PermissionError):
        await task_server._require_party(None, 1, AgentContext("bob", ["10-tasks"], []))
    monkeypatch.setenv("GBRAIN_TASK_PARTY_ONLY", "0")
    await task_server._require_party(None, 1, AgentContext("bob", ["10-tasks"], []))


class _ListPool:
    def __init__(self):
        self.query, self.params = "", ()

    async def fetch(self, query, *params):
        self.query, self.params = query, params
        return []


@pytest.mark.asyncio
async def test_list_filters_involving_and_since():
    pool = _ListPool()
    await task_store.list_tasks(pool, involving="anna", updated_since="2026-11-14T20:00:00+00:00", limit=5)
    assert "(assignee = $1 OR created_by = $1)" in pool.query
    assert "updated_at > $2" in pool.query and "ORDER BY updated_at DESC" in pool.query
    assert pool.params[0] == "anna" and pool.params[2] == 5
    with pytest.raises(ValueError):
        await task_store.list_tasks(pool, updated_since="2026-11-14T20:00:00")  # naive


class _UpdPool:
    def __init__(self):
        self.query = ""

    async def fetchrow(self, query, *params):
        self.query = query
        return None


@pytest.mark.asyncio
async def test_metadata_patch_merges():
    pool = _UpdPool()
    with pytest.raises(ValueError):  # row None -> not found, but the SQL is built
        await task_store.update_task(pool, 1, "anna", metadata_patch={"result": "ok"})
    assert "metadata = metadata || $1::jsonb" in pool.query
    with pytest.raises(ValueError):
        await task_store.update_task(pool, 1, "anna", metadata={}, metadata_patch={"a": 1})
