"""FastEmbed model loading shared by recall-mcp and ingest-worker.

Adds ``intfloat/multilingual-e5-base`` (768 dims, ~3x faster than e5-large on CPUs without AVX2)
as a custom FastEmbed model, and the e5 ``query: `` / ``passage: `` prefixes the e5 family was
trained with.
"""
from __future__ import annotations

from typing import Any

E5_BASE = "intfloat/multilingual-e5-base"


def load_text_embedding(model_name: str) -> Any:
    from fastembed import TextEmbedding

    if model_name == E5_BASE:
        known = {m["model"] for m in TextEmbedding.list_supported_models()}
        if E5_BASE not in known:
            from fastembed.common.model_description import ModelSource, PoolingType

            TextEmbedding.add_custom_model(
                model=E5_BASE, pooling=PoolingType.MEAN, normalization=True,
                sources=ModelSource(hf=E5_BASE), dim=768, model_file="onnx/model.onnx",
            )
    return TextEmbedding(model_name=model_name)


def is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def as_query(model_name: str, text: str) -> str:
    return f"query: {text}" if is_e5(model_name) else text


def as_passage(model_name: str, text: str) -> str:
    return f"passage: {text}" if is_e5(model_name) else text
