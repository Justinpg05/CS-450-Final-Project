"""Face capture, embedding, and visitor-memory comparison (FaceNet via agent_tools)."""
from __future__ import annotations

import uuid
from pathlib import Path

import torch
import torch.nn.functional as F

from agent_tools.capture_image import capture_image
from agent_tools.config import BASE_DIR
from agent_tools.load_known_embeddings import get_embedding, load_known_embeddings

DATABASE_DIR = BASE_DIR / "database"


def _ensure_database() -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)


def capture() -> str | None:
    return capture_image()


def detect_face_path(image: str | None) -> str | None:
    if image is None:
        return None
    if get_embedding(image) is None:
        return None
    return image


def embed_face_path(face_path: str):
    return get_embedding(face_path)


def _load_stored_tensor(path: Path) -> torch.Tensor | None:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    return None


def _iter_visitor_embeddings():
    _ensure_database()
    for path in DATABASE_DIR.glob("visitor_*.pt"):
        tensor = _load_stored_tensor(path)
        if tensor is not None:
            yield tensor


def compare_embedding_to_memory(embedding) -> float:
    best = -1.0
    for _, known_emb in load_known_embeddings().items():
        score = F.cosine_similarity(embedding, known_emb).item()
        if score > best:
            best = score
    for stored in _iter_visitor_embeddings():
        score = F.cosine_similarity(embedding, stored).item()
        if score > best:
            best = score
    return best


def store_embedding(embedding) -> None:
    _ensure_database()
    out = DATABASE_DIR / f"visitor_{uuid.uuid4().hex}.pt"
    torch.save(embedding.detach().cpu(), out)
