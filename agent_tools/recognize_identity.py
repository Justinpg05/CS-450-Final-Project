import torch.nn.functional as F
from .config import THRESHOLD
from .load_known_embeddings import get_embedding

def recognize_identity(image_path, known_embeddings):
    emb = get_embedding(image_path)
    if emb is None:
        return {"success": False}

    best_match = None
    best_score = -1

    for name, known_emb in known_embeddings.items():
        score = F.cosine_similarity(emb, known_emb).item()
        if score > best_score:
            best_score = score
            best_match = name

    return {
        "success": True,
        "name": best_match,
        "score": best_score,
        "recognized": best_score >= THRESHOLD
    }