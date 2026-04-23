"""Agent decision loop and reasoning logs; uses tools.py (wrappers over face_engine.py)."""
import datetime

from tools import (
    capture_image,
    compare_embeddings,
    detect_face,
    dispense_candy,
    generate_embedding,
    store_embedding,
)

# Stricter than agent_tools.config.THRESHOLD (0.7), which is used for enroll/recognize flows.
AGENT_SIMILARITY_THRESHOLD = 0.8


def log(reason: str, **kwargs) -> None:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    extra = (" " + " ".join(f"{k}={v!r}" for k, v in kwargs.items())) if kwargs else ""
    print(f"[{ts}] [Agent] {reason}{extra}")


def run_agent() -> None:
    print("[Agent] Starting iteration...")
    image = capture_image()
    if image is None:
        log("Abort: no image captured (quit or camera error).")
        return

    log("Image captured.", path=image)

    face = detect_face(image)
    if face is None:
        log("No face detected in capture.")
        print("[Agent] No face detected")
        return

    log("Face OK; computing embedding.", handle=face)

    embedding = generate_embedding(face)
    if embedding is None:
        log("Could not produce embedding.")
        return

    similarity = compare_embeddings(embedding)
    log(
        "Similarity vs known faces + visitor memory.",
        best_similarity=round(similarity, 4),
        threshold=AGENT_SIMILARITY_THRESHOLD,
    )

    if similarity > AGENT_SIMILARITY_THRESHOLD:
        log("Decision: already seen - deny candy.")
        print("[Agent] Decision: Already seen - no candy")
        return

    log("Decision: new visitor - dispense and remember embedding.")
    dispense_candy()
    store_embedding(embedding)
