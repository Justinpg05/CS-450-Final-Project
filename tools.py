"""Public tool functions for the agent; implemented in face_engine.py."""
from face_engine import (
    capture,
    compare_embedding_to_memory,
    detect_face_path,
    embed_face_path,
    store_embedding,
)


def capture_image():
    return capture()


def detect_face(image):
    return detect_face_path(image)


def generate_embedding(face):
    return embed_face_path(face)


def compare_embeddings(embedding):
    return compare_embedding_to_memory(embedding)


def dispense_candy():
    print("[Agent] Candy dispensed!")
