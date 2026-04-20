import os
from .config import KNOWN_FACES_DIR

def enroll_identity(image_path, name):
    save_path = KNOWN_FACES_DIR / f"{name}.jpg"
    os.replace(image_path, save_path)
    return str(save_path)