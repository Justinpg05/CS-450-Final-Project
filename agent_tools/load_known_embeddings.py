from .config import KNOWN_FACES_DIR, mtcnn, resnet
from PIL import Image
import torch

def get_embedding(path):
    img = Image.open(path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        return None

    face = face.unsqueeze(0)
    with torch.no_grad():
        return resnet(face)

def load_known_embeddings():
    data = {}

    for file in KNOWN_FACES_DIR.iterdir():
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            emb = get_embedding(file)
            if emb is not None:
                data[file.stem] = emb

    return data