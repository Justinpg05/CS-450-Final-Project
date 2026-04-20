from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import torch.nn.functional as F

device = torch.device("cpu")

mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        print(f"No face detected in {image_path}")
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(face)

    return embedding

emb1 = get_embedding("face1.jpg")
emb2 = get_embedding("face3.jpg")

if emb1 is not None and emb2 is not None:
    similarity = F.cosine_similarity(emb1, emb2).item()
    print("Cosine similarity:", similarity)

    if similarity > 0.7:
        print("Likely the same person")
    else:
        print("Likely different people")