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


# Known people database
known_faces = {
    "person1": "person1.jpg",
    "person2": "person2.jpg",
    "person3": "person3.jpg",
}

# Generate embeddings for known faces
known_embeddings = {}
for name, image_path in known_faces.items():
    embedding = get_embedding(image_path)
    if embedding is not None:
        known_embeddings[name] = embedding

# Test image to identify
test_embedding = get_embedding("unknown.jpg")

if test_embedding is None:
    print("Could not detect a face in unknown.jpg")
    raise SystemExit

best_match = None
best_score = -1.0

for name, embedding in known_embeddings.items():
    score = F.cosine_similarity(test_embedding, embedding).item()
    print(f"{name}: {score:.4f}")

    if score > best_score:
        best_score = score
        best_match = name

print("\nBest match:", best_match)
print("Best score:", round(best_score, 4))

threshold = 0.7
if best_score >= threshold:
    print(f"Identified as {best_match}")
else:
    print("Unknown person")