from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

device = torch.device("cpu")
print("Using device:", device)

mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

img = Image.open("test.jpg").convert("RGB")
face = mtcnn(img)

if face is None:
    print("No face detected")
    raise SystemExit

face = face.unsqueeze(0).to(device)

with torch.no_grad():
    embedding = resnet(face)

print("Embedding shape:", embedding.shape)
print("First 10 values:", embedding[0][:10].cpu().tolist())