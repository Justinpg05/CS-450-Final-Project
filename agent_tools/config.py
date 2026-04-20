from pathlib import Path
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWN_FACES_DIR = BASE_DIR / "known_faces"
KNOWN_FACES_DIR.mkdir(exist_ok=True)

CAPTURE_PATH = BASE_DIR / "captured_face.jpg"
SERVED_FILE = BASE_DIR / "served.json"
THRESHOLD = 0.7

device = torch.device("cpu")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)