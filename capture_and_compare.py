from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
import os
from pathlib import Path
import json
import time

device = torch.device("cpu")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

KNOWN_FACES_DIR = Path("known_faces")
KNOWN_FACES_DIR.mkdir(exist_ok=True)

CAPTURE_PATH = "captured_face.jpg"
THRESHOLD = 0.7
SERVED_FILE = "served.json"


# ---------- JSON MEMORY ----------
def load_served():
    if not os.path.exists(SERVED_FILE):
        return set()
    with open(SERVED_FILE, "r") as f:
        return set(json.load(f))


def save_served(served_set):
    with open(SERVED_FILE, "w") as f:
        json.dump(list(served_set), f)


served = load_served()


# ---------- EMBEDDINGS ----------
def get_embedding_from_file(image_path: str):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        print(f"No face detected in {image_path}")
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(face)

    return embedding


def load_known_embeddings():
    known_embeddings = {}

    for image_path in KNOWN_FACES_DIR.iterdir():
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        name = image_path.stem
        embedding = get_embedding_from_file(str(image_path))

        if embedding is not None:
            known_embeddings[name] = embedding

    return known_embeddings


# ---------- CAMERA ----------
_WARMUP_FRAMES = 30
_FACE_WAIT_SEC = 15.0
_MIN_FACE_PX = 60
_HAAR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _frame_has_face(bgr) -> bool:
    if _HAAR.empty():
        return True
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _HAAR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(_MIN_FACE_PX, _MIN_FACE_PX),
    )
    return len(faces) > 0


def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        return False

    print("Webcam ready — capturing automatically when a face is in frame (q to cancel)")

    for _ in range(_WARMUP_FRAMES):
        ret, _ = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            return False

    deadline = time.monotonic() + _FACE_WAIT_SEC
    last_frame = None
    overlay = (
        "Hold still — capturing automatically…"
        if not _HAAR.empty()
        else "Capturing automatically…"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame
        cv2.putText(
            frame,
            overlay,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return False

        if _frame_has_face(frame):
            cv2.imwrite(CAPTURE_PATH, frame)
            print("Captured image")
            cap.release()
            cv2.destroyAllWindows()
            return True

        if time.monotonic() >= deadline:
            break

    if last_frame is not None:
        cv2.imwrite(CAPTURE_PATH, last_frame)
        print("Captured image (timeout — best effort frame)")

    cap.release()
    cv2.destroyAllWindows()
    return last_frame is not None


# ---------- MATCHING ----------
def find_best_match(test_embedding, known_embeddings):
    best_match = None
    best_score = -1.0

    for name, embedding in known_embeddings.items():
        score = F.cosine_similarity(test_embedding, embedding).item()
        print(f"{name}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_match = name

    return best_match, best_score


# ---------- ENROLL ----------
def enroll_new_identity():
    name = input("New person. Enter name: ").strip()
    if not name:
        return None

    save_path = KNOWN_FACES_DIR / f"{name}.jpg"
    os.replace(CAPTURE_PATH, save_path)

    print(f"Saved {name}")

    embedding = get_embedding_from_file(str(save_path))
    return name, embedding


# ---------- CANDY LOGIC ----------
def handle_candy(name):
    global served

    if name not in served:
        served.add(name)
        save_served(served)
        print(f"{name} → DISPENSED CANDY")
    else:
        print(f"{name} → CANDY NOT DISPENSED")


# ---------- MAIN ----------
def main():
    known_embeddings = load_known_embeddings()

    if not capture_image():
        return

    test_embedding = get_embedding_from_file(CAPTURE_PATH)
    if test_embedding is None:
        print("No face detected")
        return

    if not known_embeddings:
        result = enroll_new_identity()
        if result:
            name, embedding = result
            handle_candy(name)
        return

    best_match, best_score = find_best_match(test_embedding, known_embeddings)

    print("\nBest:", best_match, best_score)

    if best_score >= THRESHOLD:
        handle_candy(best_match)
    else:
        result = enroll_new_identity()
        if result:
            name, embedding = result
            handle_candy(name)


if __name__ == "__main__":
    main()