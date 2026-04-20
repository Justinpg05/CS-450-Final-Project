from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

device = torch.device("cpu")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def get_embedding_from_pil(pil_img):
    face = mtcnn(pil_img)
    if face is None:
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(face)

    return embedding


def get_embedding_from_file(image_path):
    img = Image.open(image_path).convert("RGB")
    return get_embedding_from_pil(img)


known_faces = {
    "person1": "person1.jpg",
    "person2": "person2.jpg",
    "person3": "person3.jpg",
}

known_embeddings = {}
for name, image_path in known_faces.items():
    embedding = get_embedding_from_file(image_path)
    if embedding is not None:
        known_embeddings[name] = embedding
    else:
        print(f"No face found in {image_path}")

threshold = 0.7

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    raise SystemExit

print("Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    test_embedding = get_embedding_from_pil(pil_img)

    label = "No face"
    best_score = -1.0

    if test_embedding is not None:
        best_match = None

        for name, embedding in known_embeddings.items():
            score = F.cosine_similarity(test_embedding, embedding).item()
            if score > best_score:
                best_score = score
                best_match = name

        if best_score >= threshold:
            label = f"{best_match} ({best_score:.2f})"
        else:
            label = f"Unknown ({best_score:.2f})"

    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Webcam Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()