import time

import cv2

from .config import CAPTURE_PATH

# Let auto-exposure settle; then capture when a face is visible (no keypress).
WARMUP_FRAMES = 30
FACE_WAIT_SEC = 15.0
MIN_FACE_PX = 60

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _frame_has_face(bgr) -> bool:
    if _CASCADE.empty():
        return True
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(MIN_FACE_PX, MIN_FACE_PX),
    )
    return len(faces) > 0


def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return None

    for _ in range(WARMUP_FRAMES):
        ret, _ = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            return None

    deadline = time.monotonic() + FACE_WAIT_SEC
    last_frame = None
    overlay = (
        "Hold still — capturing automatically…"
        if not _CASCADE.empty()
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
            return None

        if _frame_has_face(frame):
            cv2.imwrite(str(CAPTURE_PATH), frame)
            cap.release()
            cv2.destroyAllWindows()
            return str(CAPTURE_PATH)

        if time.monotonic() >= deadline:
            break

    if last_frame is not None:
        cv2.imwrite(str(CAPTURE_PATH), last_frame)

    cap.release()
    cv2.destroyAllWindows()
    return str(CAPTURE_PATH) if last_frame is not None else None
