import cv2
from .config import CAPTURE_PATH

def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            cv2.imwrite(str(CAPTURE_PATH), frame)
            break

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return str(CAPTURE_PATH)