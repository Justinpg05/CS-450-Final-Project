import json
from .config import SERVED_FILE

def load_served():
    if not SERVED_FILE.exists():
        return set()

    with open(SERVED_FILE, "r") as f:
        return set(json.load(f))