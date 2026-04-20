import json
from .config import SERVED_FILE

def save_served(served):
    with open(SERVED_FILE, "w") as f:
        json.dump(list(served), f)