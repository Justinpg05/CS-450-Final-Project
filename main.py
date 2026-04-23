"""CLI entry: runs the candy agent in a loop (see agent.py, tools.py, face_engine.py)."""
from agent import run_agent

if __name__ == "__main__":
    print("Candy agent running in a loop. Press Ctrl+C to stop.")
    try:
        while True:
            run_agent()
    except KeyboardInterrupt:
        print("\nStopped agent loop.")
