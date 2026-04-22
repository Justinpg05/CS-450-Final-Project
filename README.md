# Face Candy Agent Tools

This project provides a set of Python tools for a face-recognition candy dispenser system.

The system:
- captures a face from the webcam
- converts it into an embedding (FaceNet)
- compares it to known identities
- stores new identities if needed
- remembers who already received candy
- outputs either:
  - DISPENSED CANDY
  - CANDY NOT DISPENSED

---

SETUP

1. Clone the repository

git clone <YOUR_REPO_URL>
cd face-candy-agent

2. Create a virtual environment

macOS / Linux:
python3.11 -m venv .venv
source .venv/bin/activate

Windows:
python -m venv .venv
.venv\Scripts\activate

3. Install dependencies (DO NOT CHANGE VERSIONS)

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

4. Verify setup

python

then run:

import torch, PIL, facenet_pytorch
print("Setup works")

If that prints without errors, you're ready.





IF YOU WANNA TEST OUT THE PROGRAM GO TO capture_and_compare.py
when you run the file it will open up a window showing your webcam
if you press space it will capture an image of your face, check the terminal
it will ask you to assign a name to an identity if its the first time

AGENTIC ENTRY (embedding memory)

Run: python main.py

Flow: main.py → agent.py (decision + reasoning logs) → tools.py (thin wrappers) → face_engine.py (capture, embeddings, cosine vs known_faces + database/*.pt) → agent_tools/ (FaceNet + paths).




---

PROJECT STRUCTURE

face-candy-agent/
├── agent_tools/
├── database/          # visitor embedding memory (.pt files; gitignored)
├── face_engine.py     # capture + embed + compare + store (uses agent_tools)
├── tools.py           # tool API used by the agent
├── agent.py           # run_agent() loop and logs
├── main.py            # entry: run_agent()
├── known_faces/
├── served.json
├── requirements.txt
└── README.md

---

HOW IT WORKS

1. Run the program
2. Webcam opens
3. Press SPACE to capture your face
4. System tries to match your face to known identities
5. If match found:
   - checks if you've already received candy
   - prints DISPENSED CANDY or CANDY NOT DISPENSED
6. If no match:
   - prompts for your name
   - saves your face into known_faces/
   - then treats you as a new identity

---

TOOLS OVERVIEW

Each file in agent_tools/ is one callable function:

capture_image → opens webcam and captures an image  
load_known_embeddings → loads all known faces  
recognize_identity → compares a face to known identities  
enroll_identity → saves a new identity  
load_served → loads memory from served.json  
decide_candy → returns DISPENSE / DENY  

---

DATA

known_faces/  
Put saved faces here, one per person:

known_faces/
justin.jpg
alex.jpg

served.json  
Stores who already received candy:

["justin"]

This file is created automatically if it doesn't exist.

---

YOUR TASK (AGENTIC AI)

You are responsible for:

- calling these tools in sequence
- building the decision loop
- managing state
- optionally adding:
  - continuous loop
  - logging
  - cooldown logic (once per day)
  - better memory handling

---

NOTES

- Uses CPU for stability
- Press SPACE to capture image
- If wrong camera is used, change index in capture_image.py
- Do NOT upgrade dependency versions

---

FUTURE IMPROVEMENTS

- multiple images per identity
- better matching accuracy
- persistent logs
- Arduino integration