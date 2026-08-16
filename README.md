# Gestura

Real-time gesture recognition that maps your hand and body movements to hamster memes. Your webcam feed runs through MediaPipe's hand, face, and pose models every frame — the detected gesture picks a meme, and both are displayed stacked in one window.

Quit with `q`.

## Gestures

| Gesture | What to do | Meme |
|---------|-----------|------|
| default | No specific gesture | poker hamster |
| thumbs_up | Thumb up, other fingers curled | thumbs up |
| thumbs_down | Thumb pointing down | thumbs down |
| side_eye | Turn your head left/right (~18°+) | side eye |
| fist_by_head | Fist held beside your cheek | lollipop |
| two_hands | Both hands visible to camera | arms out |
| glasses | Pinch (thumb+index) near your face | discord mod |
| bicep | Flex one arm, elbow out, wrist above shoulder | bicep |
| cross_arms | Arms crossed at chest | crossed arms |
| finger_mouth | Index finger pointing at your mouth | shush |
| nerd | Index finger up, away from mouth | nerd |

Priority when multiple gestures could match: glasses (pinch near face) → fist/thumbs beside head → pointer (mouth vs nerd) → crossed arms/bicep (pose-based) → two hands → head turn → default.

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the three MediaPipe models and put them next to `main.py`:

```powershell
curl -L -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
curl -L -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
curl -L -o pose_landmarker.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker/float16/latest/pose_landmarker.task"
```

Run:

```powershell
python main.py
```

## Setup (Linux/macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash setup.sh
python3 main.py
```

`setup.sh` creates the venv, installs deps, and downloads the models automatically if they're missing. On macOS, grant Terminal camera access when prompted.

## How it works

Each frame: webcam → flip + BGR→RGB → three MediaPipe landmarker (hand ×2, face, pose) → `classify_gesture()` returns a gesture name → a sliding vote buffer (12 frames, needs 7 votes) stabilizes the output → meme image + camera view stacked vertically.

Hand-shape checks run before pose-based checks so a thumbs-up near your head doesn't get swallowed by the bicep detector. The debug overlay in the top-left shows raw values for every heuristic — watch those when tuning thresholds, not by guessing.

## Tuning

All thresholds are named constants at the top of `main.py`. The debug overlay shows fingers, pinch ratio, thumb dy, face/mouth distances, and bicep pose signals live.

## Adding a new gesture

1. Add an image to `images/`
2. Add an entry to the `MEMES` dict
3. Add a branch in `classify_gesture()`
4. Add a row to this README's gesture table

## Requirements

- Python 3.10+
- A webcam
- Windows, Linux, or macOS
