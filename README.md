# ✋ Gesture Controller — PC Control via Hand Gestures

Control your entire PC hands-free using a webcam and hand gestures powered by **MediaPipe** and **PyAutoGUI**.

---

## 🚀 Quick Start

```bash
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run the controller
python main.py
```

> The camera window will open. Show your hand to the webcam and hold a gesture steady for ~0.5 seconds to activate it.
> Press **`q`** in the camera window to quit.

---

## 🖐️ Gesture Reference Table

| # | Gesture | Hand Shape | Action |
|---|---------|-----------|--------|
| 1 | **Cursor Control** | Index finger only (no thumb) | Moves the mouse cursor — your fingertip maps directly to the screen |
| 2 | **Left Click** | Thumb pinches Index tip | Single left click |
| 3 | **Right Click** | Thumb pinches Middle tip (index tucked) | Single right click |
| 4 | **Volume Up** | Middle finger only | Continuously increases system volume while held |
| 5 | **Volume Down** | Index + Pinky (spread apart) | Continuously decreases system volume while held |
| 6 | **Mute / Unmute** | Pinky finger only | Toggles system mute — fires once per gesture entry |
| 7 | **Zoom In** | Thumb + Index + Middle — spread fingers apart | Sends Ctrl + + to zoom in |
| 8 | **Zoom Out** | Thumb + Index + Middle — pinch fingers together | Sends Ctrl + - to zoom out |
| 9 | **Scroll Up / Down** | Index + Middle (no thumb) — slide hand up/down | Scrolls the active window |
| 10 | **Switch Browser Tab (Next)** | 4 fingers (no thumb) — swipe RIGHT | Ctrl + Tab — next tab in Brave/Chrome |
| 11 | **Switch Browser Tab (Prev)** | 4 fingers (no thumb) — swipe LEFT | Ctrl + Shift + Tab — previous tab |
| 12 | **Play / Pause** | Open palm (all 5 fingers) — hold still | Presses Spacebar — pauses/plays media |
| 13 | **Switch App (Next)** | Open palm — slide RIGHT | Alt + Tab |
| 14 | **Switch App (Prev)** | Open palm — slide LEFT | Alt + Shift + Tab |
| 15 | **Show / Restore Desktop** | Thumb only (all fingers curled) | Win + D — minimizes all windows or restores them |
| 16 | **Screenshot** | Fist (all fingers closed) — hold for 1 second | Win + PrintScreen — saves to Pictures > Screenshots |

---

## Tips

- **Hold for 0.5 seconds** — Every gesture (except cursor & clicks) requires you to hold it steadily for half a second before it activates. You will see Stabilizing... then Active on screen.
- **Screenshot countdown** — A live countdown "Hold fist... 0.8s" is displayed on the camera window. Only releases after the full second.
- **Cursor is instant** — No stabilization delay. The cursor follows your index fingertip in real time.
- **Volume is continuous** — Keep holding Volume Up/Down to keep adjusting. Let go to stop.
- **Mute fires once** — Pinky gesture toggles mute exactly once per entry. Lower and re-raise your pinky to toggle again.
- **Tab switching** — Make sure Brave/Chrome is in the background (do NOT click on the camera window). Hold 4 fingers firmly, then swipe 35+ pixels.
- **Zoom** — Hold Thumb+Index+Middle all up, then slowly spread your fingers apart (zoom in) or pinch them together (zoom out). Watch the d=XX diff=XX on screen.

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.8+ | Webcam capture and display |
| mediapipe | 0.10.35 | Hand landmark detection |
| pyautogui | 0.9+ | Keyboard and mouse control |
| screeninfo | 0.8+ | Detect screen resolution for cursor mapping |

Install all:
```bash
pip install -r requirements.txt
```

---

## Configuration

Open main.py and adjust these constants near the top:

| Constant | Default | Description |
|----------|---------|-------------|
| PINCH_THRESHOLD | 45 px | How close thumb must be to trigger a click |
| CURSOR_SMOOTH | 0.35 | Cursor smoothing (0=raw/jittery, 1=instant) |
| CURSOR_DEADZONE | 4 px | Ignore micro-jitter below this distance |

---

## Project Structure

```
operate through gesture/
├── main.py                  Core gesture controller
├── requirements.txt         Python dependencies
├── hand_landmarker.task     MediaPipe model (auto-downloaded on first run)
├── venv/                    Virtual environment
└── README.md                This file
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| mediapipe import error | Run inside venv: .\venv\Scripts\Activate.ps1 |
| Model file missing | Delete hand_landmarker.task and rerun — it re-downloads automatically |
| Cursor jumps erratically | Improve lighting; keep hand within the camera frame |
| Tab switching not working | Do NOT click the camera window. Brave must hold keyboard focus |
| Volume keys not working | Check that pyautogui volumeup/volumedown works on your system |
| Screenshot not saving | Check Pictures > Screenshots folder; ensure Win+PrintScreen works manually |
