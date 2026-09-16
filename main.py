import cv2
import mediapipe as mp
import pyautogui
import time
import math
import urllib.request
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self):
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading Hand Landmarker model... this may take a moment.")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.results = None
        
    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self.results = self.landmarker.detect(mp_image)
        
        if self.results.hand_landmarks and draw:
            for hand_landmarks in self.results.hand_landmarks:
                for lm in hand_landmarks:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        return img
        
    def get_position(self, img):
        lmlist = []
        if self.results and self.results.hand_landmarks:
            hand_landmarks = self.results.hand_landmarks[0]
            for id, lm in enumerate(hand_landmarks):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append((id, cx, cy))
        return lmlist

    def fingers_up(self, lmlist):
        if not lmlist:
            return []
        fingers = []
        # Robust orientation-independent finger detection using distance from wrist
        
        # Thumb: compare distance from thumb tip to pinky MCP vs thumb IP to pinky MCP
        dist_thumb_tip = math.hypot(lmlist[4][1] - lmlist[17][1], lmlist[4][2] - lmlist[17][2])
        dist_thumb_ip = math.hypot(lmlist[3][1] - lmlist[17][1], lmlist[3][2] - lmlist[17][2])
        fingers.append(1 if dist_thumb_tip > dist_thumb_ip else 0)
            
        # 4 Fingers
        for tip in [8, 12, 16, 20]:
            dist_tip = math.hypot(lmlist[tip][1] - lmlist[0][1], lmlist[tip][2] - lmlist[0][2])
            dist_pip = math.hypot(lmlist[tip-2][1] - lmlist[0][1], lmlist[tip-2][2] - lmlist[0][2])
            fingers.append(1 if dist_tip > dist_pip else 0)
        return fingers


import screeninfo

import ctypes

# Disable pyautogui failsafe so cursor can reach corners
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0   # No artificial delay

# ── Screen resolution for cursor mapping ──────────────────────────────
try:
    _mon = screeninfo.get_monitors()[0]
    SCREEN_W, SCREEN_H = _mon.width, _mon.height
except Exception:
    SCREEN_W, SCREEN_H = pyautogui.size()

PINCH_THRESHOLD  = 45   # pixels in camera frame for pinch detection
CURSOR_SMOOTH    = 0.35 # EMA alpha: higher = more responsive (0-1)
CURSOR_DEADZONE  = 4    # pixels: ignore tiny jitter below this


class GestureController:
    def __init__(self):
        self.stable_gesture    = None
        self.gesture_start_time = 0
        self.last_action_time  = 0
        self.current_mode      = None
        self.palm_action_done  = False
        self.mode_start_x      = None
        self.mode_start_y      = None
        self.mode_start_dist   = None
        # Scroll smoothing
        self.prev_scroll_y     = None
        self.scroll_velocity   = 0.0
        # Cursor smoothing
        self.cursor_x        = None
        self.cursor_y        = None
        # Click cooldown
        self.last_click_time = 0
        # Zoom gets its own timer so shared last_action_time doesn't block it
        self.last_zoom_time  = 0

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------
    def _dist(self, lmlist, a, b):
        return math.hypot(lmlist[a][1] - lmlist[b][1],
                          lmlist[a][2] - lmlist[b][2])

    # ------------------------------------------------------------------
    # Gesture classifier
    # ------------------------------------------------------------------
    def _classify(self, thumb, idx, mid, rng, pnk, lmlist):
        """
        Priority order (most specific first):
          pinch gestures  →  volume/mute  →  cursor  →  scroll  →
          tab  →  zoom  →  desktop  →  palm  →  screenshot
        """
        # -- Pinch distances --
        d_thumb_idx = self._dist(lmlist, 4, 8)   # thumb ↔ index tip
        d_thumb_mid = self._dist(lmlist, 4, 12)  # thumb ↔ middle tip

        # ── PINCH: LEFT CLICK  (thumb+index close, middle/ring/pinky down) ──
        if thumb == 1 and idx == 1 and mid == 0 and rng == 0 and pnk == 0:
            if d_thumb_idx < PINCH_THRESHOLD:
                return "left_click"
            # Not pinching → cursor mode (index + thumb spread open)
            return "cursor"

        # ── PINCH: RIGHT CLICK  (thumb+middle close, index down) ──
        if thumb == 1 and idx == 0 and mid == 1 and rng == 0 and pnk == 0:
            if d_thumb_mid < PINCH_THRESHOLD:
                return "right_click"
            return "none"

        # ── ZOOM  (thumb + index + middle all up, pinky tucked) ──
        if pnk == 0 and (
            (thumb == 1 and idx == 1 and mid == 1) or
            (self.current_mode == "zoom" and (thumb + idx + mid) >= 2 and d_thumb_idx < 60)
        ):
            return "zoom"

        # ── CURSOR  (index only, no thumb) ──
        if thumb == 0 and idx == 1 and mid == 0 and rng == 0 and pnk == 0:
            return "cursor"

        # ── VOLUME UP  (middle only) ──
        if thumb == 0 and idx == 0 and mid == 1 and rng == 0 and pnk == 0:
            return "vol_up"

        # ── VOLUME DOWN  (index + pinky) ──
        if thumb == 0 and idx == 1 and mid == 0 and rng == 0 and pnk == 1:
            return "vol_down"

        # ── MUTE  (pinky only) ──
        if thumb == 0 and idx == 0 and mid == 0 and rng == 0 and pnk == 1:
            return "mute"

        # ── SCROLL  (index + middle, no thumb) ──
        if (thumb == 0 and idx == 1 and mid == 1 and rng == 0 and pnk == 0) or \
           (self.current_mode == "scroll" and idx == 1 and mid == 1 and pnk == 0):
            return "scroll"

        # ── TAB SWITCH  (index + middle + ring + pinky, no thumb) ──
        if thumb == 0 and idx == 1 and mid == 1 and rng == 1 and pnk == 1:
            return "tab"

        # ── DESKTOP  (thumb only) ──
        if thumb == 1 and (idx + mid + rng + pnk) == 0:
            return "desktop"

        # ── PALM  (all 5 fingers) ──
        if thumb == 1 and (idx + mid + rng + pnk) >= 3:
            return "palm"

        # ── SCREENSHOT  (fist) ──
        if thumb == 0 and (idx + mid + rng + pnk) == 0:
            return "screenshot"

        return "none"

    # ------------------------------------------------------------------
    # Main per-frame processing
    # ------------------------------------------------------------------
    def process_frame(self, img, lmlist, fingers, frame_w, frame_h):
        if not lmlist:
            self.stable_gesture = None
            self.current_mode   = None
            self.cursor_x = self.cursor_y = None
            self.prev_scroll_y   = None
            self.scroll_velocity = 0.0
            return

        thumb = fingers[0]
        idx   = fingers[1]
        mid   = fingers[2]
        rng   = fingers[3]
        pnk   = fingers[4]

        wrist_x = lmlist[0][1]
        wrist_y = lmlist[0][2]
        now     = time.time()

        gesture = self._classify(thumb, idx, mid, rng, pnk, lmlist)

        # Reset timer on gesture change
        if gesture != self.stable_gesture:
            self.stable_gesture    = gesture
            self.gesture_start_time = now
            self.current_mode      = None
            self.palm_action_done  = False

        held = now - self.gesture_start_time

        # ── On-screen HUD ─────────────────────────────────────────────
        cv2.putText(img, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)

        # ── SCREENSHOT: special 2-second countdown ────────────────────
        if gesture == "screenshot":
            remaining = max(0.0, 1.0 - held)
            col = (0, 165, 255) if remaining > 0 else (0, 255, 0)
            cv2.putText(img, f"Hold fist... {remaining:.1f}s", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            if held >= 1.0 and self.current_mode != "screenshot":
                self.current_mode = "screenshot"
                pyautogui.hotkey('win', 'printscreen')
                print("Action: Screenshot")
            return

        # ── CURSOR & CLICK: no debounce needed (instant response) ─────
        if gesture in ("cursor", "left_click", "right_click"):
            # Map index fingertip (landmark 8) to screen coords
            fx, fy = lmlist[8][1], lmlist[8][2]
            # Add margin so you don't need to reach the very edge
            MARGIN = 80
            mapped_x = int(
                (fx - MARGIN) / max(frame_w - 2 * MARGIN, 1) * SCREEN_W)
            mapped_y = int(
                (fy - MARGIN) / max(frame_h - 2 * MARGIN, 1) * SCREEN_H)
            mapped_x = max(0, min(SCREEN_W - 1, mapped_x))
            mapped_y = max(0, min(SCREEN_H - 1, mapped_y))

            # Hybrid smooth: jump instantly for large moves, EMA for jitter
            if self.cursor_x is None:
                self.cursor_x, self.cursor_y = float(mapped_x), float(mapped_y)
            else:
                dx = mapped_x - self.cursor_x
                dy = mapped_y - self.cursor_y
                dist = math.hypot(dx, dy)
                if dist < CURSOR_DEADZONE:
                    pass  # ignore sub-pixel jitter
                elif dist > 120:  # large jump → snap directly
                    self.cursor_x, self.cursor_y = float(mapped_x), float(mapped_y)
                else:  # smooth interpolation
                    self.cursor_x += dx * CURSOR_SMOOTH
                    self.cursor_y += dy * CURSOR_SMOOTH

            pyautogui.moveTo(int(self.cursor_x), int(self.cursor_y))

            if gesture == "cursor":
                cv2.putText(img, "Cursor Mode", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                # Draw cursor dot on cam feed
                cv2.circle(img, (fx, fy), 8, (0, 255, 255), cv2.FILLED)

            elif gesture == "left_click":
                cv2.putText(img, "LEFT CLICK", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if now - self.last_click_time > 0.6:
                    pyautogui.click()
                    self.last_click_time = now
                    print("Action: Left Click")

            elif gesture == "right_click":
                cv2.putText(img, "RIGHT CLICK", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if now - self.last_click_time > 0.6:
                    pyautogui.rightClick()
                    self.last_click_time = now
                    print("Action: Right Click")
            return

        # Reset cursor smoothing when not in cursor mode
        if gesture not in ("cursor", "left_click", "right_click"):
            self.cursor_x = self.cursor_y = None

        # Reset scroll tracking when not in scroll mode
        if gesture != "scroll":
            self.prev_scroll_y   = None
            self.scroll_velocity = 0.0

        # ── Stabilization: scroll only needs 0.15s to feel instant and responsive ──
        stabilize_time = 0.15 if gesture == "scroll" else 0.5
        if held < stabilize_time:
            cv2.putText(img, "Stabilizing...", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 255), 2)
            return

        cv2.putText(img, "Active", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # ================================================================
        # PALM — still → Play/Pause | slide → Alt+Tab
        # ================================================================
        if gesture == "palm":
            if self.current_mode != "palm":
                self.current_mode = "palm"
                self.mode_start_x = wrist_x
                self.mode_start_y = wrist_y

            x_diff = wrist_x - self.mode_start_x
            y_diff = wrist_y - self.mode_start_y

            if abs(x_diff) > 80 and now - self.last_action_time > 1.0:
                if x_diff > 0:
                    pyautogui.hotkey('alt', 'tab')
                    cv2.putText(img, "Alt+Tab >>>", (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print("Action: Alt+Tab")
                else:
                    pyautogui.hotkey('alt', 'shift', 'tab')
                    cv2.putText(img, "<<< Alt+Shift+Tab", (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print("Action: Alt+Shift+Tab")
                self.mode_start_x    = wrist_x
                self.last_action_time = now
                self.palm_action_done = True

            elif not self.palm_action_done and abs(x_diff) < 40 and abs(y_diff) < 40:
                if now - self.last_action_time > 2.0:
                    pyautogui.press('space')
                    self.last_action_time = now
                    self.palm_action_done = True
                    cv2.putText(img, "Play/Pause", (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                    print("Action: Play/Pause")

        # ================================================================
        # VOLUME UP — middle finger only (continuous)
        # ================================================================
        elif gesture == "vol_up":
            if now - self.last_action_time > 0.08:
                pyautogui.press('volumeup')
                self.last_action_time = now
                cv2.putText(img, "Volume UP ▲", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 3)

        # ================================================================
        # VOLUME DOWN — index + pinky (continuous)
        # ================================================================
        elif gesture == "vol_down":
            if now - self.last_action_time > 0.08:
                pyautogui.press('volumedown')
                self.last_action_time = now
                cv2.putText(img, "Volume DOWN ▼", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 3)

        # ================================================================
        # MUTE — pinky only (fires once per entry)
        # ================================================================
        elif gesture == "mute":
            if self.current_mode != "mute":
                self.current_mode = "mute"
                pyautogui.press('volumemute')
                cv2.putText(img, "MUTE Toggle", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                print("Action: Mute")

        # ================================================================
        # TAB SWITCH — 4 fingers (no thumb), swipe left/right
        # ================================================================
        elif gesture == "tab":
            if self.current_mode != "tab":
                self.current_mode = "tab"
                self.mode_start_x = wrist_x
                print(f"Tab mode entered, baseline x={wrist_x}")

            x_diff = wrist_x - self.mode_start_x
            cv2.putText(img, f"swipe: {int(x_diff)}px (need 35)", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            if abs(x_diff) > 35 and now - self.last_action_time > 0.8:
                # Give focus back to whichever app was previously active
                # by briefly using Alt so the next key goes to the right window
                pyautogui.keyDown('ctrl')
                if x_diff > 0:
                    pyautogui.press('tab')        # Ctrl+Tab = next tab
                    cv2.putText(img, "Next Tab >>>", (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print("Action: Next Tab")
                else:
                    pyautogui.hotkey('shift', 'tab')  # Ctrl+Shift+Tab = prev tab
                    cv2.putText(img, "<<< Prev Tab", (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print("Action: Prev Tab")
                pyautogui.keyUp('ctrl')
                self.mode_start_x    = wrist_x
                self.last_action_time = now

        # ================================================================
        # ZOOM — thumb + index + middle spread/pinch → Ctrl+= / Ctrl+-
        # ================================================================
        elif gesture == "zoom":
            d_thumb_idx = self._dist(lmlist, 4, 8)
            d_thumb_mid = self._dist(lmlist, 4, 12)
            d_idx_mid   = self._dist(lmlist, 8, 12)
            zoom_dist   = (d_thumb_idx + d_thumb_mid + d_idx_mid) / 3.0

            if self.current_mode != "zoom" or self.mode_start_dist is None:
                self.current_mode    = "zoom"
                self.mode_start_dist = zoom_dist

            dist_diff = zoom_dist - self.mode_start_dist

            # Draw visual feedback triangle between the 3 zoom fingers
            p4  = (lmlist[4][1], lmlist[4][2])
            p8  = (lmlist[8][1], lmlist[8][2])
            p12 = (lmlist[12][1], lmlist[12][2])
            cv2.line(img, p4, p8, (255, 0, 255), 2)
            cv2.line(img, p8, p12, (255, 0, 255), 2)
            cv2.line(img, p4, p12, (255, 0, 255), 2)

            # Show live distance on screen for debugging
            cv2.putText(img, f"Zoom d={int(zoom_dist)} diff={int(dist_diff)}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            # Zoom In: fingers spreading apart (Ctrl + =)
            if (dist_diff > 20 or zoom_dist > 85) and (now - self.last_zoom_time > 0.22):
                pyautogui.hotkey('ctrl', '=')
                cv2.putText(img, "Zoom In  Ctrl++", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                print("Action: Zoom In")
                self.last_zoom_time = now

            # Zoom Out: fingers pinching together (Ctrl + -)
            elif (dist_diff < -15 or zoom_dist < 45) and (now - self.last_zoom_time > 0.22):
                pyautogui.hotkey('ctrl', '-')
                cv2.putText(img, "Zoom Out Ctrl+-", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                print("Action: Zoom Out")
                self.last_zoom_time = now

            # Neutral resting zone: smoothly adapt baseline
            else:
                self.mode_start_dist = 0.9 * self.mode_start_dist + 0.1 * zoom_dist

        # ================================================================
        # SCROLL — index + middle slide up/down (continuous smooth tracking)
        # ================================================================
        elif gesture == "scroll":
            # Track the midpoint of index (8) and middle (12) fingertips
            curr_y = (lmlist[8][2] + lmlist[12][2]) / 2.0

            if self.current_mode != "scroll" or self.prev_scroll_y is None:
                self.current_mode    = "scroll"
                self.prev_scroll_y   = curr_y
                self.scroll_velocity = 0.0

            raw_dy = curr_y - self.prev_scroll_y
            self.prev_scroll_y = curr_y

            # Smooth velocity with exponential moving average to eliminate jitter
            self.scroll_velocity = 0.6 * self.scroll_velocity + 0.4 * raw_dy

            # Visual feedback on camera feed: line & center dot between fingertips
            p8  = (lmlist[8][1], lmlist[8][2])
            p12 = (lmlist[12][1], lmlist[12][2])
            mid_pt = (int((p8[0] + p12[0]) / 2), int(curr_y))
            cv2.line(img, p8, p12, (0, 255, 255), 2)
            cv2.circle(img, mid_pt, 6, (0, 255, 255), cv2.FILLED)

            # Scroll continuously when finger motion exceeds micro-jitter threshold
            if abs(self.scroll_velocity) >= 0.7:
                # Upward finger motion (negative dy) scrolls up (positive clicks)
                scroll_delta = -int(self.scroll_velocity * 16)
                if scroll_delta != 0:
                    pyautogui.scroll(scroll_delta)
                    direction = "UP ▲" if scroll_delta > 0 else "DOWN ▼"
                    cv2.putText(img, f"Scrolling {direction}", (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # ================================================================
        # DESKTOP — thumb only → Win+D
        # ================================================================
        elif gesture == "desktop":
            if self.current_mode != "desktop":
                self.current_mode = "desktop"
                if now - self.last_action_time > 1.0:
                    pyautogui.hotkey('win', 'd')
                    self.last_action_time = now
                    cv2.putText(img, "Show Desktop", (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    print("Action: Show Desktop / Restore")


def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame_h, frame_w = frame.shape[:2] if ret else (480, 640)

    tracker    = HandTracker()
    controller = GestureController()

    print("Starting Gesture Control...")
    print("Press 'q' in the camera window to quit.\n")
    print("Gesture Map:")
    print("  Index only              → Cursor control")
    print("  Thumb + Index (pinch)   → Left Click")
    print("  Thumb + Middle (pinch)  → Right Click")
    print("  Middle only             → Volume Up")
    print("  Index + Pinky           → Volume Down")
    print("  Pinky only              → Mute toggle")
    print("  Thumb + Index + Middle  → Zoom (Ctrl++ / Ctrl+-)")
    print("  2 Fingers (slide)       → Scroll")
    print("  4 Fingers (slide L/R)   → Prev/Next browser tab")
    print("  Open Palm (still)       → Play/Pause")
    print("  Open Palm (slide L/R)   → Alt+Tab")
    print("  Thumb only              → Show/Restore Desktop")
    print("  Fist (hold 2s)          → Screenshot")

    # ── Create camera window that NEVER steals focus ──────────────────
    WIN_NAME = "Gesture Controller"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 480, 360)
    # Set WS_EX_NOACTIVATE so clicks on the window don't steal keyboard focus
    try:
        hwnd = ctypes.windll.user32.FindWindowW(None, WIN_NAME)
        if hwnd:
            GWL_EXSTYLE   = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_APPWINDOW  = 0x00040000
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE,
                (style | WS_EX_NOACTIVATE) & ~WS_EX_APPWINDOW)
    except Exception as e:
        print(f"[warn] Could not set NOACTIVATE: {e}")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture video.")
            break

        img = cv2.flip(img, 1)

        img    = tracker.find_hands(img)
        lmlist = tracker.get_position(img)

        if lmlist:
            fingers = tracker.fingers_up(lmlist)
            controller.process_frame(img, lmlist, fingers, frame_w, frame_h)

        cv2.imshow("Gesture Controller", img)

        # Keep camera window always visible but DON'T let it steal focus
        # Press 'q' to quit; spacebar toggles focus away to last window
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
