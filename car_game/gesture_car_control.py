#!/usr/bin/env python3
"""
================================================================================
                        GESTURE CAR CONTROL
          Webcam-Based Racing Game Controller (OpenCV + MediaPipe + pynput)
================================================================================

CONTROLS SCHEME:
  1. VIRTUAL STEERING WHEEL (Two Hands):
     - Grip an invisible steering wheel with both hands in front of the webcam.
     - Tilting your hands rotates the virtual wheel.
     - Neutral (< 10°): Center / No steering key.
     - Moderate Turn (10° - 35°): Pulsed / rapid taps for precise, responsive steering.
     - Hard Turn (>= 35°): Full continuous hold on LEFT or RIGHT arrow key.
     - Direct HUD overlay displays a rotating steering wheel reflecting your hand angle.
     - If < 2 hands are detected, steering automatically centers safely.

  2. CONTINUOUS SPEED / GAS CONTROL (Hand Vertical Motion):
     - Vertical position of the throttle hand (Right Hand by default) computes continuous
       throttle between 0.0 (bottom) and 1.0 (top).
     - Exponential moving average (EMA) smoothing eliminates frame-to-frame jitter.
     - Digital Game Mapping: Throttle > GAS_IDLE_THRESHOLD modulates/holds the UP arrow key.
     - Live HUD displays the continuous 0.0-1.0 throttle bar and current ON/OFF state.
     - Extension points included for drop-in analog gamepad axes (e.g. vgamepad).

  3. DEDICATED BRAKE GESTURE:
     - Make a 'Closed_Fist' gesture on either hand to trigger BRAKE (DOWN arrow).
     - Instantly overrides and cuts throttle for safety.
     - Displays bright 'BRAKE ON' status badge in HUD.

  4. HUD / OVERLAY:
     - Real-time racing telemetry bar (FPS, Steer Angle, Throttle Meter, Brake Status).
     - Hand skeleton landmark tracking visualization.
     - Rotating racing wheel HUD with 12-o'clock top marker and dynamic status colors.

  5. CLEAN EXIT & SAFETY:
     - Press 'q' or Esc to exit.
     - Guaranteed release of all pressed OS keys upon window close, exit, or Ctrl+C.
================================================================================
"""

import math
import os
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# OS-level keyboard emulation
from pynput.keyboard import Controller as KeyboardController, Key


# ==============================================================================
#                             CONFIG SECTION
# ==============================================================================

# --- Key Bindings ---
KEY_GAS = Key.up        # Key to accelerate (UP arrow)
KEY_BRAKE = Key.down    # Key to brake / reverse (DOWN arrow)
KEY_LEFT = Key.left     # Key to steer left (LEFT arrow)
KEY_RIGHT = Key.right   # Key to steer right (RIGHT arrow)

# --- Virtual Steering Wheel Settings ---
# Deadzone: angles below this (in degrees) are treated as straight / center
STEER_DEADZONE_DEG: float = 8.0
# Max angle: angles at or beyond this (in degrees) trigger 100% full key hold
STEER_MAX_DEG: float = 30.0
# Steering smoothing factor (Exponential Moving Average, 0.0 = frozen, 1.0 = instant)
STEER_SMOOTHING_ALPHA: float = 0.35
# Pulse cycle period (in milliseconds) for moderate turn tapping
STEER_TAP_INTERVAL_MS: float = 120.0
# Minimum active pulse duration (in milliseconds) during a moderate tap
STEER_MIN_PULSE_MS: float = 30.0
# Radius of the HUD steering wheel graphic (in pixels)
STEER_WHEEL_RADIUS: int = 90

# --- Continuous Throttle / Gas Settings (Proportional Speed Modulation) ---
# Below this throttle, gas is completely OFF (Coast / Idle)
GAS_IDLE_THRESHOLD: float = 0.18
# Above this throttle, gas is 100% full continuous hold
GAS_FULL_THRESHOLD: float = 0.75
# Pulse period (in ms) for proportional throttle modulation in keyboard games
GAS_PULSE_INTERVAL_MS: float = 140.0
# Hand height calibration (normalized Y: 0.0 = top of camera, 1.0 = bottom)
GAS_Y_TOP: float = 0.20       # Hand raised high -> 100% throttle
GAS_Y_BOTTOM: float = 0.60    # Hand at normal wheel rest height -> 0% throttle
# Throttle smoothing factor (EMA, 0.0 to 1.0)
GAS_SMOOTHING_ALPHA: float = 0.22
# Which hand to use for throttle: "RIGHT", "LEFT", "HIGHEST", or "AVERAGE"
GAS_HAND_PREFERENCE: str = "RIGHT"

# --- Start / Ignition Gesture Settings ---
# Gesture to start the game / ignition (Victory / Peace sign ✌️)
START_GESTURES: List[str] = ["Victory"]
START_MIN_CONFIDENCE: float = 0.60
KEY_START_GAME = Key.space  # Key sent to start the browser game (Space / Enter)
AUTO_RUN_ON_START: bool = False  # If False, user controls speed naturally via hand height

# --- Brake Gesture Settings ---
# MediaPipe recognized gesture names that trigger brake
BRAKE_GESTURES: List[str] = ["Closed_Fist"]
# Minimum gesture confidence score (0.0 to 1.0)
BRAKE_MIN_CONFIDENCE: float = 0.60
# If True, activating brake automatically releases gas key
BRAKE_OVERRIDE_GAS: bool = True

# --- Camera & Video Settings ---
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
# Flip camera horizontally so physical left hand appears on the left side of screen
MIRROR_VIEW: bool = True

# --- Model Auto-Download Settings ---
MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
)
MODEL_FILENAME: str = "gesture_recognizer.task"

# --- Visual HUD Colors (BGR) ---
COLOR_BG_PANEL = (20, 20, 26)       # Dark slate panel
COLOR_ACCENT_CYAN = (255, 215, 0)   # Neon Cyan/Sky
COLOR_GREEN = (60, 220, 70)         # Racing Green (Active Gas / Good)
COLOR_RED = (50, 50, 235)           # Racing Red (Brake / Warning)
COLOR_AMBER = (0, 165, 255)         # Warning Amber (Tapping / Moderate Steer)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (110, 110, 120)
COLOR_DARK_GRAY = (45, 45, 55)


# ==============================================================================
#                       MODEL AUTO-DOWNLOAD & SETUP
# ==============================================================================

def ensure_model_file(model_path: str = MODEL_FILENAME, url: str = MODEL_URL) -> str:
    """
    Checks if the MediaPipe gesture recognizer task model exists locally.
    If not, downloads it automatically from Google Cloud Storage.
    """
    if os.path.exists(model_path):
        return model_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_rel_path = os.path.join(script_dir, model_path)
    if os.path.exists(script_rel_path):
        return script_rel_path

    print(f"[Setup] MediaPipe Gesture Recognizer model not found locally.")
    print(f"[Setup] Downloading from: {url} ...")

    def _progress_hook(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_tot = total_size / (1024 * 1024)
            sys.stdout.write(f"\r[Setup] Downloading: {percent:.1f}% ({mb_down:.1f}/{mb_tot:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, model_path, reporthook=_progress_hook)
        print(f"\n[Setup] Successfully downloaded model to: {model_path}")
    except Exception as e:
        print(f"\n[Error] Failed to download model: {e}")
        print("[Error] Please verify your internet connection or download gesture_recognizer.task manually.")
        raise e

    return model_path


# ==============================================================================
#                            KEY STATE CONTROLLER
# ==============================================================================

class KeyState:
    """
    Manages OS-level keyboard states using pynput.
    Ensures that press() and release() are ONLY invoked on actual state transitions,
    avoiding redundant OS key repeat events and high CPU load.
    """

    def __init__(self):
        self.keyboard = KeyboardController()
        # Internal state dictionary: Key -> bool (True = Pressed, False = Released)
        self._states: Dict[Key, bool] = {
            KEY_GAS: False,
            KEY_BRAKE: False,
            KEY_LEFT: False,
            KEY_RIGHT: False,
            KEY_START_GAME: False,
        }

    def tap_key(self, key: Key) -> None:
        """Briefly taps a key (press then release) for one-off actions like starting the game."""
        try:
            self.keyboard.press(key)
            time.sleep(0.04)
            self.keyboard.release(key)
        except Exception as e:
            print(f"[KeyState] Error tapping {key}: {e}")

    def set_state(self, key: Key, should_press: bool) -> None:
        """
        Updates the key state. Calls press/release ONLY on actual transition.
        """
        is_pressed = self._states.get(key, False)
        if should_press and not is_pressed:
            try:
                self.keyboard.press(key)
                self._states[key] = True
            except Exception as e:
                print(f"[KeyState] Error pressing {key}: {e}")
        elif not should_press and is_pressed:
            try:
                self.keyboard.release(key)
                self._states[key] = False
            except Exception as e:
                print(f"[KeyState] Error releasing {key}: {e}")

    def get_state(self, key: Key) -> bool:
        """Returns the current tracked state of the specified key."""
        return self._states.get(key, False)

    def release_all(self) -> None:
        """Releases all currently held keys. Must be called upon exit/cleanup."""
        for key, is_pressed in list(self._states.items()):
            if is_pressed:
                try:
                    self.keyboard.release(key)
                except Exception as e:
                    print(f"[KeyState] Error releasing key during cleanup: {e}")
                self._states[key] = False
        print("[KeyState] All keys successfully released.")


# ==============================================================================
#                       MATH & TELEMETRY COMPUTATIONS
# ==============================================================================

class SmoothedValue:
    """Simple Exponential Moving Average (EMA) filter."""
    def __init__(self, alpha: float, initial_value: float = 0.0):
        self.alpha = max(0.01, min(1.0, alpha))
        self.value = initial_value
        self.initialized = False

    def update(self, new_val: float) -> float:
        if not self.initialized:
            self.value = new_val
            self.initialized = True
        else:
            self.value = self.alpha * new_val + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self, value: float = 0.0):
        self.value = value
        self.initialized = False


def compute_palm_center(landmarks, frame_w: int, frame_h: int) -> Tuple[int, int]:
    """
    Computes a stable palm center in pixel coordinates by averaging the Wrist (0)
    and Middle Finger MCP (9) landmarks.
    """
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    cx = int(((wrist.x + middle_mcp.x) / 2.0) * frame_w)
    cy = int(((wrist.y + middle_mcp.y) / 2.0) * frame_h)
    return cx, cy


def compute_steer_angle(left_pt: Tuple[int, int], right_pt: Tuple[int, int]) -> float:
    """
    Calculates the steering angle in degrees relative to the horizontal axis.
    Left hand is on the left side of screen (x_L < x_R).
    
    Returns:
      Angle in degrees:
        - 0.0 = Perfectly horizontal (Center)
        - Positive (> 0) = Tilting clockwise / Right turn (Right hand down, Left hand up)
        - Negative (< 0) = Tilting counter-clockwise / Left turn (Left hand down, Right hand up)
    """
    dx = right_pt[0] - left_pt[0]
    dy = right_pt[1] - left_pt[1]
    if dx <= 0:
        # Fallback if hands cross over
        dx = 1
    # dy > 0 means right hand is lower than left hand (since Y increases downward in screen coords)
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)


def compute_continuous_throttle(hand_y_norm: float, y_top: float = GAS_Y_TOP, y_bottom: float = GAS_Y_BOTTOM) -> float:
    """
    Maps normalized hand Y position (0.0=top, 1.0=bottom) to a continuous throttle [0.0, 1.0].
    Moving hand UP (smaller Y) increases throttle towards 1.0.
    Moving hand DOWN (larger Y) decreases throttle towards 0.0.
    """
    if y_bottom <= y_top:
        return 0.0
    # Higher hand = lower Y = higher throttle
    raw_val = (y_bottom - hand_y_norm) / (y_bottom - y_top)
    return max(0.0, min(1.0, raw_val))


# ==============================================================================
#                         HUD & GRAPHICS RENDERING
# ==============================================================================

# Standard MediaPipe Hand connections for skeleton drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]


def draw_hand_skeleton(frame: np.ndarray, landmarks, frame_w: int, frame_h: int, color_line=(0, 220, 255), color_joint=(255, 255, 255)):
    """Draws sleek futuristic hand skeleton and joint circles."""
    points = []
    for lm in landmarks:
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)
        points.append((px, py))

    # Draw connection bones
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1 = points[start_idx]
        pt2 = points[end_idx]
        cv2.line(frame, pt1, pt2, color_line, 2, cv2.LINE_AA)

    # Draw joints
    for idx, pt in enumerate(points):
        radius = 4 if idx in [4, 8, 12, 16, 20] else 2  # Larger fingertips
        cv2.circle(frame, pt, radius, color_joint, -1, cv2.LINE_AA)


def draw_rotating_steering_wheel(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    angle_deg: float,
    steer_mode: str,
    left_hand_pt: Optional[Tuple[int, int]] = None,
    right_hand_pt: Optional[Tuple[int, int]] = None
):
    """
    Renders a dynamic, visually rich racing steering wheel HUD overlay.
    - Outer glow ring and rim
    - Spoke crossbar rotated in real-time matching angle_deg
    - Top 12-o'clock racing center stripe
    - Center hub and angle readout
    - Connecting guide lines to player's hands
    """
    cx, cy = center

    # Dynamic rim color based on steering status
    if "FULL" in steer_mode:
        rim_color = COLOR_RED if "LEFT" in steer_mode else COLOR_GREEN
    elif "TAP" in steer_mode:
        rim_color = COLOR_AMBER
    else:
        rim_color = COLOR_ACCENT_CYAN

    # Optional: Draw subtle connecting lines from hands to wheel center
    if left_hand_pt is not None and right_hand_pt is not None:
        cv2.line(frame, left_hand_pt, (cx, cy), (60, 60, 80), 1, cv2.LINE_AA)
        cv2.line(frame, right_hand_pt, (cx, cy), (60, 60, 80), 1, cv2.LINE_AA)
        # Highlight hand grip points
        cv2.circle(frame, left_hand_pt, 7, rim_color, 2, cv2.LINE_AA)
        cv2.circle(frame, right_hand_pt, 7, rim_color, 2, cv2.LINE_AA)

    # 1. Outer Translucent / Subtle Shadow Ring
    cv2.circle(frame, (cx, cy), radius + 4, (10, 10, 15), 3, cv2.LINE_AA)

    # 2. Main Outer Steering Wheel Rim
    cv2.circle(frame, (cx, cy), radius, rim_color, 4, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius - 8, (70, 70, 85), 1, cv2.LINE_AA)

    # 3. Compute Rotated Spokes
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # Horizontal main spoke (left to right across the wheel)
    spoke_len = radius - 8
    lx = int(cx - spoke_len * cos_a)
    ly = int(cy - spoke_len * sin_a)
    rx = int(cx + spoke_len * cos_a)
    ry = int(cy + spoke_len * sin_a)
    cv2.line(frame, (lx, ly), (rx, ry), rim_color, 3, cv2.LINE_AA)

    # Lower spoke (downwards T-spoke)
    dx = int(cx - (spoke_len * 0.75) * sin_a)
    dy = int(cy + (spoke_len * 0.75) * cos_a)
    cv2.line(frame, (cx, cy), (dx, dy), (160, 160, 175), 2, cv2.LINE_AA)

    # 4. Top 12-o'clock Racing Center Marker (Rotates with wheel)
    top_x = int(cx + radius * sin_a)
    top_y = int(cy - radius * cos_a)
    cv2.circle(frame, (top_x, top_y), 6, COLOR_RED, -1, cv2.LINE_AA)
    cv2.circle(frame, (top_x, top_y), 7, COLOR_WHITE, 1, cv2.LINE_AA)

    # 5. Center Hub Dot & Emblem
    cv2.circle(frame, (cx, cy), 14, COLOR_BG_PANEL, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 14, rim_color, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 4, COLOR_WHITE, -1, cv2.LINE_AA)

    # 6. Wheel Angle Text Below Hub
    angle_str = f"{angle_deg:+.1f}°"
    text_size = cv2.getTextSize(angle_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    tx = cx - text_size[0] // 2
    ty = cy + radius + 25
    # Shadow + Text
    cv2.putText(frame, angle_str, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, angle_str, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rim_color, 2, cv2.LINE_AA)


def draw_hud_panel(
    frame: np.ndarray,
    steer_angle: float,
    steer_state: str,
    throttle_val: float,
    gas_active: bool,
    brake_active: bool,
    fps: float,
    num_hands: int,
    engine_started: bool = False,
    show_start_banner: bool = False
):
    """
    Renders the modern, high-contrast racing telemetry dashboard bar at the top of the screen.
    Includes live gauges for Throttle, Steering Angle, Brake Status, Engine State, and FPS.
    """
    h, w, _ = frame.shape
    bar_height = 82
    bar_y = 12

    # Draw semi-transparent dark background HUD bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (16, bar_y), (w - 16, bar_y + bar_height), COLOR_BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    # HUD Border
    cv2.rectangle(frame, (16, bar_y), (w - 16, bar_y + bar_height), (55, 55, 70), 1, cv2.LINE_AA)

    # Section 1: STEERING TELEMETRY
    # --------------------------------------------------
    col1_x = 35
    cv2.putText(frame, "STEER", (col1_x, bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)
    
    steer_color = COLOR_WHITE
    if "FULL" in steer_state:
        steer_color = COLOR_RED if "LEFT" in steer_state else COLOR_GREEN
    elif "TAP" in steer_state:
        steer_color = COLOR_AMBER

    steer_txt = f"{steer_angle:+.1f}°  [{steer_state}]"
    cv2.putText(frame, steer_txt, (col1_x, bar_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, steer_color, 2, cv2.LINE_AA)

    # Section 2: THROTTLE / GAS TELEMETRY & BAR
    # --------------------------------------------------
    col2_x = 330
    cv2.putText(frame, "THROTTLE / GAS", (col2_x, bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)
    
    # Throttle Progress Bar
    bar_w = 160
    bar_h = 16
    bar_top = bar_y + 40
    cv2.rectangle(frame, (col2_x, bar_top), (col2_x + bar_w, bar_top + bar_h), COLOR_DARK_GRAY, -1)
    
    fill_w = int(bar_w * max(0.0, min(1.0, throttle_val)))
    fill_color = COLOR_GREEN if gas_active else COLOR_GRAY
    if fill_w > 0:
        cv2.rectangle(frame, (col2_x, bar_top), (col2_x + fill_w, bar_top + bar_h), fill_color, -1)
    cv2.rectangle(frame, (col2_x, bar_top), (col2_x + bar_w, bar_top + bar_h), (80, 80, 95), 1, cv2.LINE_AA)

    # Gas threshold marker line on bar
    thresh_x = int(col2_x + bar_w * GAS_IDLE_THRESHOLD)
    cv2.line(frame, (thresh_x, bar_top - 2), (thresh_x, bar_top + bar_h + 2), COLOR_WHITE, 1, cv2.LINE_AA)

    # Gas numeric readout & state
    gas_label = f"{throttle_val:.2f} " + ("[GAS ON]" if gas_active else "[IDLE]")
    gas_text_color = COLOR_GREEN if gas_active else COLOR_WHITE
    cv2.putText(frame, gas_label, (col2_x + bar_w + 12, bar_top + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.52, gas_text_color, 2, cv2.LINE_AA)

    # Section 3: BRAKE & ENGINE BADGES
    # --------------------------------------------------
    col3_x = 680
    cv2.putText(frame, "SYSTEMS", (col3_x, bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)
    
    badge_w, badge_h = 110, 24
    badge_top = bar_y + 38
    badge_bg = COLOR_RED if brake_active else COLOR_DARK_GRAY
    cv2.rectangle(frame, (col3_x, badge_top), (col3_x + badge_w, badge_top + badge_h), badge_bg, -1)
    cv2.rectangle(frame, (col3_x, badge_top), (col3_x + badge_w, badge_top + badge_h), (90, 90, 100), 1, cv2.LINE_AA)
    
    badge_txt = "BRAKE ON" if brake_active else "BRAKE OFF"
    b_txt_color = COLOR_WHITE if brake_active else COLOR_GRAY
    cv2.putText(frame, badge_txt, (col3_x + 8, badge_top + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, b_txt_color, 2, cv2.LINE_AA)

    # Engine Ignition Badge
    eng_x = col3_x + badge_w + 12
    eng_w = 145
    eng_bg = (30, 120, 40) if engine_started else (35, 70, 110)
    cv2.rectangle(frame, (eng_x, badge_top), (eng_x + eng_w, badge_top + badge_h), eng_bg, -1)
    cv2.rectangle(frame, (eng_x, badge_top), (eng_x + eng_w, badge_top + badge_h), (80, 140, 90) if engine_started else (80, 100, 130), 1, cv2.LINE_AA)
    eng_txt = "ENGINE: RUN" if engine_started else "PEACE 2 START"
    eng_color = COLOR_WHITE if engine_started else COLOR_AMBER
    cv2.putText(frame, eng_txt, (eng_x + 8, badge_top + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, eng_color, 2, cv2.LINE_AA)

    # Section 4: FPS & SYSTEM STATUS
    # --------------------------------------------------
    col4_x = w - 210
    cv2.putText(frame, "STATUS", (col4_x, bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)
    fps_txt = f"FPS: {fps:4.1f} | {num_hands} HANDS"
    hands_color = COLOR_ACCENT_CYAN if num_hands >= 2 else (COLOR_AMBER if num_hands == 1 else COLOR_GRAY)
    cv2.putText(frame, fps_txt, (col4_x, bar_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, hands_color, 1, cv2.LINE_AA)

    # Floating Start Alert Banner (Displayed when Victory gesture triggers start)
    if show_start_banner:
        banner_w, banner_h = 560, 52
        bx = (w - banner_w) // 2
        by = bar_y + bar_height + 25
        b_overlay = frame.copy()
        cv2.rectangle(b_overlay, (bx, by), (bx + banner_w, by + banner_h), (20, 80, 30), -1)
        cv2.addWeighted(b_overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), COLOR_GREEN, 2, cv2.LINE_AA)
        banner_msg = "VICTORY SIGN DETECTED - GAME & CAR STARTED!"
        cv2.putText(frame, banner_msg, (bx + 18, by + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_WHITE, 2, cv2.LINE_AA)

    # Bottom Quick-Help Guide Bar
    help_overlay = frame.copy()
    cv2.rectangle(help_overlay, (0, h - 28), (w, h), (10, 10, 15), -1)
    cv2.addWeighted(help_overlay, 0.75, frame, 0.25, 0, frame)
    help_text = "[PEACE SIGN 2 FINGERS] Start Game  |  [TILT 2 HANDS] Steer  |  [RIGHT HAND UP] Gas  |  [CLOSED FIST] Brake  |  [Q] Quit"
    cv2.putText(frame, help_text, (20, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 190), 1, cv2.LINE_AA)


# ==============================================================================
#                                MAIN APPLICATION
# ==============================================================================

def main():
    print("=" * 70)
    print("            GESTURE CAR CONTROL - STARTING ENGINE")
    print("=" * 70)
    print(f"[Config] Steering Deadzone: ±{STEER_DEADZONE_DEG}° | Full Lock: ±{STEER_MAX_DEG}°")
    print(f"[Config] Gas Thresholds: Idle={GAS_IDLE_THRESHOLD:.2f}, Full={GAS_FULL_THRESHOLD:.2f} (Y: {GAS_Y_TOP:.2f} top -> {GAS_Y_BOTTOM:.2f} bottom)")
    print(f"[Config] Brake Gesture: {BRAKE_GESTURES} (Confidence >= {BRAKE_MIN_CONFIDENCE:.2f})")
    print(f"[Config] Key Mappings: UP=Gas, DOWN=Brake, LEFT=SteerLeft, RIGHT=SteerRight")
    print("-" * 70)

    # 1. Ensure Model is downloaded
    model_path = ensure_model_file()

    # 2. Initialize MediaPipe Gesture Recognizer Task
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # 3. Initialize Key State Manager
    key_state = KeyState()

    # 4. Open Webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[Error] Could not open webcam at index {CAMERA_INDEX}.")
        print("[Error] Please ensure your webcam is plugged in and not in use by another app.")
        return

    # Telemetry and Filter Smoothers
    steer_smoother = SmoothedValue(alpha=STEER_SMOOTHING_ALPHA, initial_value=0.0)
    throttle_smoother = SmoothedValue(alpha=GAS_SMOOTHING_ALPHA, initial_value=0.0)

    # Engine & Game Start State
    engine_started: bool = False
    start_banner_until: float = 0.0
    last_start_trigger_time: float = 0.0

    # Performance / FPS Tracking
    prev_time = time.time()
    fps_val = 0.0

    # Tapping / Pulsing Timers for Progressive Steering & Throttle Modulation
    steer_cycle_start_time = time.time()
    gas_cycle_start_time = time.time()

    print("[Ready] Running! Focus your browser racing game window and enjoy driving!")
    print("[Ready] Flash the Victory / Peace sign ✌️ to START the game & engine!")
    print("[Ready] Press 'q' or Esc in the webcam window to exit safely.\n")

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                print("[Warning] Failed to grab frame from camera. Retrying...")
                time.sleep(0.05)
                continue

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_val = 0.9 * fps_val + 0.1 * (1.0 / dt) if fps_val > 0 else (1.0 / dt)

            # Mirror camera horizontally for intuitive steering interaction
            if MIRROR_VIEW:
                frame = cv2.flip(raw_frame, 1)
            else:
                frame = raw_frame

            frame_h, frame_w, _ = frame.shape

            # Convert BGR frame to MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Run Gesture Recognizer inference
            recognition_result = recognizer.recognize(mp_image)

            detected_hands_landmarks = recognition_result.hand_landmarks
            detected_gestures = recognition_result.gestures
            detected_handedness = recognition_result.handedness
            num_hands = len(detected_hands_landmarks)

            # ------------------------------------------------------------------
            # 1. GESTURE DETECTION (BRAKE & VICTORY / GAME START)
            # ------------------------------------------------------------------
            brake_active = False
            start_gesture_active = False

            for hand_idx, gesture_list in enumerate(detected_gestures):
                if gesture_list and len(gesture_list) > 0:
                    top_gesture = gesture_list[0]
                    # Check for Brake gesture (Closed Fist)
                    if top_gesture.category_name in BRAKE_GESTURES and top_gesture.score >= BRAKE_MIN_CONFIDENCE:
                        brake_active = True
                    # Check for Game Start gesture (Victory / Peace Sign ✌️)
                    if top_gesture.category_name in START_GESTURES and top_gesture.score >= START_MIN_CONFIDENCE:
                        start_gesture_active = True

            # Trigger Game Start when Victory / Peace Sign is flashed (with 2.5s debounce)
            if start_gesture_active and (curr_time - last_start_trigger_time > 2.5):
                print("[Ignition] ✌️ Victory / Peace sign detected! Starting game & engine...")
                key_state.tap_key(KEY_START_GAME)
                engine_started = True
                last_start_trigger_time = curr_time
                start_banner_until = curr_time + 3.0  # Display banner for 3 seconds

            # ------------------------------------------------------------------
            # 2. TWO-HAND VIRTUAL STEERING WHEEL LOGIC
            # ------------------------------------------------------------------
            raw_steer_angle = 0.0
            steer_state = "CENTER"
            left_hand_pt: Optional[Tuple[int, int]] = None
            right_hand_pt: Optional[Tuple[int, int]] = None
            wheel_center = (frame_w // 2, int(frame_h * 0.65))

            if num_hands >= 2:
                # Compute palm centers for both hands
                palm1 = compute_palm_center(detected_hands_landmarks[0], frame_w, frame_h)
                palm2 = compute_palm_center(detected_hands_landmarks[1], frame_w, frame_h)

                # Assign left vs right based on X-coordinate on screen
                if palm1[0] < palm2[0]:
                    left_hand_pt, right_hand_pt = palm1, palm2
                    left_lms, right_lms = detected_hands_landmarks[0], detected_hands_landmarks[1]
                else:
                    left_hand_pt, right_hand_pt = palm2, palm1
                    left_lms, right_lms = detected_hands_landmarks[1], detected_hands_landmarks[0]

                # Compute steering angle
                raw_steer_angle = compute_steer_angle(left_hand_pt, right_hand_pt)
                smoothed_steer_angle = steer_smoother.update(raw_steer_angle)

                # Wheel center placed between the two hands for authentic HUD alignment
                wheel_center = (
                    (left_hand_pt[0] + right_hand_pt[0]) // 2,
                    (left_hand_pt[1] + right_hand_pt[1]) // 2
                )
            else:
                # If 0 or 1 hand is visible, steering safely decays to center
                smoothed_steer_angle = steer_smoother.update(0.0)
                steer_state = "CENTER"

            # Determine Steering Key Commands
            steer_left_press = False
            steer_right_press = False
            abs_angle = abs(smoothed_steer_angle)

            if abs_angle < STEER_DEADZONE_DEG or num_hands < 2:
                steer_state = "CENTER" if num_hands >= 2 else "CENTER (NO HANDS)"
                steer_left_press = False
                steer_right_press = False
            elif abs_angle >= STEER_MAX_DEG:
                # Hard Turn: 100% continuous key hold
                if smoothed_steer_angle > 0:
                    steer_state = "RIGHT (FULL)"
                    steer_right_press = True
                else:
                    steer_state = "LEFT (FULL)"
                    steer_left_press = True
            else:
                # Moderate Turn: Pulsed Tapping proportional to angle intensity
                turn_ratio = (abs_angle - STEER_DEADZONE_DEG) / (STEER_MAX_DEG - STEER_DEADZONE_DEG)
                # Calculate active pulse time within the cycle period
                pulse_duration_ms = STEER_MIN_PULSE_MS + turn_ratio * (STEER_TAP_INTERVAL_MS - STEER_MIN_PULSE_MS)
                
                # Elapsed time within the current pulse period
                cycle_elapsed_ms = ((curr_time - steer_cycle_start_time) * 1000.0) % STEER_TAP_INTERVAL_MS
                is_pulse_on = cycle_elapsed_ms < pulse_duration_ms

                if smoothed_steer_angle > 0:
                    steer_state = "RIGHT (TAP)"
                    steer_right_press = is_pulse_on
                else:
                    steer_state = "LEFT (TAP)"
                    steer_left_press = is_pulse_on

            # Apply Steering Key States
            key_state.set_state(KEY_LEFT, steer_left_press)
            key_state.set_state(KEY_RIGHT, steer_right_press)

            # ------------------------------------------------------------------
            # 3. CONTINUOUS SPEED / GAS CONTROL (Hand Vertical Motion)
            # ------------------------------------------------------------------
            raw_throttle = 0.0
            if num_hands > 0:
                # Determine which hand to use for throttle calculation
                throttle_hand_lms = None
                if num_hands >= 2 and right_hand_pt is not None:
                    # In two-hand mode, use the right hand by default for gas
                    throttle_hand_lms = right_lms if GAS_HAND_PREFERENCE == "RIGHT" else left_lms
                else:
                    # Single hand visible: use the detected hand
                    throttle_hand_lms = detected_hands_landmarks[0]

                if throttle_hand_lms is not None:
                    # Palm center vertical position (normalized Y: 0.0 top, 1.0 bottom)
                    wrist_y = throttle_hand_lms[0].y
                    middle_y = throttle_hand_lms[9].y
                    palm_y_norm = (wrist_y + middle_y) / 2.0
                    raw_throttle = compute_continuous_throttle(palm_y_norm, GAS_Y_TOP, GAS_Y_BOTTOM)

            # Smooth throttle value with Exponential Moving Average (EMA)
            smoothed_throttle = throttle_smoother.update(raw_throttle)

            # PROPORTIONAL THROTTLE MODULATION (PWM Duty-Cycle for Keyboard)
            # Prevents car from accelerating uncontrollably to max speed!
            gas_press = False
            if brake_active or smoothed_throttle < GAS_IDLE_THRESHOLD:
                gas_press = False
            elif smoothed_throttle >= GAS_FULL_THRESHOLD:
                # Full throttle: continuous hold (100% power)
                gas_press = True
            else:
                # Moderate throttle: pulse UP arrow in cycles proportional to height
                duty_ratio = (smoothed_throttle - GAS_IDLE_THRESHOLD) / (GAS_FULL_THRESHOLD - GAS_IDLE_THRESHOLD)
                pulse_on_ms = max(25.0, duty_ratio * GAS_PULSE_INTERVAL_MS)
                gas_cycle_ms = ((curr_time - gas_cycle_start_time) * 1000.0) % GAS_PULSE_INTERVAL_MS
                gas_press = (gas_cycle_ms < pulse_on_ms)

            # Active gas indicator for HUD
            gas_active = (smoothed_throttle >= GAS_IDLE_THRESHOLD) and (not brake_active)

            # Apply Gas & Brake Key States
            if brake_active:
                key_state.set_state(KEY_BRAKE, True)
                if BRAKE_OVERRIDE_GAS:
                    key_state.set_state(KEY_GAS, False)
            else:
                key_state.set_state(KEY_BRAKE, False)
                key_state.set_state(KEY_GAS, gas_press)

            # ------------------------------------------------------------------
            # 4. RENDER OVERLAY & HUD
            # ------------------------------------------------------------------
            # Draw skeletons for all visible hands
            for hand_lms in detected_hands_landmarks:
                draw_hand_skeleton(frame, hand_lms, frame_w, frame_h)

            # Draw rotating steering wheel HUD
            draw_rotating_steering_wheel(
                frame=frame,
                center=wheel_center,
                radius=STEER_WHEEL_RADIUS,
                angle_deg=smoothed_steer_angle,
                steer_mode=steer_state,
                left_hand_pt=left_hand_pt,
                right_hand_pt=right_hand_pt
            )

            # Draw telemetry dashboard panel
            show_banner = (curr_time < start_banner_until)
            draw_hud_panel(
                frame=frame,
                steer_angle=smoothed_steer_angle,
                steer_state=steer_state,
                throttle_val=smoothed_throttle,
                gas_active=gas_active,
                brake_active=brake_active,
                fps=fps_val,
                num_hands=num_hands,
                engine_started=engine_started,
                show_start_banner=show_banner
            )

            # Display frame in OpenCV window
            cv2.imshow("Gesture Car Control - Racing HUD", frame)

            # Process window events & check for quit ('q' or Esc)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("\n[Exit] User requested exit ('q' or Esc). Shutting down...")
                break

            # Check if user closed window with 'X' button
            if cv2.getWindowProperty("Gesture Car Control - Racing HUD", cv2.WND_PROP_VISIBLE) < 1:
                print("\n[Exit] Window closed by user. Shutting down...")
                break

    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user (Ctrl+C). Shutting down...")
    except Exception as e:
        print(f"\n[Exception] Unexpected error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # CLEANUP: Guarantee that all OS-level keys are released safely!
        print("[Cleanup] Releasing resources and resetting keyboard...")
        key_state.release_all()
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        if 'recognizer' in locals():
            recognizer.close()
        print("[Cleanup] Shutdown complete. Drive safely!")


if __name__ == "__main__":
    main()
