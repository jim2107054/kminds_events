#!/usr/bin/env python3
"""
================================================================================
                    AI ROCK, PAPER, SCISSORS SHOWDOWN
       Webcam Gesture Recognition Game (OpenCV + MediaPipe Tasks AI)
================================================================================

HOW TO PLAY:
  1. Stand in front of your webcam.
  2. Press SPACE or show 👍 (Thumbs Up) to start a round countdown (3... 2... 1... SHOOT!).
  3. When "SHOOT!" appears, hold up your move:
       ✊ ROCK     : Closed Fist
       ✋ PAPER    : Open Palm
       ✌️ SCISSORS : Victory / Peace Sign (2 fingers)
  4. The AI will reveal its choice simultaneously.
  5. Scores, streaks, and match round history are tracked in real-time on the HUD!

GAME MODES:
  - Mode 1: Player vs AI Bot (3 Difficulty settings: Normal, Smart, Chaos)
  - Mode 2: 2-Player Local (Both players stand in front of webcam)

KEYBOARD SHORTCUTS:
  - [SPACE / ENTER] : Start Round / Next Round
  - [R]             : Reset Scores
  - [M]             : Switch Mode (Player vs AI <-> 2-Player)
  - [D]             : Cycle Difficulty (Normal -> Smart -> Chaos)
  - [Q / ESC]       : Quit Game
================================================================================
"""

import math
import os
import random
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe Tasks
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==============================================================================
#                             CONFIG SECTION
# ==============================================================================

# Camera Settings
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
MIRROR_VIEW: bool = True

# Countdown & Round Timings (Seconds)
COUNTDOWN_SECONDS: float = 3.0       # 3-second countdown before shooting
RESULT_DISPLAY_SECONDS: float = 3.0  # Time to display round winner
MATCH_WINNING_SCORE: int = 5         # First to 5 points wins the match

# Gesture Classification Settings
CONFIDENCE_THRESHOLD: float = 0.60
GESTURE_MAP: Dict[str, str] = {
    "Closed_Fist": "ROCK",
    "Open_Palm": "PAPER",
    "Victory": "SCISSORS",
}
GESTURE_EMOJI: Dict[str, str] = {
    "ROCK": "ROCK [Fist]",
    "PAPER": "PAPER [Palm]",
    "SCISSORS": "SCISSORS [Peace]",
    "UNKNOWN": "UNKNOWN"
}

# Model Download Settings
MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
)
MODEL_FILENAME: str = "gesture_recognizer.task"

# Color Palette (BGR)
COLOR_BG_DARK = (18, 18, 24)
COLOR_CARD_BG = (28, 28, 38)
COLOR_BORDER = (60, 60, 75)
COLOR_CYAN = (255, 215, 0)
COLOR_GREEN = (60, 230, 80)
COLOR_RED = (50, 60, 240)
COLOR_AMBER = (0, 175, 255)
COLOR_PURPLE = (230, 80, 180)
COLOR_GOLD = (50, 215, 255)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (120, 120, 130)


# ==============================================================================
#                       MODEL AUTO-DOWNLOAD & SETUP
# ==============================================================================

def ensure_model_file() -> str:
    """Checks local directory and parent directories for gesture_recognizer.task, or downloads it."""
    candidate_paths = [
        MODEL_FILENAME,
        os.path.join("..", MODEL_FILENAME),
        os.path.join("car_game", MODEL_FILENAME),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return os.path.abspath(path)

    print(f"[Setup] MediaPipe Gesture model not found. Downloading from {MODEL_URL} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        print(f"[Setup] Download complete: {MODEL_FILENAME}")
        return os.path.abspath(MODEL_FILENAME)
    except Exception as e:
        print(f"[Error] Could not download model: {e}")
        raise e


# ==============================================================================
#                            GAME LOGIC & AI
# ==============================================================================

MOVES = ["ROCK", "PAPER", "SCISSORS"]

# Rules matrix: move -> move it defeats
WIN_RULES = {
    "ROCK": "SCISSORS",
    "PAPER": "ROCK",
    "SCISSORS": "PAPER"
}


def evaluate_winner(p1_move: str, p2_move: str) -> str:
    """
    Returns: 'P1' (Player 1 wins), 'P2' (Player 2/AI wins), or 'TIE'.
    """
    if p1_move not in MOVES or p2_move not in MOVES:
        return "INVALID"
    if p1_move == p2_move:
        return "TIE"
    if WIN_RULES[p1_move] == p2_move:
        return "P1"
    return "P2"


class AIBot:
    """AI Opponent supporting Normal (Random), Smart (Pattern Counter), and Chaos difficulty."""
    def __init__(self):
        self.player_history: List[str] = []

    def get_move(self, difficulty: str = "SMART") -> str:
        if difficulty == "NORMAL" or len(self.player_history) < 2:
            return random.choice(MOVES)
        elif difficulty == "SMART":
            # Smart AI: Predicts player's most frequent move and chooses its counter
            most_common = max(set(self.player_history), key=self.player_history.count)
            # Counter to player's favorite move
            counter_moves = {v: k for k, v in WIN_RULES.items()}
            return counter_moves[most_common]
        elif difficulty == "CHAOS":
            # Counter to the player's immediate last move with 60% probability
            last_move = self.player_history[-1]
            if random.random() < 0.65:
                counter_moves = {v: k for k, v in WIN_RULES.items()}
                return counter_moves[last_move]
            return random.choice(MOVES)
        return random.choice(MOVES)

    def record_player_move(self, move: str):
        if move in MOVES:
            self.player_history.append(move)
            if len(self.player_history) > 30:
                self.player_history.pop(0)


# ==============================================================================
#                         HUD & GRAPHICS RENDERING
# ==============================================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]


def draw_skeleton(frame: np.ndarray, landmarks, w: int, h: int, color=(0, 220, 255)):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, points[s], points[e], color, 2, cv2.LINE_AA)
    for idx, pt in enumerate(points):
        r = 4 if idx in [4, 8, 12, 16, 20] else 2
        cv2.circle(frame, pt, r, COLOR_WHITE, -1, cv2.LINE_AA)


def draw_card(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    size: Tuple[int, int],
    title: str,
    move_name: str,
    color_accent: Tuple[int, int, int],
    is_winner: bool = False
):
    """Draws a sleek choice card with glowing borders for Player or AI choice."""
    x, y = top_left
    w, h = size

    # Background Card
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_CARD_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    border_color = COLOR_GOLD if is_winner else color_accent
    border_thick = 3 if is_winner else 1
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, border_thick, cv2.LINE_AA)

    # Title header
    cv2.putText(frame, title, (x + 16, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)

    # Move Name
    move_color = color_accent if move_name in MOVES else COLOR_GRAY
    display_text = move_name
    text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
    tx = x + (w - text_size[0]) // 2
    ty = y + 75
    cv2.putText(frame, display_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.85, move_color, 2, cv2.LINE_AA)

    # Winner badge
    if is_winner:
        badge_txt = "★ ROUND WINNER ★"
        b_size = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        bx = x + (w - b_size[0]) // 2
        cv2.putText(frame, badge_txt, (bx, y + h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GOLD, 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    game_state: str,
    countdown_val: float,
    p1_move: str,
    p2_move: str,
    round_result: str,
    p1_score: int,
    p2_score: int,
    streak: int,
    difficulty: str,
    mode: str,
    fps: float
):
    h, w, _ = frame.shape

    # 1. Top Scoreboard Dashboard Bar
    bar_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 15), (w - 20, 15 + bar_h), COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (20, 15), (w - 20, 15 + bar_h), COLOR_BORDER, 1, cv2.LINE_AA)

    # Player 1 Score
    cv2.putText(frame, "YOU (PLAYER 1)", (45, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{p1_score}", (45, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 3, cv2.LINE_AA)

    # VS Center Badge
    vs_text = f"FIRST TO {MATCH_WINNING_SCORE}"
    cv2.putText(frame, "VS", (w // 2 - 20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.90, COLOR_AMBER, 2, cv2.LINE_AA)
    cv2.putText(frame, vs_text, (w // 2 - 48, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_GRAY, 1, cv2.LINE_AA)

    # Player 2 / AI Score
    p2_label = "PLAYER 2" if mode == "2PLAYER" else f"AI BOT [{difficulty}]"
    p2_size = cv2.getTextSize(p2_label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
    cv2.putText(frame, p2_label, (w - 45 - p2_size[0], 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_PURPLE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{p2_score}", (w - 75, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 3, cv2.LINE_AA)

    # Streak & FPS
    cv2.putText(frame, f"🔥 WIN STREAK: {streak}", (180, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_GOLD if streak > 0 else COLOR_GRAY, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (180, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GRAY, 1, cv2.LINE_AA)

    # 2. Main Gameplay Display by State
    if game_state == "IDLE":
        # Waiting for round start banner
        banner_w, banner_h = 580, 75
        bx = (w - banner_w) // 2
        by = h // 2 - 40
        b_overlay = frame.copy()
        cv2.rectangle(b_overlay, (bx, by), (bx + banner_w, by + banner_h), (25, 25, 35), -1)
        cv2.addWeighted(b_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), COLOR_CYAN, 2, cv2.LINE_AA)
        
        msg1 = "PRESS [SPACE] OR SHOW 👍 TO PLAY!"
        msg2 = "Hold ✊ ROCK, ✋ PAPER, or ✌️ SCISSORS on Shoot"
        cv2.putText(frame, msg1, (bx + 40, by + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, msg2, (bx + 55, by + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_CYAN, 1, cv2.LINE_AA)

    elif game_state == "COUNTDOWN":
        # Animated Countdown Circle in Center
        cx, cy = w // 2, h // 2
        sec_left = max(0, int(countdown_val) + 1)
        
        # Pulse animation circle
        rad = 70 + int((countdown_val % 1.0) * 20)
        cv2.circle(frame, (cx, cy), rad, COLOR_CARD_BG, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), rad, COLOR_AMBER, 4, cv2.LINE_AA)
        
        cnt_text = f"{sec_left}" if sec_left > 0 else "SHOOT!"
        ts = cv2.getTextSize(cnt_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0]
        cv2.putText(frame, cnt_text, (cx - ts[0] // 2, cy + ts[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, COLOR_WHITE, 4, cv2.LINE_AA)
        
        status_sub = "READY YOUR HAND: ✊ ROCK | ✋ PAPER | ✌️ SCISSORS"
        ss = cv2.getTextSize(status_sub, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
        cv2.putText(frame, status_sub, (w // 2 - ss[0] // 2, cy + rad + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_CYAN, 1, cv2.LINE_AA)

    elif game_state == "RESULT":
        # Show Choice Cards for Player 1 and Player 2/AI
        card_w, card_h = 320, 110
        p1_x = 80
        p2_x = w - card_w - 80
        card_y = h // 2 - 55

        draw_card(frame, (p1_x, card_y), (card_w, card_h), "YOU PLAYED:", p1_move, COLOR_CYAN, is_winner=(round_result == "P1"))
        draw_card(frame, (p2_x, card_y), (card_w, card_h), p2_label + " PLAYED:", p2_move, COLOR_PURPLE, is_winner=(round_result == "P2"))

        # Center Winner Announcement
        cx, cy = w // 2, h // 2
        if round_result == "P1":
            res_txt = "YOU WIN THIS ROUND! 🎉"
            res_color = COLOR_GREEN
        elif round_result == "P2":
            res_txt = f"{'PLAYER 2' if mode == '2PLAYER' else 'AI BOT'} WINS ROUND! 💀"
            res_color = COLOR_RED
        elif round_result == "TIE":
            res_txt = "IT'S A TIE! 🤝"
            res_color = COLOR_AMBER
        else:
            res_txt = "NO GESTURE DETECTED!"
            res_color = COLOR_GRAY

        r_size = cv2.getTextSize(res_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
        
        res_bg_w = r_size[0] + 40
        res_bg_h = 55
        rx = cx - res_bg_w // 2
        ry = card_y + card_h + 30
        
        r_overlay = frame.copy()
        cv2.rectangle(r_overlay, (rx, ry), (rx + res_bg_w, ry + res_bg_h), (20, 20, 30), -1)
        cv2.addWeighted(r_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (rx, ry), (rx + res_bg_w, ry + res_bg_h), res_color, 2, cv2.LINE_AA)
        cv2.putText(frame, res_txt, (rx + 20, ry + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, res_color, 2, cv2.LINE_AA)

    elif game_state == "MATCH_OVER":
        # Match victory screen
        banner_w, banner_h = 620, 110
        bx = (w - banner_w) // 2
        by = h // 2 - 55
        
        b_overlay = frame.copy()
        cv2.rectangle(b_overlay, (bx, by), (bx + banner_w, by + banner_h), (20, 20, 30), -1)
        cv2.addWeighted(b_overlay, 0.90, frame, 0.10, 0, frame)
        
        winner_is_p1 = (p1_score >= MATCH_WINNING_SCORE)
        win_col = COLOR_GOLD if winner_is_p1 else COLOR_RED
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), win_col, 3, cv2.LINE_AA)
        
        win_msg = "🏆 VICTORY! YOU WON THE MATCH! 🏆" if winner_is_p1 else "💀 DEFEAT! OPPONENT WON THE MATCH!"
        cv2.putText(frame, win_msg, (bx + 30, by + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, win_col, 2, cv2.LINE_AA)
        cv2.putText(frame, "PRESS [SPACE] OR [R] TO START A NEW MATCH", (bx + 70, by + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)

    # 3. Bottom Guide / Keyboard Shortcuts Bar
    help_overlay = frame.copy()
    cv2.rectangle(help_overlay, (0, h - 28), (w, h), (10, 10, 15), -1)
    cv2.addWeighted(help_overlay, 0.80, frame, 0.20, 0, frame)
    guide_text = "[SPACE] Next Round  |  [👍 THUMBS UP] Start  |  [M] Mode  |  [D] Difficulty  |  [R] Reset  |  [Q] Quit"
    cv2.putText(frame, guide_text, (25, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 190), 1, cv2.LINE_AA)


# ==============================================================================
#                                MAIN APPLICATION
# ==============================================================================

def main():
    print("=" * 70)
    print("        AI ROCK, PAPER, SCISSORS SHOWDOWN - LAUNCHING")
    print("=" * 70)
    print("[Controls] ✊ Closed Fist = ROCK | ✋ Open Palm = PAPER | ✌️ Victory = SCISSORS")
    print("[Controls] 👍 Thumbs Up or SPACE = Start Countdown")
    print("[Controls] M = Toggle Mode | D = Toggle Difficulty | R = Reset | Q = Quit")
    print("-" * 70)

    # 1. Initialize Model
    model_path = ensure_model_file()
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

    # 2. Camera Setup
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[Error] Could not open webcam index {CAMERA_INDEX}.")
        return

    # 3. Game State
    ai_bot = AIBot()
    game_mode = "PVA"         # "PVA" (Player vs AI) or "2PLAYER"
    difficulties = ["NORMAL", "SMART", "CHAOS"]
    diff_idx = 1             # Default to SMART
    
    p1_score = 0
    p2_score = 0
    win_streak = 0
    
    game_state = "IDLE"      # "IDLE", "COUNTDOWN", "RESULT", "MATCH_OVER"
    countdown_timer_start = 0.0
    result_timer_start = 0.0
    
    last_p1_move = "NONE"
    last_p2_move = "NONE"
    last_round_result = "NONE"

    prev_time = time.time()
    fps_val = 0.0
    thumbs_up_cooldown = 0.0

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                time.sleep(0.03)
                continue

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_val = 0.9 * fps_val + 0.1 * (1.0 / dt) if fps_val > 0 else (1.0 / dt)

            if MIRROR_VIEW:
                frame = cv2.flip(raw_frame, 1)
            else:
                frame = raw_frame

            frame_h, frame_w, _ = frame.shape

            # MediaPipe Inference
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            recognition_result = recognizer.recognize(mp_image)

            detected_hands_landmarks = recognition_result.hand_landmarks
            detected_gestures = recognition_result.gestures

            # Draw Hand Skeletons
            for lms in detected_hands_landmarks:
                draw_skeleton(frame, lms, frame_w, frame_h)

            # Classify Gestures for visible hands
            current_gestures = []
            thumbs_up_detected = False

            for g_list in detected_gestures:
                if g_list and len(g_list) > 0:
                    top_g = g_list[0]
                    if top_g.score >= CONFIDENCE_THRESHOLD:
                        if top_g.category_name in GESTURE_MAP:
                            current_gestures.append(GESTURE_MAP[top_g.category_name])
                        elif top_g.category_name == "Thumb_Up":
                            thumbs_up_detected = True

            # Quick-Start with Thumbs Up
            if thumbs_up_detected and (curr_time - thumbs_up_cooldown > 2.0):
                if game_state in ["IDLE", "RESULT"]:
                    if game_state == "RESULT" and (p1_score >= MATCH_WINNING_SCORE or p2_score >= MATCH_WINNING_SCORE):
                        p1_score = 0
                        p2_score = 0
                        win_streak = 0
                    game_state = "COUNTDOWN"
                    countdown_timer_start = curr_time
                    thumbs_up_cooldown = curr_time
                    print("[Game] Thumbs Up detected -> Starting Countdown!")

            # State Machine Update
            countdown_remaining = 0.0
            if game_state == "COUNTDOWN":
                elapsed = curr_time - countdown_timer_start
                countdown_remaining = max(0.0, COUNTDOWN_SECONDS - elapsed)

                if elapsed >= COUNTDOWN_SECONDS:
                    # Capture moves at "SHOOT!" moment
                    if game_mode == "PVA":
                        # Single Player vs AI
                        if len(current_gestures) > 0:
                            last_p1_move = current_gestures[0]
                        else:
                            last_p1_move = "UNKNOWN"

                        last_p2_move = ai_bot.get_move(difficulties[diff_idx])
                        if last_p1_move in MOVES:
                            ai_bot.record_player_move(last_p1_move)

                    else:
                        # 2-Player Local Mode
                        if len(current_gestures) >= 2:
                            # Assign left hand in image to P1, right hand to P2
                            last_p1_move = current_gestures[0]
                            last_p2_move = current_gestures[1]
                        elif len(current_gestures) == 1:
                            last_p1_move = current_gestures[0]
                            last_p2_move = "UNKNOWN"
                        else:
                            last_p1_move = "UNKNOWN"
                            last_p2_move = "UNKNOWN"

                    # Evaluate outcome
                    last_round_result = evaluate_winner(last_p1_move, last_p2_move)

                    if last_round_result == "P1":
                        p1_score += 1
                        win_streak += 1
                    elif last_round_result == "P2":
                        p2_score += 1
                        win_streak = 0

                    result_timer_start = curr_time
                    if p1_score >= MATCH_WINNING_SCORE or p2_score >= MATCH_WINNING_SCORE:
                        game_state = "MATCH_OVER"
                    else:
                        game_state = "RESULT"

                    print(f"[Round] P1: {last_p1_move} vs P2: {last_p2_move} -> Result: {last_round_result}")

            elif game_state == "RESULT":
                # Auto-transition to IDLE after result display
                if curr_time - result_timer_start >= RESULT_DISPLAY_SECONDS:
                    game_state = "IDLE"

            # Render Full Racing HUD
            draw_hud(
                frame=frame,
                game_state=game_state,
                countdown_val=countdown_remaining,
                p1_move=last_p1_move,
                p2_move=last_p2_move,
                round_result=last_round_result,
                p1_score=p1_score,
                p2_score=p2_score,
                streak=win_streak,
                difficulty=difficulties[diff_idx],
                mode=game_mode,
                fps=fps_val
            )

            cv2.imshow("Rock Paper Scissors Showdown - AI Game", frame)

            # Keyboard Handling
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("\n[Exit] User quit.")
                break
            elif key in [32, 13]:  # Space or Enter
                if game_state in ["IDLE", "RESULT"]:
                    game_state = "COUNTDOWN"
                    countdown_timer_start = curr_time
                elif game_state == "MATCH_OVER":
                    p1_score = 0
                    p2_score = 0
                    win_streak = 0
                    game_state = "COUNTDOWN"
                    countdown_timer_start = curr_time
            elif key in [ord('r'), ord('R')]:
                p1_score = 0
                p2_score = 0
                win_streak = 0
                game_state = "IDLE"
                print("[Game] Scores reset.")
            elif key in [ord('m'), ord('M')]:
                game_mode = "2PLAYER" if game_mode == "PVA" else "PVA"
                p1_score = 0
                p2_score = 0
                win_streak = 0
                game_state = "IDLE"
                print(f"[Game] Mode switched to: {game_mode}")
            elif key in [ord('d'), ord('D')]:
                diff_idx = (diff_idx + 1) % len(difficulties)
                print(f"[Game] AI Difficulty set to: {difficulties[diff_idx]}")

            if cv2.getWindowProperty("Rock Paper Scissors Showdown - AI Game", cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\n[Exit] Interrupted.")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        if 'recognizer' in locals():
            recognizer.close()
        print("[Exit] Thanks for playing!")


if __name__ == "__main__":
    main()
