# 🏎️ Gesture Car Control

Control any browser or desktop racing game using webcam hand gestures with real-time HUD telemetry, MediaPipe AI tracking, and OS-level keyboard controls.

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install opencv-python mediapipe pynput numpy
```

### 2. Run the Game
- **From root directory**:
  ```bash
  python car_game/gesture_car_control.py
  ```
- **Or from inside the `car_game` folder**:
  ```bash
  cd car_game
  python gesture_car_control.py
  ```
*(The AI gesture model `gesture_recognizer.task` will automatically download on first run.)*

### 3. Play
1. Open any racing game in your browser (e.g. [Slow Roads](https://slowroads.io)).
2. **Click inside the game window** so it has keyboard focus.
3. Use hand gestures in front of your webcam to drive!

---

## 🎮 Hand Gesture Controls

| Gesture | Action | Game Key / Response |
| :--- | :--- | :--- |
| **✌️ Victory / Peace Sign** | **Start Game & Engine** | Taps `Space` / `Enter` to start & surges throttle |
| **👐 Two-Hand Wheel Tilt** | **Steer Left / Right** | Tilt left/right like a steering wheel (`Left` / `Right` arrows) |
| **✋ Right Hand Up / Down** | **Gas (Accelerate)** | Move hand higher for more gas (`Up` arrow) |
| **✊ Closed Fist** | **Brake / Reverse** | Cuts gas and holds `Down` arrow |
| **🔍 `+` / `-`** | **Scale Window** | Scale HUD window larger or smaller |
| **🎯 `0`** | **Reset Window Size** | Reset to default compact overlay size (640x360) |
| **📌 `t`** | **Pin Always-on-Top** | Toggle floating Picture-in-Picture mode over game |
| **⌨️ `q` or `Esc`** | **Exit** | Safely releases all keys & closes webcam |

---

## ⚙️ Key Configuration (Top of `gesture_car_control.py`)

All thresholds, window sizes, and keybindings can be tuned directly at the top of the file:

```python
# --- Window Scaling & Display Settings ---
WINDOW_INITIAL_WIDTH = 640   # Default HUD window width
WINDOW_INITIAL_HEIGHT = 360  # Default HUD window height (16:9 ratio)
WINDOW_RESIZABLE = True      # Drag borders to resize freely
WINDOW_ALWAYS_ON_TOP = True  # Float over browser game window

# --- Key Bindings ---
KEY_GAS = Key.up          # Accelerate
KEY_BRAKE = Key.down      # Brake / Reverse
KEY_LEFT = Key.left       # Steer Left
KEY_RIGHT = Key.right     # Steer Right
KEY_START_GAME = Key.space # Start / Ignition

# --- Steering Settings ---
STEER_DEADZONE_DEG = 8.0  # Straight / Center threshold (degrees)
STEER_MAX_DEG = 30.0      # Full continuous turn lock (degrees)

# --- Gas / Speed Modulation Settings ---
GAS_IDLE_THRESHOLD = 0.18 # Below this: Coasting / Idle (0% gas)
GAS_FULL_THRESHOLD = 0.75 # Above this: 100% full gas hold
GAS_Y_TOP = 0.20          # Hand height for 100% gas (higher up)
GAS_Y_BOTTOM = 0.60       # Hand height for 0% gas (normal wheel height)
```

---

## 🏁 Recommended Games
- **[Slow Roads (slowroads.io)](https://slowroads.io)** *(Recommended for smooth testing)*
- **[Madalin Stunt Cars 2](https://www.crazygames.com/game/madalin-stunt-cars-2)**
- **[Drift Hunters](https://www.crazygames.com/game/drift-hunters)**
- Any desktop or browser racing game supporting Arrow Keys.
