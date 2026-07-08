"""
共享状态 — 视觉进程 ↔ 语音进程 通信

视觉进程写入 (empathy_track_servo_v3.py):
  from shared_state import write_vision_state
  write_vision_state(emotion="Happy", confidence=0.92)

语音进程读取 (pipeline.py):
  from shared_state import read_vision_state
  state = read_vision_state()
  state["emotion"]  # => "Happy"

安全设计:
  - 文件 /tmp/morpheus_vision.json
  - 视觉挂了 → 语音读到旧数据, 不影响
  - 语音挂了 → 视觉继续写, 不影响
"""
import json
import os
import time
import threading

VISION_STATE_FILE = "/tmp/morpheus_vision.json"
VOICE_STATE_FILE = "/tmp/morpheus_voice.json"

# ─── 跨进程接口（文件） ───

def write_vision_state(emotion="neutral", confidence=0.0, face_detected=False, user_identity=None, face_box=None, face_center=None):
    """视觉进程调用：写入当前情绪状态"""
    state = {
        "emotion": emotion,
        "confidence": round(float(confidence), 3),
        "face_detected": bool(face_detected),
        "user_identity": user_identity,
        "face_box": face_box,
        "face_center": face_center,
        "timestamp": time.time()
    }
    try:
        with open(VISION_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass


def read_vision_state():
    """语音进程调用：读取视觉状态"""
    try:
        with open(VISION_STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "face_detected": False,
            "user_identity": None,
            "timestamp": 0.0
        }


def get_emotion_prompt():
    """给 LLM 用的情绪描述，直接拼到 system prompt"""
    state = read_vision_state()
    if not state["face_detected"] or state["confidence"] < 0.3:
        return ""
    return f"用户当前情绪: {state['emotion']} ({state['confidence']*100:.0f}%)"



# ─── 语音状态（跨进程） ───

def write_voice_state(speaking=None, status=None, user_text=None, bot_text=None, token_speed=None, tts_latency=None):
    """语音进程调用：写入说话状态 + 对话内容（仅更新非 None 字段）"""
    try:
        with open(VOICE_STATE_FILE) as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {"speaking": False, "status": "idle", "user_text": "", "bot_text": ""}
    if speaking is not None:
        state["speaking"] = bool(speaking)
    if status is not None:
        state["status"] = status
    if user_text is not None:
        state["user_text"] = user_text
    if bot_text is not None:
        state["bot_text"] = bot_text
    if token_speed is not None:
        state["token_speed"] = token_speed
    if tts_latency is not None:
        state["tts_latency"] = tts_latency
    try:
        with open(VOICE_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass


def read_voice_state():
    """视觉/UI进程调用：读取语音状态"""
    try:
        with open(VOICE_STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "speaking": False, "status": "idle",
            "user_text": "", "bot_text": "", "timestamp": 0.0
        }


# ─── 进程内接口（线程安全，原有逻辑不变） ───

class SharedState:
    """线程安全的共享状态（进程内用）"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self._data = {}
        self._data_lock = threading.Lock()

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattribute__(name)
        with self._data_lock:
            return self._data.get(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        with self._data_lock:
            self._data[name] = value


# 全局单例（进程内用）
shared = SharedState()
