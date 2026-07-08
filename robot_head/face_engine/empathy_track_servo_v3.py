#!/usr/bin/env python3
import random as _rnd
"""
Morpheus 情绪识别 + 共情表情 + PID 视觉追踪 集成版 v3
基于 v2 的共情管线 + face_track_fixed.py 的 ServoTracker。
通道分离：追踪控制 {6,23,30,31,32}，表情控制其余24通道。
"""
import sys
sys.path.insert(0, "/home/elf/robot_head")
from shared_state import write_vision_state, read_voice_state
from face_engine.panel_renderer import render_panel
n_panel_cache = None
_panel_fc = 0
# ─── 偏置参数 ───
ANGRY_BIAS = 0.42
DISGUST_BIAS = 0.35
SAD_BIAS = 0.15
# 追踪通道（不参与共情，由 PID 独立控制）
TRACK_CHANNELS = {6, 23, 30, 31, 32}
TRACK_MAX_PREDICTIONS = 50
import cv2, numpy as np, os, time, json, sys, argparse, threading
SHOW_WINDOW = True
import torch
from itertools import product as product
from math import ceil
from rknnlite.api import RKNNLite
from face_engine.mor_servo_dev import PCA9685, angle_to_pulse_us, angle_to_pulse_us_with_mid, pulse_us_to_angle_with_mid
from face_engine.set_start_bound import get_hardware_target, TABLE_V_CONFIG, clamp_angle
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE), "models")
RETINAFACE_MODEL = os.path.join(MODELS_DIR, "RetinaFace_mobile320.rknn")
MBF_MODEL = os.path.join(MODELS_DIR, "w600k_mbf.rknn")
EMOTION_MODEL = os.path.join(MODELS_DIR, "emotion_mobilenetv2_v9_nchw_fp16.rknn")
EMPATHY_CKPT = os.path.join(MODELS_DIR, "empathy_best_new.pth")
UPPER_MODEL_PATH = os.path.join(MODELS_DIR, "upper_face_bs2angle.pth")
LOWER_MODEL_PATH = os.path.join(MODELS_DIR, "lower_face_bs2angle.pth")
DB_PATH = os.path.join(MODELS_DIR, "face_db.json")
sys.path.insert(0, BASE)
from face_engine.empathy_model import EmpathyModel
from face_engine.empathy_data import ALL_GENBATCH_NAMES
CAMERA_INDEX = 21
MODEL_SIZE = (320, 320)
SCORE_THRESH = 0.6
NMS_THRESH = 0.4
EMBED_SIZE = (112, 112)
SIM_THRESH = 0.6
SEQ_LEN = 48
ARKIT_BS_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]
EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
# ═══════════════════════════════════════════════════════════════
#  RetinaFace helpers
# ═══════════════════════════════════════════════════════════════
def prior_box(image_size):
    anchors = []
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for ms in min_sizes[k]:
                s_kx = ms / image_size[1]; s_ky = ms / image_size[0]
                cx = (j + 0.5) * steps[k] / image_size[1]
                cy = (i + 0.5) * steps[k] / image_size[0]
                anchors += [cx, cy, s_kx, s_ky]
    return np.array(anchors).reshape(-1, 4).astype(np.float32)
_priors = prior_box(MODEL_SIZE)
def decode_boxes(l, p, v=(0.1, 0.2)):
    b = np.concatenate((p[:, :2] + l[:, :2] * v[0] * p[:, 2:], p[:, 2:] * np.exp(l[:, 2:] * v[1])), axis=1)
    b[:, :2] -= b[:, 2:] / 2; b[:, 2:] += b[:, :2]; return b
def decode_landmarks(pre, p, v=(0.1, 0.2)):
    return np.concatenate([p[:, :2] + pre[:, k:k+2] * v[0] * p[:, 2:] for k in range(0,10,2)], axis=1)
def nms(dets, thresh):
    if len(dets) == 0: return []
    x1,y1,x2,y2,s = dets[:,0],dets[:,1],dets[:,2],dets[:,3],dets[:,4]
    areas=(x2-x1+1)*(y2-y1+1); order=s.argsort()[::-1]; keep=[]
    while order.size>0:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        w=np.maximum(0.,xx2-xx1+1); h=np.maximum(0.,yy2-yy1+1)
        ovr=w*h/(areas[i]+areas[order[1:]]-w*h)
        order=order[np.where(ovr<=thresh)[0]+1]
    return keep
def letterbox_resize(image, size, bg=114):
    tw, th = size; h, w = image.shape[:2]
    scale = min(tw/w, th/h); nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.ones((th, tw, 3), dtype=np.uint8) * bg
    ox, oy = (tw-nw)//2, (th-nh)//2
    canvas[oy:oy+nh, ox:ox+nw] = resized
    return canvas, scale, ox, oy
CANONICAL = np.array([
    [38.2946, 51.6963], [73.5318, 51.6963], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.3655],
], dtype=np.float32)
def align_face(image, landmarks_5):
    src = landmarks_5.reshape(5, 2).astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, CANONICAL, method=cv2.LMEDS)
    if M is None: return None
    return cv2.warpAffine(image, M, EMBED_SIZE, flags=cv2.INTER_LINEAR)
def preprocess_for_mbf(face_bgr):
    rgb = face_bgr[..., ::-1]; chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw.astype(np.float32), 0)
def crop_face_for_emotion_rknn(frame, bbox):
    x1,y1,x2,y2 = map(int, bbox); h,w = frame.shape[:2]
    bw, bh = x2-x1, y2-y1; cx, cy = (x1+x2)/2, (y1+y2)/2
    margin = 0.3; nw = int(bw*(1+margin)); nh = int(bh*(1+margin))
    x1 = max(0, int(cx-nw/2)); y1 = max(0, int(cy-nh/2))
    x2 = min(w-1, int(cx+nw/2)); y2 = min(h-1, int(cy+nh/2))
    face = frame[y1:y2, x1:x2]
    if face.size == 0: return None
    face128 = cv2.resize(face, (128, 128))
    rgb = face128[..., ::-1].astype(np.uint8)
    return np.expand_dims(rgb, 0)
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
# ═══════════════════════════════════════════════════════════════
#  Kalman Filter (from face_track_fixed.py)
# ═══════════════════════════════════════════════════════════════
class SimpleKalman:
    def __init__(self):
        self.state = np.zeros(4); self.P = np.eye(4) * 10
        self.F = np.eye(4); self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.1; self.R = np.eye(2) * 5
    def predict(self, dt):
        self.F[0, 2] = dt; self.F[1, 3] = dt
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    def update(self, z):
        y = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)
# ═══════════════════════════════════════════════════════════════
#  Servo Tracker — PID 眼/颈追踪 (from face_track_fixed.py)
#  接收共享 PCA9685 实例，只控制 {6,23,30,31,32}
# ═══════════════════════════════════════════════════════════════
TRACK_CONFIG = {
    6:  (30.0,  110.0,  60.0),
    23: (55.0,  95.0,  76.0),
    30: (50.0,  165.0, 155.0),
    31: (20.0,  100.0, 34.0),
    32: (0,   180.0, 90.0),
}
class ServoTracker:
    def __init__(self, pcas):
        self.pcas = pcas
        self.curr_angles = {}
        for ch, cfg in TRACK_CONFIG.items():
            try:
                pca, lch = self.get_target(ch)
                if pca:
                    p = pca.get_servo_pulse_us(lch)
                    if p > 1.0:
                        self.curr_angles[ch] = pulse_us_to_angle_with_mid(p, 0, 90, 180, 600, 1500, 2400)
                    else:
                        self.curr_angles[ch] = cfg[2]
                else:
                    self.curr_angles[ch] = cfg[2]
            except Exception:
                self.curr_angles[ch] = cfg[2]
        self.kp_x = self.kp_y = 0.5
        self.ki_x = self.ki_y = 0.001
        self.kd_x = self.kd_y = 0.08
        self.i_max = 3.0
        self.deriv_alpha = 0.15
        self.error_sum_x = self.error_sum_y = 0.0
        self.last_error_x = self.last_error_y = 0.0
        self.deriv_x = self.deriv_y = 0.0
        self.smooth = 0.45; self.max_step = 3.0; self.deadzone = 8
        self.micro_zone_x, self.micro_zone_y = 115, 50
        self.eye_x_mid = TRACK_CONFIG[23][2]
        self.align_p = 0.3; self.align_deadzone = 3.0
        self.kf = SimpleKalman(); self.last_time = time.time()
        self.has_detected_first_time = False; self.prediction_count = 0
        self.MAX_PREDICTIONS = TRACK_MAX_PREDICTIONS; self.is_sleeping = False
    def get_target(self, global_ch):
        if global_ch <= 15: return self.pcas[0x40], global_ch
        if global_ch <= 31: return self.pcas[0x41], global_ch - 16
        return self.pcas[0x42], 0
    def update_servo(self, ch, target_angle):
        current_a = self.curr_angles[ch]; diff = target_angle - current_a
        if abs(diff) > self.max_step:
            diff = self.max_step if diff > 0 else -self.max_step
        new_angle = current_a + diff * self.smooth
        min_a, max_a, _ = TRACK_CONFIG[ch]; new_angle = max(min_a, min(max_a, new_angle))
        pca, local_ch = self.get_target(ch)
        if pca:
            try:
                pulse = angle_to_pulse_us_with_mid(new_angle, 0, 90, 180, 600, 1500, 2400)
                pca.set_servo_pulse_us(local_ch, pulse)
                self.curr_angles[ch] = new_angle
            except OSError:
                pass
    def track(self, error_x, error_y, is_real_data=True):
        self.is_sleeping = False; now = time.time(); dt = now - self.last_time; self.last_time = now
        self.kf.predict(dt)
        if is_real_data:
            self.kf.update(np.array([error_x, error_y]))
            self.has_detected_first_time = True; self.prediction_count = 0
        else:
            self.prediction_count += 1
            error_x, error_y = self.kf.state[0], self.kf.state[1]
        if self.prediction_count > self.MAX_PREDICTIONS:
            self.sleep(); return
        if is_real_data:
            self.error_sum_x += error_x * dt; self.error_sum_y += error_y * dt
            self.error_sum_x = max(-self.i_max, min(self.i_max, self.error_sum_x))
            self.error_sum_y = max(-self.i_max, min(self.i_max, self.error_sum_y))
        if self.has_detected_first_time and dt > 0.001:
            raw_dx = (error_x - self.last_error_x) / dt
            raw_dy = (error_y - self.last_error_y) / dt
            self.deriv_x = self.deriv_x * (1 - self.deriv_alpha) + raw_dx * self.deriv_alpha
            self.deriv_y = self.deriv_y * (1 - self.deriv_alpha) + raw_dy * self.deriv_alpha
        self.last_error_x = error_x; self.last_error_y = error_y
        move_x = error_x * self.kp_x + self.error_sum_x * self.ki_x + self.deriv_x * self.kd_x
        move_y = error_y * self.kp_y + self.error_sum_y * self.ki_y + self.deriv_y * self.kd_y
        # X: 小偏差只用眼球，大偏差加脖子
        if abs(error_x) < self.micro_zone_x:
            self.update_servo(23, self.curr_angles[23] - move_x * 0.8)
        else:
            self.update_servo(23, self.curr_angles[23] - move_x * 0.25)
            self.update_servo(32, self.curr_angles[32] - move_x * 0.75)
        if is_real_data and abs(error_x) < (self.micro_zone_x * 0.4):
            eye_x_offset = self.curr_angles[23] - self.eye_x_mid
            if abs(eye_x_offset) > self.align_deadzone:
                align_speed = max(-0.8, min(0.8, eye_x_offset * self.align_p))
                self.update_servo(32, self.curr_angles[32] + align_speed)
                self.update_servo(23, self.curr_angles[23] - align_speed)
        # Y
        if abs(error_y) < self.micro_zone_y:
            self.update_servo(6, self.curr_angles[6] - move_y * 1.0)
        else:
            self.update_servo(6, self.curr_angles[6] - move_y * 0.4)
            self.update_servo(30, self.curr_angles[30] - move_y * 0.8)
            self.update_servo(31, self.curr_angles[31] + move_y * 0.8)
    def reset_pose(self):
        for ch in TRACK_CONFIG:
            self.update_servo(ch, TRACK_CONFIG[ch][2])
    def sleep(self):
        if not self.is_sleeping:
            print(">>> 追踪目标丢失，追踪舵机进入待机")
            for ch in TRACK_CONFIG:
                pca, local_ch = self.get_target(ch)
                if pca:
                    try:
                        pca.set_channel_full_off(local_ch, True)
                    except OSError:
                        pass
            self.is_sleeping = True; self.has_detected_first_time = False
            self.error_sum_x = self.error_sum_y = 0.0
            self.deriv_x = self.deriv_y = 0.0
    def cleanup(self):
        for ch in TRACK_CONFIG:
            pca, local_ch = self.get_target(ch)
            if pca:
                try:
                    pca.set_channel_full_off(local_ch, True)
                except OSError:
                    pass
# ═══════════════════════════════════════════════════════════════
#  FaceBS2Angle MLP 模型
# ═══════════════════════════════════════════════════════════════
class FaceBS2Angle(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 128, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
            prev = h
        layers.append(torch.nn.Linear(prev, output_dim))
        layers.append(torch.nn.Sigmoid())
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
def bs_to_servo(bs_arkit_52, models):
    bs_keys = ARKIT_BS_NAMES
    device = models["device"]
    bs_dict = {"_neutral": 0.0}
    for arkit_idx in range(51):
        bs_dict[bs_keys[arkit_idx+1]] = float(bs_arkit_52[arkit_idx])
    upper_input = [bs_dict.get(k, 0.0) for k in models["upper_bs_keys"]]
    upper_t = torch.tensor([upper_input], dtype=torch.float32, device=device)
    with torch.no_grad():
        upper_pred = models["upper_model"](upper_t).cpu().numpy()[0]
    lower_input = [bs_dict.get(k, 0.0) for k in models["lower_bs_keys"]]
    lower_t = torch.tensor([lower_input], dtype=torch.float32, device=device)
    with torch.no_grad():
        lower_pred = models["lower_model"](lower_t).cpu().numpy()[0]
    motor_ranges = models["motor_ranges"]
    angles = {}
    for mid in models["used_motors_full"]:
        minv, maxv = motor_ranges[mid]
        angles[str(mid)] = float((minv + maxv) / 2.0)
    for i, mid in enumerate(models["upper_motor_ids"]):
        minv, maxv = motor_ranges[mid]
        angles[str(mid)] = round(float(upper_pred[i]) * (maxv - minv) + minv, 2)
    for i, mid in enumerate(models["lower_motor_ids"]):
        minv, maxv = motor_ranges[mid]
        angles[str(mid)] = round(float(lower_pred[i]) * (maxv - minv) + minv, 2)
    return angles
# ═══════════════════════════════════════════════════════════════
#  Threaded Frame Capture — capture runs in background thread
# ═══════════════════════════════════════════════════════════════
class FrameCapture:
    """Capture frames in a background thread so AI and capture overlap."""
    def __init__(self, cam_idx, width=640, height=480, fps=30):
        self.cam_idx = cam_idx
        self.width = width
        self.height = height
        self.target_fps = fps
        self._frame = None
        self._ret = False
        self._seq = 0
        self.running = True
        self._open_camera()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    def _open_camera(self):
        try:
            cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap = cap
        except Exception:
            self.cap = None
    def _capture_loop(self):
        while self.running:
            if self.cap is None:
                self._ret = False
                time.sleep(0.2)
                self._open_camera()
                continue
            try:
                ret, frame = self.cap.read()
                if ret:
                    self._frame = frame
                    self._ret = True
                    self._seq += 1
                else:
                    self._ret = False
                    self.cap.release()
                    self.cap = None
            except Exception:
                self._ret = False
                if self.cap:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    time.sleep(0.1)
    def read(self):
        return self._ret, self._frame
    def is_opened(self):
        return self._ret
    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()
# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=CAMERA_INDEX)
    args, _ = p.parse_known_args()
    cam_idx = args.cam
    device = torch.device("cpu")
    print(f"Device: {device}")
    # ── 1. 加载 NPU 模型 ──
    print("[1/6] Loading RetinaFace NPU...")
    rkn_face = RKNNLite()
    rkn_face.load_rknn(RETINAFACE_MODEL)
    rkn_face.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    print("  OK")
    print("[2/6] Loading MobileFaceNet NPU...")
    rkn_mbf = RKNNLite()
    rkn_mbf.load_rknn(MBF_MODEL)
    rkn_mbf.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    print("  OK")
    print("[3/6] Loading Emotion RKNN (NPU CORE_2)...")
    rkn_emo = RKNNLite()
    rkn_emo.load_rknn(EMOTION_MODEL)
    rkn_emo.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    print(f"  OK: {EMOTION_MODEL}")
    # ── 2. 加载人脸库 ──
    print("[4/6] Loading face database...")
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found"); return
    with open(DB_PATH) as f:
        face_db = json.load(f)
    if len(face_db) == 0:
        print("ERROR: face_db.json is empty"); return
    ref_embeddings = [np.array(e["embedding"], dtype=np.float32) for e in face_db]
    ref_names = [e["name"] for e in face_db]
    print(f"  Registered: {ref_names}")
    # ── 3. 加载共情模型 ──
    print("[5/6] Loading empathy model...")
    ckpt = torch.load(EMPATHY_CKPT, map_location=device)
    cfg = ckpt.get("cfg", {})
    empathy_model = EmpathyModel(
        input_dim=7, output_dim=52,
        d_model=cfg.get("d_model", 64),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
        max_seq_len=cfg.get("seq_len", SEQ_LEN),
    ).to(device)
    empathy_model.load_state_dict(ckpt["model_state_dict"])
    empathy_model.eval()
    print(f"  OK (epoch={ckpt.get('epoch','?')})")
    # 计算中立 BS
    neutral_emo = np.zeros(7, dtype=np.float32)
    neutral_emo[4] = 0.95
    neutral_seq = np.tile(neutral_emo, (SEQ_LEN, 1))
    neut_t = torch.FloatTensor(neutral_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        _, neutral_bs_t = empathy_model(neut_t)
    neutral_bs = neutral_bs_t[0, -1].cpu().numpy()
    print("  Neutral BS computed")
    # ── 4. 加载角度模型 ──
    print("[6/6] Loading angle models...")
    upper_ckpt = torch.load(UPPER_MODEL_PATH, map_location=device)
    lower_ckpt = torch.load(LOWER_MODEL_PATH, map_location=device)
    upper_model = FaceBS2Angle(len(upper_ckpt["upper_bs_keys"]), len(upper_ckpt["upper_motor_ids"])).to(device)
    upper_model.load_state_dict(upper_ckpt["model_state_dict"]); upper_model.eval()
    lower_model = FaceBS2Angle(len(lower_ckpt["lower_bs_idx"]), len(lower_ckpt["lower_motor_ids"])).to(device)
    lower_model.load_state_dict(lower_ckpt["model_state_dict"]); lower_model.eval()
    used_motors = sorted(set(upper_ckpt["upper_motor_ids"]) | set(lower_ckpt["lower_motor_ids"]))
    motor_ranges = {}
    motor_ranges.update(upper_ckpt["motor_ranges"])
    motor_ranges.update(lower_ckpt["motor_ranges"])
    models = {
        "device": device,
        "upper_model": upper_model, "lower_model": lower_model,
        "upper_bs_keys": upper_ckpt["upper_bs_keys"],
        "lower_bs_keys": [ARKIT_BS_NAMES[i] for i in lower_ckpt["lower_bs_idx"]],
        "upper_motor_ids": upper_ckpt["upper_motor_ids"],
        "lower_motor_ids": lower_ckpt["lower_motor_ids"],
        "motor_ranges": motor_ranges,
        "used_motors_full": used_motors,
    }
    print(f"  OK: {len(used_motors)} motors, tracking channels: {sorted(TRACK_CHANNELS)}")
    # ── PCA9685 初始化 ──
    print("Initializing PCA9685...")
    pcas = {
        0x40: PCA9685(bus_id=4, address=0x40, freq_hz=50),
        0x41: PCA9685(bus_id=4, address=0x41, freq_hz=50),
        0x42: PCA9685(bus_id=4, address=0x42, freq_hz=50),
    }
    # 表情舵机回中立
    neutral_vec = np.array([(motor_ranges[mid][0] + motor_ranges[mid][1]) / 2.0 for mid in used_motors], dtype=np.float64)
    for i, mid in enumerate(used_motors):
        angle = float(neutral_vec[i])
        pca, ch = get_hardware_target(mid, pcas)
        if pca:
            try:
                pulse = angle_to_pulse_us(angle, 0.0, 180.0, 600.0, 2400.0)
                pca.set_servo_pulse_us(ch, pulse)
            except OSError:
                pass
    print(f"  PCA9685 OK: {len(used_motors)} expression servos at neutral")
    # ── 初始化视觉追踪（共享 PCA9685）──
    tracker = ServoTracker(pcas)
    tracker.reset_pose()
    print("  Tracker OK")
    # ── 摄像头（多线程 capture，与 AI 处理流水线并行）──
    cap = FrameCapture(cam_idx)
    time.sleep(0.3)  # wait for first frame
    if not cap.is_opened():
        print(f"ERROR: Cannot open camera {cam_idx}"); cap.release(); return
    print(f"\nCamera: {cam_idx} OK ({cap.width}x{cap.height})")
    # ── 状态变量 ──
    emotion_buffer = []
    _blink_t = time.time() + _rnd.uniform(3, 6)
    _blinking = False
    no_face_frames = 0
    last_raw_vec = neutral_vec.copy()
    last_written_angles = {}  # mid → angle, for skipping unchanged writes
    last_seq = -1  # for skipping duplicate frames
    smooth_cx = smooth_cy = None  # EMA-smoothed face position
    smoothed_bs = neutral_bs.copy()
    display_bs = neutral_bs.copy()
    last_log = 0
    last_emo7 = None
    last_valid_box = None
    frame_count = 0
    fps_timer = time.time(); fps = 0.0
    smooth_emo7 = None
    frame_target = 1.0 / 25.0  # 25fps cap
    print("\nRunning... (Ctrl+C to stop)\n")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or cap._seq == last_seq:
                last_seq = cap._seq
                time.sleep(0.001)
                continue
            last_seq = cap._seq
            frame_start = time.time()
            h, w = frame.shape[:2]
            frame_count += 1
            if frame_count % 15 == 0:
                now_fps = time.time()
                fps = 15 / (now_fps - fps_timer + 1e-6)
                fps_timer = now_fps
                if frame_count % 60 == 0:
                    print(f"  FPS: {fps:.1f}", flush=True)
            cx_center, cy_center = w // 2, h // 2
            # ── RetinaFace 人脸检测 ──
            lb, sc, ox, oy = letterbox_resize(frame, MODEL_SIZE)
            out = rkn_face.inference(inputs=[np.expand_dims(lb[..., ::-1], 0)])
            loc, conf, lms_raw = out[0].squeeze(0), out[1].squeeze(0), out[2].squeeze(0)
            boxes = decode_boxes(loc, _priors) * np.array([320, 320, 320, 320])
            boxes[:, 0::2] = np.clip((boxes[:, 0::2] - ox) / sc, 0, w)
            boxes[:, 1::2] = np.clip((boxes[:, 1::2] - oy) / sc, 0, h)
            lms = decode_landmarks(lms_raw, _priors) * np.array([320, 320] * 5)
            lms[:, 0::2] = np.clip((lms[:, 0::2] - ox) / sc, 0, w)
            lms[:, 1::2] = np.clip((lms[:, 1::2] - oy) / sc, 0, h)
            inds = np.where(conf[:, 1] > SCORE_THRESH)[0]
            boxes = boxes[inds]; lms = lms[inds]; scores = conf[inds, 1]
            face_detected = False
            registered_found = False
            best_name = ""
            best_sim = 0.0
            best_box = None
            best_lm = None
            best_aligned = None
            face_cx, face_cy = None, None  # for tracking
            if len(scores) > 0:
                dets = np.hstack((boxes, scores[:, None])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                boxes = boxes[keep]; lms = lms[keep]; scores = scores[keep]
                # MobileFaceNet recognition — 只追踪注册用户
                for i in range(len(scores)):
                    aligned = align_face(frame, lms[i])
                    if aligned is None: continue
                    emb = rkn_mbf.inference(inputs=[preprocess_for_mbf(aligned)])[0].flatten()
                    for j, ref_emb in enumerate(ref_embeddings):
                        s = cosine_sim(ref_emb, emb)
                        if s > SIM_THRESH and s > best_sim and ref_names[j] == "yinyin":
                            best_sim = s
                            best_name = ref_names[j]
                            best_box = boxes[i]
                            best_lm = lms[i]
                            best_aligned = aligned
                            registered_found = True
                            # 左眼关键点（landmark 索引1）
                            face_cx = lms[i][2]
                            face_cy = lms[i][3]
                            face_detected = True
            # ── PID 视觉追踪（每帧执行，人脸位置先 EMA 平滑）──
            if face_detected and face_cx is not None and face_cy is not None:
                # 指数平滑人脸位置，消除帧间跳动
                if smooth_cx is None:
                    smooth_cx, smooth_cy = face_cx, face_cy
                else:
                    alpha = 0.4  # 0=最平滑, 1=最灵敏
                    smooth_cx = smooth_cx * (1 - alpha) + face_cx * alpha
                    smooth_cy = smooth_cy * (1 - alpha) + face_cy * alpha
                error_x = smooth_cx - cx_center
                error_y = smooth_cy - cy_center
                if abs(error_x) > tracker.deadzone or abs(error_y) > tracker.deadzone:
                    tracker.track(error_x, error_y, is_real_data=True)
                else:
                    tracker.track(0, 0, is_real_data=True)
                no_face_frames = 0
            else:
                no_face_frames += 1
                if tracker.has_detected_first_time:
                    tracker.track(0, 0, is_real_data=False)
                else:
                    tracker.sleep()
                # 长时间无脸 → 表情归中立
                if no_face_frames > 60:
                    last_raw_vec = neutral_vec.copy()
                    emotion_buffer = []
            # ── 情绪推理（每3帧）──
            emo7 = None
            if registered_found:
                last_valid_box = best_box
                if True:
                    emo_in = crop_face_for_emotion_rknn(frame, best_box)
                    if emo_in is not None:
                        emo_out = rkn_emo.inference(inputs=[emo_in])[0].flatten()
                        raw_emo7 = emo_out.astype(np.float32)
                        raw_emo7[0] += ANGRY_BIAS
                        raw_emo7[1] += DISGUST_BIAS
                        raw_emo7[5] += SAD_BIAS
                        if smooth_emo7 is None:
                            smooth_emo7 = raw_emo7
                        else:
                            smooth_emo7 = smooth_emo7 * 0.7 + raw_emo7 * 0.3
                        emo7 = smooth_emo7.copy()
                        last_emo7 = emo7
                else:
                    if last_emo7 is not None:
                        emo7 = last_emo7
            # ── 共情 + BS2Angle ──
            if emo7 is not None:
                emotion_buffer.append(emo7)
                if len(emotion_buffer) > SEQ_LEN:
                    emotion_buffer.pop(0)
                if len(emotion_buffer) >= 3:
                    seq = np.stack(emotion_buffer)
                    if len(seq) >= SEQ_LEN:
                        window = seq[-SEQ_LEN:]
                    else:
                        reps = int(np.ceil(SEQ_LEN / len(seq)))
                        window = np.tile(seq, (reps, 1))[:SEQ_LEN]
                    tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, bs_out = empathy_model(tensor)
                    empathy_bs = bs_out[0, -1].cpu().numpy()
                    smoothed_bs = smoothed_bs * 0.5 + empathy_bs * 0.5
                    blend = min(1.0, len(emotion_buffer) / 3)
                    delta = smoothed_bs - neutral_bs
                    amplified_bs = neutral_bs + delta * 3.5
                    amplified_bs[24] = min(amplified_bs[24], 0.3)  # jawOpen cap
                    amplified_bs = np.clip(amplified_bs, 0.0, 1.0)
                    final_bs = neutral_bs * (1 - blend) + amplified_bs * blend
                    display_bs = final_bs.copy()
                else:
                    final_bs = neutral_bs.copy()
                    display_bs = final_bs.copy()
                # BS → 角度
                raw_angles_dict = bs_to_servo(final_bs, models)
                last_raw_vec = np.array([raw_angles_dict[str(mid)] for mid in used_motors], dtype=np.float64)
            else:
                final_bs = neutral_bs.copy()
                display_bs = final_bs.copy()
            # ── 写表情舵机（眨眼时让出眼睑控制） ──
            _vs = read_voice_state()
            _speaking = _vs.get("speaking", False)
            _lower_set = set(models["lower_motor_ids"])
            _EYELID_SET = {4,5,24,25}
            smooth_vec = last_raw_vec
            for i, mid in enumerate(used_motors):
                if mid in TRACK_CHANNELS:
                    continue  # 追踪通道由 tracker 控制
                if _speaking and mid in _lower_set:
                    continue  # 说话时下半脸由 Audio2Face 控制
                if _blinking and mid in _EYELID_SET:
                    continue  # 眨眼时眼睑由眨眼逻辑控制，共情让出
                angle = float(smooth_vec[i])
                angle = clamp_angle(mid, angle)
                prev = last_written_angles.get(mid)
                if prev is not None and abs(angle - prev) < 1.0:
                    continue
                last_written_angles[mid] = angle
                pca, ch = get_hardware_target(mid, pcas)
                if pca:
                    try:
                        pulse = angle_to_pulse_us(angle, 0.0, 180.0, 600.0, 2400.0)
                        pca.set_servo_pulse_us(ch, pulse)
                    except OSError:
                        pass
            # ── 随机眨眼（高优先级，共情让出眼睑控制） ──
            if time.time() > _blink_t:
                _BLINK_EYES = {4:120, 5:100, 24:20, 25:60}  # 闭眼位置
                _blink_open = {}  # 用中位值作为回位参考
                for _ch, (_min,_max,_mid) in TABLE_V_CONFIG.items():
                    if _ch in _BLINK_EYES:
                        _blink_open[_ch] = _mid
                # 闭眼
                for _ch,_cp in _BLINK_EYES.items():
                    _pca,_lc = get_hardware_target(_ch, pcas)
                    if _pca:
                        _pca.set_servo_pulse_us(_lc, angle_to_pulse_us(_cp,0,180,600,2400))
                _blinking = True
                time.sleep(0.12)
                # 回到原位
                for _ch,_op in _blink_open.items():
                    _pca,_lc = get_hardware_target(_ch, pcas)
                    if _pca:
                        _pca.set_servo_pulse_us(_lc, angle_to_pulse_us(_op,0,180,600,2400))
                _blinking = False
                _blink_t = time.time() + _rnd.uniform(3, 6)
            # ── 日志 + 写共享状态 ──
            now = time.time()
            if emo7 is not None:
                write_vision_state(
                    emotion=EMO_LABELS[np.argmax(emo7)],
                    confidence=float(emo7[np.argmax(emo7)]),
                    face_detected=True,
                    user_identity=best_name if best_name else None,
                )
            if now - last_log >= 1.0 and emo7 is not None:
                top_emo = EMO_LABELS[np.argmax(emo7)]
                print(f"[{best_name:8s}] {top_emo:8s} {emo7[np.argmax(emo7)]:.2f} sim={best_sim:.2f} buf={len(emotion_buffer)} err=({error_x:.0f},{error_y:.0f})")
                last_log = now
            # ── 显示窗口 ──
            if SHOW_WINDOW:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if registered_found and best_box is not None:
                    x1,y1,x2,y2 = map(int, best_box)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f"{best_name} {best_sim:.2f}", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                if last_emo7 is not None:
                    top_emo = EMO_LABELS[np.argmax(last_emo7)]
                    label_y = y2 + 20 if (registered_found and best_box is not None) else 70
                    cv2.putText(frame, f"{top_emo} {last_emo7[np.argmax(last_emo7)]:.2f}",
                                (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                # BS 可视化
                bs_items = [
                    ("mouthSmile",  max(display_bs[44], display_bs[45])),
                    ("jawOpen",     display_bs[25]),
                    ("browDown",    max(display_bs[1], display_bs[2])),
                    ("browInnerUp", display_bs[3]),
                    ("mouthFunnel", display_bs[27]),
                    ("eyeBlink",    max(display_bs[9], display_bs[10])),
                    ("eyeWide",     max(display_bs[21], display_bs[22])),
                    ("mouthFrown",  max(display_bs[30], display_bs[31])),
                ]
                by = 24
                cv2.putText(frame, "-- Empathy BS --", (w-230, by),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                for key, val in bs_items:
                    by += 19
                    bw = int(val * 100)
                    clr = (0,255,255) if val > 0.3 else (100,100,100)
                    cv2.putText(frame, f"{key}:", (w-230, by),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
                    cv2.rectangle(frame, (w-135, by-10), (w-135+min(bw,100), by-2), clr, -1)
                    cv2.putText(frame, f"{val:.2f}", (w-28, by),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1)
                if face_detected:
                    # 显示追踪十字线（使用平滑后的位置）
                    disp_cx = smooth_cx if smooth_cx is not None else face_cx
                    disp_cy = smooth_cy if smooth_cy is not None else face_cy
                    cv2.circle(frame, (int(disp_cx), int(disp_cy)), 5, (255,0,0), 1)
                    cv2.line(frame, (cx_center, cy_center), (int(disp_cx), int(disp_cy)), (255,255,0), 1)
                if SHOW_WINDOW:
                    _panel = render_panel(frame.shape[0], fps, face_detected, emo7, EMO_LABELS, best_name, _speaking)
                    cv2.imshow("Morpheus", np.hstack([frame, _panel]))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            # 限帧 25fps
            elapsed = time.time() - frame_start
            if elapsed < frame_target:
                time.sleep(frame_target - elapsed)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        cap.release()
        rkn_face.release(); rkn_mbf.release(); rkn_emo.release()
        tracker.cleanup()
        for p in pcas.values(): p.close()
        print("Done.")
if __name__ == "__main__":
    main()
