"""
舵机控制 — PCA9685 封装
从 tts_unitalk_servo.py 提取
"""
import time
import threading
import subprocess
import numpy as np

BASE = __file__
from face_engine.mor_servo_dev import PCA9685, angle_to_pulse_us

# PCA9685 地址（I2C bus 4 上检测到 0x40, 0x41, 0x42）
PCA_ADDRS = [0x40, 0x41, 0x42]
I2C_BUS = 4

# 舵机限幅表：channel -> (min_angle, max_angle, mid_angle)
TABLE_V_CONFIG = {
    0:  (10.0,   90.0,  90.0),
    1:  (50.0,  100.0, 50.0),
    2:  (60.0,  105.0, 90.0),
    3:  (70.0,  100.0,  75.0),
    4:  (20.0,  120.0, 40.0),
    5:  (70.0,  100.0, 100.0),
    6:  (30.0,  110.0,  60.0),
    7:  (54.0,  90.0,  90.0),
    8:  (45.0,  135.0, 135.0),
    9:  (70.0,  115.0,  70.0),
    10: (70.0,  90.0, 70.0),
    11: (70.0,  100.0,  70.0),
    12: (90.0,  140.0, 135.0),
    13: (45.0,  135.0, 130.0),
    14: (105.0,  150.0, 135.0),
    15: (60.0,  90.0, 90.0),
    16: (125.0,  150.0, 130.0),
    17: (60.0,  115.0, 100.0),
    18: (45.0,  135.0, 70.0),
    19: (50.0,  120.0, 50.0),
    20: (20.0,  62.0,  62.0),
    21: (45.0,  135.0, 70.0),
    22: (30.0,  100.0,  100.0),
    23: (55.0,  95.0,  75.0),
    24: (20.0,   60.0,  50.0),
    25: (60.0,  150.0,  120.0),
    26: (65.0,  100.0, 90.0),
    27: (60.0,  115.0, 100.0),
    28: (0.0,   40.0,  0.0),
    29: (0.0,   90.0, 0.0),
    30: (110.0,  165.0, 155.0),
    31: (20.0,  75.0, 24.0),
    32: (0, 180.0, 90),
}


class ServoController:
    """舵机控制器 — 管理多片 PCA9685"""

    def __init__(self, bus=I2C_BUS, addrs=PCA_ADDRS):
        self.bus = bus
        self.addrs = addrs
        self.pcas = {}
        self.lock = threading.Lock()
        self.initialized = False

    def initialize(self):
        for addr in self.addrs:
            self.pcas[addr] = PCA9685(bus_id=self.bus, address=addr, freq_hz=50)
        self.initialized = True
        print(f"  PCA9685: {len(self.pcas)}片 ({[hex(a) for a in self.addrs]})")

    def get_hardware_target(self, global_ch):
        """全局通道号 → (PCA9685实例, 本地通道)"""
        if 0 <= global_ch <= 15:
            return self.pcas.get(0x40), global_ch
        elif 16 <= global_ch <= 31:
            return self.pcas.get(0x41), global_ch - 16
        elif global_ch == 32:
            return self.pcas.get(0x42), global_ch - 32
        return None, None

    def clamp_angle(self, channel, angle):
        if channel in TABLE_V_CONFIG:
            min_a, max_a, _ = TABLE_V_CONFIG[channel]
            return max(min_a, min(max_a, angle))
        return max(0.0, min(180.0, angle))

    def write_angles(self, angles_dict, used_motors, default_angles):
        """写入舵机角度（线程安全）"""
        full = default_angles.copy()
        full.update(angles_dict)
        with self.lock:
                for ch in used_motors:
                    angle = self.clamp_angle(ch, full[ch])
                    pca, local_ch = self.get_hardware_target(ch)
                    if pca is None:
                        continue
                    pulse = angle_to_pulse_us(angle, 0, 180, 600, 2400)
                    pca.set_servo_pulse_us(local_ch, pulse)

    def play_sync(self, wav_path, all_angles, used_motors, default_angles, fps=30):
        """播放音频同时同步驱动舵机"""
        N = len(all_angles)
        duration = N / fps
        print(f"  ▶ 播放 {duration:.1f}s ({N}帧)")
        proc = subprocess.Popen(["aplay", wav_path],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.perf_counter()
        for i in range(N):
            target_t = t0 + i / fps
            sleep = target_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            self.write_angles(all_angles[i], used_motors, default_angles)
        proc.wait()

    def reset_to_default(self, used_motors, default_angles, speed=30):
        """缓慢回到中位"""
        for mid in used_motors:
            target = default_angles.get(mid, 90.0)
            self.write_angles({mid: target}, used_motors, default_angles)

    def close(self):
        for pca in self.pcas.values():
            try:
                pca.close()
            except Exception:
                pass
