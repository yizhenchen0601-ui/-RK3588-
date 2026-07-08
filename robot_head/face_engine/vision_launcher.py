"""
视觉管线启动器 — 独立进程运行 empathy_track_servo_v3.py

用法:
  from face_engine.vision_launcher import VisionProcess
  vision = VisionProcess()
  vision.start()          # 后台启动
  vision.is_running()     # 检查是否在跑
  vision.stop()           # 安全停止

或者命令行:
  python3 -m face_engine.vision_launcher [--cam 21] [--no-window]
"""
import os
import sys
import time
import signal
import subprocess
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION_SCRIPT = os.path.join(BASE, "../Morpheus/empathy_track_servo_v3.py")


class VisionProcess:
    """视觉进程管理器 — 独立进程，不影响语音管线"""

    def __init__(self, cam=21, no_window=False):
        self.cam = cam
        self.no_window = no_window
        self.proc = None

    def start(self):
        """后台启动视觉进程"""
        if self.proc and self.proc.poll() is None:
            print("  视觉进程已在运行")
            return

        cmd = [
            "python3", VISION_SCRIPT,
            "--cam", str(self.cam),
        ]
        if self.no_window:
            cmd.append("--no-window")

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid  # 独立进程组，方便 kill
        )
        print(f"  视觉进程已启动 (PID {self.proc.pid})")

    def stop(self, timeout=3):
        """停止视觉进程"""
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=timeout)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            print("  视觉进程已停止")

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def returncode(self):
        return self.proc.poll() if self.proc else None


def main():
    parser = argparse.ArgumentParser(description="启动视觉追踪+情绪识别进程")
    parser.add_argument("--cam", type=int, default=21)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    # 直接执行 empathy_track_servo_v3.py
    cmd = [sys.executable, VISION_SCRIPT, "--cam", str(args.cam)]
    if args.no_window:
        cmd.append("--no-window")
    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
