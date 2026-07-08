"""
音唇同步流水线 — 原版 StreamingPipeline 架构
处理线程：TTS→WAV→UniTalk→MLP → play_queue
播放线程：play_queue → aplay + 30fps 舵机同步
"""
import time
import queue
import threading
import subprocess
import wave
import numpy as np
from config import TTS_SAMPLE_RATE, ASR_SAMPLE_RATE
from shared_state import write_voice_state
from face_engine.audio2face import compute_angles

FPS = 30


class SpeechPipeline:

    def __init__(self, tts, a2f, servo, player=None):
        self.tts = tts
        self.a2f = a2f
        self.servo = servo
        self.player = player  # 用于填 ref_deque（AEC 参考信号）
        self.proc_queue = queue.Queue()   # 文字输入
        self.play_queue = queue.Queue()   # (wav, angles)
        self._counter = 0
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()

        threading.Thread(target=self._process_worker, daemon=True).start()
        threading.Thread(target=self._play_worker, daemon=True).start()

    def put(self, text):
        if not text or not text.strip():
            return
        self._idle.clear()
        self.proc_queue.put(text)

    def wait_idle(self):
        self._idle.wait()

    def flush(self):
        """清空待处理队列，用于打断"""
        for _ in range(self.proc_queue.qsize()):
            try: self.proc_queue.get_nowait()
            except: pass
        for _ in range(self.play_queue.qsize()):
            try: self.play_queue.get_nowait()
            except: pass
    def stop(self):
        self._stop.set()
        self.proc_queue.put(None)
        self.play_queue.put(None)

    def _process_worker(self):
        while not self._stop.is_set():
            try:
                text = self.proc_queue.get(timeout=0.2)
            except queue.Empty:
                self._idle.set()
                continue
            if text is None:
                self.proc_queue.task_done()
                return

            i = self._counter
            self._counter += 1
            wav = f"/tmp/morpheus_tts_{i}.wav"

            t0 = time.perf_counter()

            audio_bytes = self.tts.synthesize(text)
            if not audio_bytes:
                self.proc_queue.task_done()
                continue

            audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
            with wave.open(wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(TTS_SAMPLE_RATE)
                wf.writeframes(audio_i16.tobytes())

            bs_52 = self.a2f.process(wav)
            angles = compute_angles(
                self.a2f.upper_model, self.a2f.lower_model, bs_52,
                self.a2f.upper_idx, self.a2f.lower_idx,
                self.a2f.motor_ranges, self.a2f.upper_motor_ids,
                self.a2f.lower_motor_ids, self.a2f.default_angles,
                self.a2f.device, self.a2f.lower_only,
            )

            t_proc = time.perf_counter() - t0
            from shared_state import write_voice_state
            write_voice_state(tts_latency=round(t_proc, 2))
            dur = len(bs_52) / FPS
            print(f"  \u2514 [{i}] TTS+UniTalk={t_proc:.2f}s 音频={dur:.1f}s")

            self.proc_queue.task_done()
            self.play_queue.put((wav, angles))

    def _play_worker(self):
        while not self._stop.is_set():
            try:
                item = self.play_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                self.play_queue.task_done()
                return
            wav, angles = item

            N = len(angles)
            duration = N / FPS
            write_voice_state(speaking=True)
            print(f"  \U0001f3b5 播放 {duration:.1f}s ({N}帧)")

            # 1. 填 AEC ref_deque（跟原版 AudioPlayer.play() 一样处理）
            if self.player and hasattr(self.player, 'ref_deque') and hasattr(self.player, 'aec'):
                import wave
                with wave.open(wav, "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                audio_i16 = np.frombuffer(frames, dtype=np.int16)
                aec = self.player.aec
                rate = wf.getframerate()
                if rate != ASR_SAMPLE_RATE:
                    ratio = ASR_SAMPLE_RATE / rate
                    ref_len = int(len(audio_i16) * ratio)
                    ref = np.interp(
                        np.linspace(0, len(audio_i16) - 1, ref_len),
                        np.arange(len(audio_i16)),
                        audio_i16.astype(np.float32)
                    ).astype(np.int16)
                else:
                    ref = audio_i16
                ref_deque = self.player.ref_deque
                ref_deque.clear()
                silence_frame = b"\x00" * (aec.frame_size * 2)
                for _ in range(4):
                    ref_deque.append(silence_frame)
                ref_raw = np.ascontiguousarray(ref, dtype=np.int16).tobytes()
                fs2 = aec.frame_size * 2
                for j in range(0, len(ref_raw) - fs2 + 1, fs2):
                    ref_deque.append(ref_raw[j:j + fs2])
                print(f"  [AEC] ref_deque: {len(ref_deque)}帧 ({len(ref_deque)*aec.frame_size/16000:.1f}s)")

            # 2. aplay 播音频（同步，保证 t0 对齐）
            import subprocess
            proc = subprocess.Popen(["aplay", wav],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 3. 30fps 舵机同步
            t0 = time.perf_counter()
            for idx, frame_angles in enumerate(angles):
                target_t = t0 + idx / FPS
                sleep = target_t - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                self.servo.write_angles(
                    frame_angles,
                    self.a2f.used_motors,
                    self.a2f.default_angles,
                )
            proc.wait()
            write_voice_state(speaking=False)
            self.play_queue.task_done()
