import os, time, queue, threading, subprocess as sp
import numpy as np
from collections import deque
from config import *

class AECProcessor:
    def __init__(self, sample_rate=ASR_SAMPLE_RATE, frame_ms=AEC_FRAME_SIZE_MS):
        from aec_audio_processing import AudioProcessor
        self.ap = AudioProcessor(
            enable_aec=True, enable_ns=True, enable_agc=False, enable_vad=False
        )
        self.ap.set_stream_format(sample_rate, 1)
        self.ap.set_reverse_stream_format(sample_rate, 1)
        self.frame_size = self.ap.get_frame_size()
        self.sample_rate = sample_rate

    def process(self, audio_int16: np.ndarray, ref_deque: deque = None) -> np.ndarray:
        data = np.ascontiguousarray(audio_int16, dtype=np.int16).tobytes()
        out = []
        fs2 = self.frame_size * 2
        for i in range(0, len(data) - fs2 + 1, fs2):
            chunk = data[i : i + fs2]
            if ref_deque is not None:
                try:
                    self.ap.process_reverse_stream(ref_deque.popleft())
                except IndexError:
                    pass
            result = self.ap.process_stream(chunk)
            out.append(np.frombuffer(result, dtype=np.int16))
        return np.concatenate(out) if out else audio_int16

class AudioCapture:
    def __init__(self, source=MIC_SOURCE, rate=ASR_SAMPLE_RATE,
                 chunk_sec=CAPTURE_CHUNK_SEC, channels=1):
        self.source = source
        self.rate = rate
        self.chunk_size = int(rate * chunk_sec)
        self.channels = channels
        self._proc = None
        self._stop = threading.Event()

    def start(self):
        self._proc = sp.Popen(
            ["parec", "--device=" + self.source, "--rate=" + str(self.rate),
             "--channels=" + str(self.channels), "--format=s16le", "--raw"],
            stdout=sp.PIPE, stderr=sp.DEVNULL)
        return self

    def read(self) -> np.ndarray:
        raw = self._proc.stdout.read(self.chunk_size * 2)
        if not raw or len(raw) < self.chunk_size * 2:
            return np.array([], dtype=np.int16)
        return np.frombuffer(raw, dtype=np.int16)

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.kill()
            self._proc.wait()

TTS_VOLUME = 0.85

class AudioPlayer:
    def __init__(self, sink=AUDIO_SINK, rate=TTS_SAMPLE_RATE, channels=1,
                 aec: AECProcessor = None, volume: float = TTS_VOLUME):
        self.sink = sink
        self.rate = rate
        self.channels = channels
        self.aec = aec
        self.volume = max(0.05, min(1.0, volume))
        self._lock = threading.Lock()
        self._proc = None
        self._q = queue.Queue(maxsize=120)
        self._stop = threading.Event()
        self._playing_until = 0.0
        self.last_play_time = 0.0
        self.ref_deque = deque()

    def start(self):
        self._proc = sp.Popen(
            ["pacat", "--device=" + self.sink, "--rate=" + str(self.rate),
             "--channels=" + str(self.channels), "--format=s16le", "--raw"],
            stdin=sp.PIPE, stderr=sp.DEVNULL)
        threading.Thread(target=self._worker, daemon=True).start()
        return self

    def _worker(self):
        while not self._stop.is_set():
            try:
                audio = self._q.get(timeout=0.01)
            except queue.Empty:
                continue
            if audio is None:
                break
            try:
                self._proc.stdin.write(audio.tobytes() if isinstance(audio, np.ndarray) else audio)
                self._proc.stdin.flush()
            except BrokenPipeError:
                break

    def play(self, audio_int16: np.ndarray):
        duration = len(audio_int16) / self.rate
        now = time.time()
        self._playing_until = now + duration + 2.0
        self.last_play_time = now

        with self._lock:
            if self.aec:
                if self.rate != ASR_SAMPLE_RATE:
                    ratio = ASR_SAMPLE_RATE / self.rate
                    ref_len = int(len(audio_int16) * ratio)
                    ref = np.interp(
                        np.linspace(0, len(audio_int16) - 1, ref_len),
                        np.arange(len(audio_int16)),
                        audio_int16.astype(np.float32)
                    ).astype(np.int16)
                else:
                    ref = audio_int16
                self.ref_deque.clear()
                silence_frame = b"\x00" * (self.aec.frame_size * 2)
                for _ in range(4):
                    self.ref_deque.append(silence_frame)
                ref_raw = np.ascontiguousarray(ref, dtype=np.int16).tobytes()
                fs2 = self.aec.frame_size * 2
                for j in range(0, len(ref_raw) - fs2 + 1, fs2):
                    self.ref_deque.append(ref_raw[j:j + fs2])

            play_audio = (audio_int16.astype(np.float64) * self.volume).astype(np.int16)
            self._q.put(np.ascontiguousarray(play_audio, dtype=np.int16))

    def is_playing(self) -> bool:
        return time.time() < self._playing_until or not self._q.empty()

    def play_raw(self, audio_int16: np.ndarray):
        duration = len(audio_int16) / self.rate
        self._playing_until = time.time() + duration + 2.0
        self.last_play_time = time.time()
        if self.volume < 1.0:
            audio_int16 = (audio_int16.astype(np.float64) * self.volume).astype(np.int16)
        self._q.put(np.ascontiguousarray(audio_int16, dtype=np.int16))

    def interrupt(self):
        self._playing_until = 0.0
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except:
                pass
        sp.run(["pkill", "-f", "pacat.*local_asr_pipeline"],
               stdout=sp.DEVNULL, stderr=sp.DEVNULL)

    def drain(self):
        rem = self._playing_until - time.time()
        if rem > 0:
            time.sleep(rem)

    def stop(self):
        self._stop.set()
        self._q.put(None)
        self.interrupt()
        if self._proc:
            try:
                self._proc.wait(timeout=3)
            except:
                pass
