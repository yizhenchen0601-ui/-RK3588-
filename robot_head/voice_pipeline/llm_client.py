"""LLM 客户端 — RKLLM (NPU 推理)"""

import io
import re
import time
import threading
import subprocess as sp
from config import *

# RKLLM 日志噪音过滤
RKLLM_NOISE = re.compile(
    r"^\s*[IWE]\s+rkllm:|^[\s\-_=|/.()]+$|^\s*(Stage|Prefill|Generate|Peak|rtf_avg|100%)"
)


class RKLLMClient:
    """通过子进程调用 rkllm_demo (Qwen2.5 0.5B on NPU)

    协议:
      - 写入 stdin: 用户文本 + 换行
      - stdout 读到 "robot:" 标记后开始产出回复字符
      - 遇到 "\nuser:" 标记表示回复结束
    """

    def __init__(self, system_prompt=SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.interrupt = threading.Event()

        if system_prompt:
            with open("/tmp/morpheus_sys_prompt.txt", "w", encoding="utf-8") as f:
                f.write("<|im_start|>system\n" + system_prompt.strip() + "\n<|im_end|>\n")

        self.proc = sp.Popen(
            ["./llm_demo", LLM_MODEL, str(LLM_MAX_NEW_TOKENS), str(LLM_MAX_TOTAL_TOKENS)]
            + (["/tmp/morpheus_sys_prompt.txt"] if system_prompt else []),
            cwd=LLM_DIR,
            env={"LD_LIBRARY_PATH": "./lib", "RKLLM_LOG_LEVEL": "0", "RKNN_NPU_CORE": "0,1"},
            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self.reader = io.TextIOWrapper(self.proc.stdout, encoding="utf-8", errors="replace")
        self._wait_for_marker("user:")
        print("  RKLLM ready")

    def _drain_stderr(self):
        while True:
            try:
                chunk = self.proc.stderr.read(4096)
                if not chunk:
                    break
            except:
                break

    def _wait_for_marker(self, marker: str, timeout: float = 60.0) -> bool:
        buf = ""
        t0 = time.time()
        while time.time() - t0 < timeout:
            c = self.reader.read(1)
            if not c:
                return False
            buf += c
            if marker in buf:
                return True
        return False

    def chat(self, text: str):
        """对话，返回流式生成器（逐字符产出）"""
        self.interrupt.clear()
        self.proc.stdin.write((text + "\n").encode())
        self.proc.stdin.flush()

        if not self._wait_for_marker("robot:"):
            yield "(no response)"
            return

        line_buf = ""
        full = ""
        while True:
            if self.interrupt.is_set():
                self._skip_to_next_prompt()
                return

            c = self.reader.read(1)
            if not c:
                break
            full += c

            line_buf += c
            if c == "\n":
                if RKLLM_NOISE.match(line_buf.strip()):
                    line_buf = ""
                    continue
                line_buf = ""

            if "\nuser:" in full:
                full = full[:full.rfind("\nuser:")]
                break

            yield c

    def _skip_to_next_prompt(self):
        buf = ""
        while True:
            c = self.reader.read(1)
            if not c:
                break
            buf += c
            if "\nuser:" in buf:
                break

    def clear_history(self):
        """RKLLM 子进程无历史管理，pass"""
        pass

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.kill()
        self.proc.wait(timeout=5)
