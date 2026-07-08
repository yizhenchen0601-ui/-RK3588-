"""
Morpheus UI 渲染器 — 在 HDMI 屏幕上绘制美观的交互界面
独立模块，崩溃不影响主管线
依赖: PIL + Noto Sans CJK 字体
"""
import numpy as np
import cv2
import time
from PIL import Image, ImageDraw, ImageFont
from shared_state import read_voice_state

# ─── 常量 ───
PANEL_W = 400          # 右面板宽度
CANVAS_W = 1280 + PANEL_W  # 总宽度
CANVAS_H = 720         # 总高度
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
FONT_SIZE = 18
FONT_SMALL = 15
FONT_TITLE = 22

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
LGRAY = (200, 200, 200)
BG = (30, 30, 35)       # 面板背景
GREEN = (50, 200, 100)
YELLOW = (220, 200, 50)
RED = (220, 80, 80)
BLUE = (80, 140, 220)
ACCENT = (100, 180, 255)

EMO_COLORS = {
    "Angry": (220, 60, 60), "Disgust": (140, 100, 60),
    "Fear": (160, 80, 160), "Happy": (60, 200, 100),
    "Neutral": (160, 160, 160), "Sad": (80, 120, 200),
    "Surprise": (200, 180, 60)
}
EMO_ICONS = {
    "Angry": "\U0001f620", "Disgust": "\U0001f922", "Fear": "\U0001f628",
    "Happy": "\U0001f60a", "Neutral": "\U0001f610", "Sad": "\U0001f61e",
    "Surprise": "\U0001f62e"
}
STATUS_ICONS = {"idle": "\U0001f4a4", "listening": "\U0001f50a", "thinking": "\U0001f914", "speaking": "\U0001f399\ufe0f"}


class UIRenderer:
    """UI 渲染器 — 在摄像头画面右侧绘制信息面板"""

    def __init__(self):
        self.font = None
        self.font_s = None
        self.font_t = None
        self._load_fonts()
        self.prev_conversation = []  # [(role, text), ...]
        self.last_voice_state = {}

    def _load_fonts(self):
        try:
            self.font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
            self.font_s = ImageFont.truetype(FONT_PATH, FONT_SMALL)
            self.font_t = ImageFont.truetype(FONT_PATH, FONT_TITLE)
        except Exception:
            self.font = None

    def render(self, frame, fps, vision_state, best_name, emo7=None, last_emo7=None):
        """主渲染入口 — 返回带 UI 的完整画布"""
        try:
            return self._render(frame, fps, vision_state, best_name, emo7, last_emo7)
        except Exception as e:
            # 崩了也不影响原始画面
            return np.hstack([frame, np.zeros((CANVAS_H, PANEL_W, 3), dtype=np.uint8)])

    def _render(self, frame, fps, vs, best_name, emo7, last_emo7):
        h, w = frame.shape[:2]
        # 缩放摄像头画面到标准尺寸
        if w != 1280 or h != 720:
            frame = cv2.resize(frame, (1280, 720))

        # 创建右侧面板
        panel = np.full((720, PANEL_W, 3), BG, dtype=np.uint8)

        # 用 PIL 绘制文字
        if self.font:
            panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(panel_rgb)
            draw = ImageDraw.Draw(pil_img)

            y = 15
            # ─── 标题 ───
            y = self._text(draw, "\U0001f916 灵心", 15, y, WHITE, self.font_t) + 5
            y = self._line(draw, 15, y, PANEL_W - 30, y, GRAY) + 10

            # ─── 状态 ───
            voice = read_voice_state()
            speaking = voice.get("speaking", False)
            face = vs.get("face_detected", False)
            if speaking:
                status = "speaking"
            elif face and emo7 is not None:
                status = "thinking"
            elif face:
                status = "listening"
            else:
                status = "idle"
            status_color = {"speaking": RED, "thinking": YELLOW, "listening": GREEN, "idle": GRAY}
            status_text = {"speaking": "\U0001f534 说话中", "thinking": "\U0001f7e1 思考中",
                          "listening": "\U0001f7e2 聆听中", "idle": "\U000026ab 待机中"}
            self._text(draw, status_text[status], 15, y, status_color[status], self.font)
            if face and best_name:
                self._text(draw, f"\U0001f464 {best_name}", PANEL_W - 160, y, LGRAY, self.font_s)
            y += 35
            y = self._line(draw, 15, y, PANEL_W - 30, y, GRAY) + 10

            # ─── 情绪 (Top-3) ───
            if emo7 is not None:
                self._text(draw, "\U0001f4a1 情绪", 15, y, LGRAY, self.font_s)
                y += 22
                emo_labels = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
                top3 = np.argsort(emo7)[-3:][::-1]
                for idx in top3:
                    val = emo7[idx]
                    emo = emo_labels[idx]
                    color = EMO_COLORS.get(emo, WHITE)
                    icon = EMO_ICONS.get(emo, "")
                    bar_w = int(val * 180)
                    self._text(draw, f"{icon} {emo}", 15, y, color, self.font)
                    draw.rectangle([170, y + 2, 170 + bar_w, y + 16], fill=color + (200,))
                    self._text(draw, f"{val:.2f}", 355, y, GRAY, self.font_s)
                    y += 22
            else:
                self._text(draw, "\U0001f4a1 情绪 --", 15, y, GRAY, self.font)
                y += 22

            y = self._line(draw, 15, y, PANEL_W - 30, y, GRAY) + 10

            # ─── 对话 ───
            self._text(draw, "\U0001f4ac 对话", 15, y, LGRAY, self.font_s)
            y += 25
            # 从 shared_state 读 ASR 文本
            if voice.get("text", ""):
                self._draw_bubble(draw, f"\U0001f464 {voice.get('text', '')}", 15, y, BLUE, 132)
                y += 42 if len(voice.get('text', '')) > 20 else 35
            # Voice process writes last_response to vision.json maybe?
            # For now just show placeholder
            if face:
                self._draw_bubble(draw, "\U0001f916 我在听...", 15, y, ACCENT, 132)
                y += 35
            else:
                self._draw_bubble(draw, "\U0001f916 等人中...", 15, y, GRAY, 132)
                y += 35

            y = self._line(draw, 15, y, PANEL_W - 30, y, GRAY) + 10

            # ─── 系统信息 ───
            self._text(draw, "\U00002699 系统", 15, y, LGRAY, self.font_s)
            y += 22
            items = [
                ("\U0001f3af FPS", f"{fps:.0f}", GREEN),
                ("\U0001f9e0 NPU", "Core 2 (视觉)", BLUE),
                ("\U0001f4be 内存", f"{vs.get('_mem', 0):.1f}/7.7 GB", LGRAY),
                ("\U0001f50c 舵机", "24 通道", LGRAY),
            ]
            for icon, label, color in items:
                self._text(draw, f"{icon} {label}", 20, y, color, self.font)
                self._text(draw, self._trunc(label, 18), 220, y, LGRAY, self.font_s)
                y += 22

            # 转回 OpenCV
            panel = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 合并画面
        canvas = np.hstack([frame, panel])
        return canvas

    # ─── 辅助方法 ───
    def _text(self, draw, text, x, y, color, font):
        draw.text((x, y), text, fill=color + (255,), font=font)
        return y + font.size + 2

    def _line(self, draw, x1, y1, x2, y2, color):
        draw.line([(x1, y1), (x2, y2)], fill=color + (100,), width=1)
        return y2

    def _draw_bubble(self, draw, text, x, y, color, max_w):
        """画对话气泡"""
        tw = len(text) * 9
        bw = min(tw + 20, max_w)
        draw.rounded_rectangle([x, y, x + bw, y + 30], radius=8, fill=color + (60,))
        self._text(draw, text, x + 8, y + 4, WHITE, self.font)

    def _trunc(self, text, max_len):
        return text[:max_len] + ".." if len(text) > max_len else text
