"""
Panel renderer — 右侧信息面板，用 PIL 渲染中文
"""
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
from shared_state import read_voice_state
from face_engine.diag_stats import gc as gc2, ga as ga2

# ─── 缓存字体（只加载一次） ───
_FONT_CACHE = {}
def _get_font(size):
    if size not in _FONT_CACHE:
        try:
            from PIL import ImageFont
            _FONT_CACHE[size] = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", size)
        except:
            _FONT_CACHE[size] = None
    return _FONT_CACHE[size]

FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
PW = 340


_conv_buf = []

def render_panel(h, fps, face_detected, emo7=None, emo_labels=None, best_name="", speaking=False):
    """返回右侧面板 (numpy array, H x PW x 3)"""
    pf = _get_font(14)
    pf_big = _get_font(16)
    pf_sm = _get_font(12)

    img = Image.new("RGB", (PW, h), (35, 35, 40))
    if pf is None:
        return np.array(img)

    d = ImageDraw.Draw(img)
    y = 20

    # Title
    d.text((15, y), "MORPHEUS", font=pf_big, fill=(100, 170, 255))
    y += 38
    d.line([(15, y), (PW - 15, y)], fill=(80, 80, 80))
    y += 12

    # Status
    if speaking:
        sd, st = (220, 80, 80), "SPEAKING"
    elif emo7 is not None:
        sd, st = (220, 200, 50), "THINKING"
    elif face_detected:
        sd, st = (50, 200, 100), "ATTENTIVE"
    else:
        sd, st = (80, 80, 80), "IDLE"
    d.ellipse([(15, y), (27, y + 12)], fill=sd)
    d.text((33, y - 2), st, font=pf, fill=sd)
    if best_name:
        d.text((PW - 150, y - 2), best_name, font=pf_sm, fill=(180, 180, 180))
    y += 32
    d.line([(15, y), (PW - 15, y)], fill=(80, 80, 80))
    y += 12

    # Emotion
    if emo7 is not None and emo_labels:
        d.text((15, y), "EMOTION", font=pf_sm, fill=(180, 180, 180))
        y += 20
        te = emo_labels[np.argmax(emo7)]
        tv = float(emo7[np.argmax(emo7)])
        d.text((15, y), te, font=pf, fill=(255, 255, 255))
        bw = min(180, int(tv * 180))
        d.rectangle([(90, y), (90 + bw, y + 12)], fill=(50, 200, 100))
        d.text((PW - 80, y), f"{tv:.0%}", font=pf_sm, fill=(180, 180, 180))
        y += 22
    y += 5
    d.line([(15, y), (PW - 15, y)], fill=(80, 80, 80))
    y += 12

    # Conversation
    d.text((15, y), "CONVERSATION", font=pf_sm, fill=(180, 180, 180))
    y += 20
    try:
        vc = read_voice_state()
        ut = vc.get("user_text", "")
        bt = vc.get("bot_text", "")
        if ut:
            d.text((15, y), ">> " + ut[:60], font=pf_sm, fill=(100, 180, 255))
            y += 18
        if bt:
            d.text((15, y), "<< " + bt[:60], font=pf_sm, fill=(200, 180, 100))
            y += 18
        if not ut and not bt:
            d.text((15, y), "Waiting...", font=pf_sm, fill=(80, 80, 80))
            y += 18
    except:
        pass
    y += 5
    d.line([(15, y), (PW - 15, y)], fill=(80, 80, 80))
    y += 12

    # System
    d.text((15, y), "SYSTEM", font=pf_sm, fill=(180, 180, 180))
    y += 22
    npu_s = "--"
    try:
        npu_s = open("/sys/class/devfreq/fdab0000.npu/load").read().strip().split("@")[1].replace("Hz","")
    except:
        pass
    mem_s = "--"
    try:
        _mt = 0
        for l in open("/proc/meminfo"):
            if "MemTotal" in l: _mt = int(l.split()[1])
            if "MemAvailable" in l: mem_s = "{:.1f}G".format((_mt - int(l.split()[1]))/1024/1024)
    except: pass
    _npu_mhz = "--" if npu_s == "--" else str(int(int(npu_s)//1000000))
    d.text((15, y), f"FPS {fps:.0f}", font=pf_sm, fill=(255, 255, 255))
    y += 18
    d.text((15, y), f"NPU {_npu_mhz}MHz", font=pf_sm, fill=(255, 255, 255))
    y += 18
    _cput = str(round(int(open("/sys/class/thermal/thermal_zone0/temp").read())/1000,1)) + "°" if npu_s != "--" else "--"
    _vc2 = None
    try:
        _vc2 = read_voice_state()
        _ts2 = _vc2.get("token_speed", "?")
    except:
        _ts2 = "?"
    d.text((15, y), f"LLM {_ts2}t/s", font=pf_sm, fill=(255, 255, 255))
    y += 18
    _face_s = "YES" if face_detected else "NO"
    d.text((15, y), f"MEM {mem_s}", font=pf_sm, fill=(255, 255, 255))
    y += 18
    d.text((15, y), f"CPU {_cput}", font=pf_sm, fill=(255, 255, 255))
    y += 18
    _diag = ga2()
    d.text((15, y), "ASR #" + str(_diag["n"]) + "  last " + _diag["t"], font=pf_sm, fill=(255, 255, 255))
    y += 18
    d.text((15, y), "ON " + gc2(), font=pf_sm, fill=(255, 255, 255))
    y += 18
    try:
        _lt = _vc2.get("tts_latency", "")
        d.text((15, y), f"TTS+UT {_lt if _lt else chr(45)+chr(45)}s", font=pf_sm, fill=(255, 255, 255))
    except:
        pass
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
