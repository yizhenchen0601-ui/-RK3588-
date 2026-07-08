"""
机器人头部系统 — 统一配置
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")

# ============ ASR ============
ASR_MODEL_TYPE = "14m_fp32"
ASR_DIR = os.path.join(MODELS_DIR, "asr_model")
ASR_SAMPLE_RATE = 16000

# ============ LLM ============
LLM_MODEL = os.path.join(MODELS_DIR, "lingxin_1.5b.rkllm")
LLM_DIR = "/home/elf/rknn-llm/examples/rkllm_api_demo/deploy/install/demo_Linux_aarch64"
LLM_MAX_NEW_TOKENS = 2048
LLM_MAX_TOTAL_TOKENS = 4096

# ============ TTS ============
PIPER_MODEL = os.path.join(MODELS_DIR, "zh_CN-chaowen-medium.onnx")
TTS_SAMPLE_RATE = 22050

# ============ Audio ============
# 用 pacmd list-sinks / list-sources 查看你的设备名
AUDIO_SINK = "alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.analog-stereo"
MIC_SOURCE = "alsa_input.usb-Yundea_Technology_Yundea_8MICA_433039373936312E-01.multichannel-input"
CAPTURE_CHUNK_SEC = 0.1

# ============ AEC ============
ENABLE_AEC = True
AEC_FRAME_SIZE_MS = 10

# ============ Pipeline ============
MAX_SPEECH_SEC = 15.0
SILENCE_TIMEOUT_SEC = 1.2
PARTIAL_INTERVAL = 2
ENERGY_THRESHOLD = 20
ENERGY_THRESHOLD_HIGH = 80
NOISE_GATE = 0.003

# ============ System Prompt ============
SYSTEM_PROMPT = """你是灵心，一名温暖专业的心理咨询师。用户来找你时，先安抚情绪、表达共情，然后快速给出可行的建议或解决方案。语气温柔坚定，让用户感到被关心和支持。回答简短自然，像朋友一样谈心，不要列点。
"""

# ============ Audio2Face ============
A2F_DEVICE = "cpu"
A2F_LOWER_ONLY = True

# ============ Servo ============
SERVO_I2C_BUS = 4
SERVO_ADDRS = [0x40, 0x41, 0x42]
