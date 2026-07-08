# Morpheus 机器人头部系统 (RK3588)

基于 Rockchip RK3588 的情感交互机器人头部，融合语音对话 + 音唇同步 + 表情控制 + 视觉追踪 + 声源定位。
对话引擎采用 **MiniSoulChat**（灵心）—— 以 Qwen2.5 1.5B 为基座，基于华南理工大学邢晓芬课题组 SoulChat 微调的心理咨询垂直领域模型，部署于 RK3588 NPU 端侧推理。

## 架构

```
┌─ 视觉进程 ────────────────────────────────────┐
│  Camera → RetinaFace → w600k(识别) → emotion   │
│     → EmpathyModel(19模板权重) → 52BS → 舵机    │
│     → PID追踪(5舵机: 6,23,30,31,32)            │
│              ↓                                 │
│         /tmp/morpheus_vision.json              │
└────────────────────────────────────────────────┘
                        │
┌─ 语音进程 ────────────────────────────────────┐
│  [耳] Mic → AEC → sherpa-onnx(ASR) → text     │
│  [脑] MiniSoulChat(RKLLM @NPU) → 逐句切分     │
│  [口] SpeechPipeline: TTS→WAV→UniTalk→MLP     │
│         → aplay(音频) + 30fps 舵机同步(表情)    │
│              ↓                                 │
│         /tmp/morpheus_voice.json               │
└────────────────────────────────────────────────┘
                        │
┌─ 声源定位 (可选) ─────────────────────────────┐
│  8MICA麦克风阵列(ch2,ch3) → TDOA → 脖子舵机转向 │
└────────────────────────────────────────────────┘
```

## 项目特色

- **🧠 自研端侧 LLM** — MiniSoulChat（灵心），基于 **Qwen2.5 1.5B** + **SoulChat**（华南理工大学邢晓芬课题组）微调，量化后通过 **RKLLM** 在 NPU 上本地推理，无需云端
- **🎯 心理咨询垂直领域** — 系统 prompt 定位为温暖专业的心理咨询师，结合视觉情绪感知实现共情对话
- **😊 共情闭环** — 视觉情绪识别 → EmpathyModel (19模板权重) → LLM prompt 注入 → 情感感知回复
- **👄 音唇同步** — UniTalker 音频驱动 52 维 blendshape → MLP 映射到 33 通道舵机，30fps 实时同步
- **👀 视觉追踪** — RetinaFace 人脸检测 + PID 控制 5 个舵机（脖子/眼睛）跟随人脸
- **🎤 声源定位** — 4 通道麦克风阵列 TDOA 算法，自动转向说话者
- **🔗 全本地化** — ASR / LLM / TTS / 人脸检测识别情绪 / 音唇同步全部在板端运行，不依赖任何云端服务

## 文件说明

```
robot_head/
├── run.sh               ← 一键启动 [voice|vision|core|doa|all]
├── config.py            ← 统一配置
├── shared_state.py      ← 跨进程通信（视觉↔语音，JSON文件）
├── pipeline.py          ← 语音主入口
├── sound_locator.py     ← 声源定位库 (TDOA)
│
├── voice_pipeline/      ← 语音模块
│   ├── audio_io.py      ← 音频采集/播放/AEC (PulseAudio)
│   ├── asr_engine.py    ← 流式语音识别 (sherpa-onnx)
│   ├── llm_client.py    ← MiniSoulChat RKLLM 客户端 (NPU子进程)
│   └── tts_client.py    ← Piper TTS 中文语音合成
│
├── face_engine/         ← 表情 + 视觉模块
│   ├── empathy_track_servo_v3.py  ← 视觉主进程 (检测/识别/情绪/追踪)
│   ├── audio2face.py    ← UniTalker + MLP (音频→52BS→舵机角度)
│   ├── speech_pipeline.py ← 音唇同步流水线 (TTS→播放+30fps舵机)
│   ├── servo_control.py ← PCA9685 舵机控制封装
│   ├── mor_servo_dev.py ← 舵机驱动 (CLI调试)
│   ├── set_start_bound.py ← 舵机限幅表
│   ├── empathy_model.py ← 共情Transformer模型 (情绪→19模板权重→52BS)
│   ├── empathy_data.py  ← 共情模板数据
│   ├── panel_renderer.py ← NPU负载面板渲染（叠在视频上）
│   ├── doa_process.py   ← 声源定位进程
│   └── vision_launcher.py ← 视觉模块启动辅助
│
├── unitalker/           ← UniTalker 音频驱动人脸模型
│   ├── infer_bs.py      ← 推理入口
│   └── models/          ← WavLM + UniTalker 网络定义
│
└── models/              ← 模型权重 (软链接)
    ├── lingxin_1.5b.rkllm          ← MiniSoulChat (Qwen2.5 1.5B + SoulChat微调, NPU)
    ├── UniTalker-B-D0-D7.pt        ← 音频→52BS
    ├── zh_CN-chaowen-medium.onnx   ← Piper TTS
    ├── upper/lower_face_bs2angle.pth ← BS→舵机角度 MLP
    ├── angle2bs_full.pth           ← 全脸BS映射
    ├── RetinaFace_mobile320.rknn   ← 人脸检测 (NPU)
    ├── w600k_mbf.rknn              ← 人脸识别 (NPU)
    ├── emotion_mobilenetv2_v9_nchw_fp16.rknn ← 情绪识别 (NPU)
    ├── empathy_best_new.pth        ← 共情Transformer
    ├── face_db.json                ← 已知人脸库
    └── inhouse_template.npy        ← 模板
```

## 启动

```bash
cd /home/elf/robot_head

./run.sh voice    # 仅语音对话（ASR+LLM+TTS+音唇同步）
./run.sh vision   # 仅视觉模块（人脸检测/识别/情绪/追踪）
./run.sh core     # 语音 + 视觉（推荐日常使用）
./run.sh doa      # 仅声源定位
./run.sh all      # 完整系统（语音+视觉+声源定位）
```

## 各模块独立测试

### 语音管线
```bash
cd ~/local_asr_pipeline && ./run.sh
```

### 音唇同步
```bash
cd ~/Morpheus
python3 tts_unitalk_servo.py --text "你好呀"
python3 tts_unitalk_servo.py --llm   # LLM 对话模式
```

### 舵机调试
```bash
cd ~/Morpheus
python3 mor_servo_dev.py --bus 4 --channel 0 get              # 读取舵机状态
python3 mor_servo_dev.py --bus 4 --channel 0 set --angle 90   # 转动舵机
```

### 声源定位测试
```bash
cd ~/robot_head
python3 sound_locator.py    # 拍手/说话测试方位估计
```

## 跨进程通信

视觉进程写入 `/tmp/morpheus_vision.json`，语音进程写入 `/tmp/morpheus_voice.json`，通过文件共享状态。

```python
from shared_state import write_vision_state, read_vision_state
from shared_state import write_voice_state, read_voice_state

# 视觉进程 → 情绪+人脸
write_vision_state(emotion="Happy", confidence=0.92, face_detected=True)

# 语音进程 → 说话状态
write_voice_state(speaking=True, status="speaking", user_text="你好", bot_text="你好呀")

# 读取（另一进程）
state = read_vision_state()
emotion_hint = get_emotion_prompt()  # → "用户当前情绪: Happy (92%)"
```

LLM 自动感知情绪：`"[用户当前情绪: Happy (92%)] 我今天好开心"` 注入到对话 prompt。

### 状态字段

| 字段 | 类型 | 写入者 | 读取者 |
|------|------|--------|--------|
| `emotion` | str | 视觉进程 | LLM prompt |
| `confidence` | float | 视觉进程 | LLM prompt |
| `face_detected` | bool | 视觉进程 | 待机/唤醒/主动搭话 |
| `user_identity` | str/None | 人脸识别 | LLM prompt |
| `face_box` | [x1,y1,x2,y2] | 视觉进程 | 显示/追踪 |
| `speaking` | bool | 语音管线 | 打断逻辑/DOA |
| `status` | str | 语音管线 | 系统状态 |
| `user_text` | str | 语音管线 | 对话记录 |
| `bot_text` | str | 语音管线 | 对话记录 |

## 硬件连接

| 设备 | 接口 |
|------|------|
| PCA9685 ×3 (舵机) | I2C bus 4, 地址 0x40/0x41/0x42 |
| 麦克风 (Yundea 8MICA) | USB Audio, 4通道 |
| 喇叭 | USB Audio (PulseAudio) |
| 摄像头 | Video device (index 21) |

**舵机分配**：表情控制 28 通道 + 追踪控制 5 通道 {6, 23, 30, 31, 32}

## 延迟特性

SpeechPipeline 的生产者-消费者模式确保句子 N 的 TTS+UniTalk 处理与句子 N-1 的播放重叠，只有第一句话有初始化延迟。

典型延迟：ASR ~500ms（端点检测）+ MiniSoulChat ~1-3s（NPU推理，流式逐句输出）+ TTS+UniTalk ~0.5-1s/句

## 依赖

- RK3588 开发板 + NPU (6 TOPS)
- Conda 环境 `robot`
- Piper TTS + g2pW
- sherpa-onnx (流式ASR)
- RKLLM (rkllm_demo，NPU LLM推理)
- WebRTC AEC (aec_audio_processing)
- UniTalker + WavLM
- rknn-toolkit2 / rknnlite (NPU模型推理)
- OpenCV + torch
- sounddevice + scipy (声源定位)
