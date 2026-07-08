# 基于RK3588的智能交互仿生人头，实现多模态感知与实时动作响应

基于 Rockchip RK3588 的情感交互仿生人头，融合语音对话 + 音唇同步 + 表情控制 + 视觉追踪。

## 架构

```
 耳线程:  麦克风 → AEC → ASR(流式) → 文本队列
 主线程:  文本 → LLM(NPU) → 逐句切分 → SpeechPipeline
 SpeechPipeline: TTS → WAV → UniTalk → MLP → 舵机(30fps) + 音频播放
 视觉线程: USB摄像头 → 人脸检测 → PID追踪 + 情绪识别 → 注入LLM
```

对话引擎采用 **灵心 (LingXin)** — 基于 Qwen2.5 1.5B 微调的心理咨询垂直领域模型，部署于 RK3588 NPU 端侧推理，完全本地化运行。

## 硬件

| 组件 | 说明 |
|------|------|
| SoC | RK3588 |
| USB 摄像头 | 人脸检测 + 视觉追踪 |
| 麦克风 | USB 麦克风阵列 |
| 喇叭 | USB 音频设备 |
| 舵机控制 | PCA9685 x 3 (I2C bus 4, 0x40/0x41/0x42) |
| 舵机 | 3 路脖子 + 26 路脸部 = 29 路 |

## 项目结构

```
robot_head/
├── config.py                 # 统一配置
├── pipeline.py               # 语音主管线
├── shared_state.py           # 跨进程通信
├── run.sh                    # 启动脚本
│
├── voice_pipeline/
│   ├── audio_io.py           # AEC / 音频采集 / 播放
│   ├── asr_engine.py         # 流式 ASR (sherpa-onnx)
│   ├── llm_client.py         # 灵心 RKLLM 客户端 (NPU子进程)
│   └── tts_client.py         # Piper TTS
│
├── face_engine/
│   ├── servo_control.py      # PCA9685 舵机控制
│   ├── speech_pipeline.py    # TTS→WAV→UniTalk→MLP→舵机同步
│   ├── audio2face.py         # UniTalk 推理 + 角度计算
│   ├── empathy_track_servo_v3.py  # PID 视觉追踪 + 情绪
│   ├── mor_servo_dev.py      # 舵机驱动
│   ├── panel_renderer.py     # 面板渲染
│   ├── ui_renderer.py        # UI 渲染
│   ├── empathy_model.py      # 共情模型
│   └── empathy_data.py       # 共情模板数据
│
├── unitalker/                # UniTalk 音唇同步
│   ├── infer_bs.py
│   └── models/
│
└── models/                   # 模型文件
    ├── lingxin_1.5b.rkllm    # 灵心 (Qwen2.5 1.5B 微调, NPU)
    └── ...
```

## 启动

```bash
cd ~/robot_head
python pipeline.py            # 语音对话（ASR+LLM+TTS+音唇同步）
./run.sh core                 # 语音+视觉（推荐）
```

### 参数

- `--no-aec` — 关闭回声消除
- `--wav-dir=<目录>` — 保存 TTS 音频

## 关键配置 (config.py)

| 参数 | 说明 |
|------|------|
| ASR_MODEL_TYPE | ASR 模型 (14m_fp32) |
| ENABLE_AEC | 回声消除开关 |
| ENERGY_THRESHOLD | VAD 阈值 |
| ENERGY_THRESHOLD_HIGH | TTS 播放时 VAD 阈值 |
| NOISE_GATE | 底噪门限 |
| A2F_LOWER_ONLY | 仅下半脸音频驱动 |

## PID 追踪参数

```python
kp = 0.6    # 比例
ki = 0.001  # 积分
kd = 0.005  # 微分
```

## 跨进程通信

视觉进程写 `/tmp/morpheus_vision.json`（情绪/人脸状态）
语音进程写 `/tmp/morpheus_voice.json`（对话文本/说话状态）

## 依赖

- RK3588 + NPU
- sherpa-onnx (流式 ASR)
- RKLLM Runtime (NPU 推理)
- Piper TTS
- UniTalk (音唇同步)
- OpenCV + torch
- rknn-toolkit2 / rknnlite
