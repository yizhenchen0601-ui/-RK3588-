# 机器人头部系统 (RK3588)

## 项目结构

```
robot_head/
├── config.py                    # 统一配置
├── pipeline.py                  # 主管线
├── shared_state.py              # 跨进程通信（视觉↔语音）
├── run.sh                       # 启动脚本
│
├── voice_pipeline/              # 语音管线
│   ├── audio_io.py              # AEC / 音频采集 / 播放
│   ├── asr_engine.py            # 流式 ASR (sherpa-onnx 14m_fp32)
│   ├── llm_client.py            # RKLLM 客户端 (调用 llm_demo_v3)
│   └── tts_client.py            # Piper TTS
│
├── face_engine/                 # 面部/表情引擎
│   ├── servo_control.py         # PCA9685 舵机控制
│   ├── speech_pipeline.py       # TTS→WAV→UniTalk→MLP→30fps舵机同步
│   ├── audio2face.py            # UniTalk 推理 + 面部角度计算
│   ├── empathy_track_servo_v3.py # PID 视觉追踪 + 情绪识别
│   ├── mor_servo_dev.py         # PCA9685 底层驱动
│   ├── panel_renderer.py        # 面板渲染
│   └── ui_renderer.py           # UI 渲染
│
├── unitalker/                   # UniTalk 音唇同步模型
│   ├── models/                  # 模型定义
│   ├── dataset/                 # 数据集工具
│   └── infer_bs.py              # 推理脚本
│
└── models/                      # 模型文件
    ├── asr_model/               # ASR 模型 (14m_fp32)
    ├── lingxin_1.5b.rkllm       # LLM 模型
    └── zh_CN-chaowen-medium.onnx # Piper TTS
```

## 架构

```
 Ear Thread:   mic → AEC → ASR(流式) → text queue
 Main Thread:  text → LLM(流式) → 逐句切分 → SpeechPipeline
 SpeechPipeline: TTS → WAV → UniTalk → MLP → angles → aplay + 舵机
```

## 硬件

| 组件 | 说明 |
|------|------|
| SoC | RK3588 (ARM64) |
| 麦克风 | Yundea 8MICA USB 麦克风阵列 |
| 喇叭 | C-Media USB 音频设备 |
| 舵机控制 | PCA9685 x 3 (I2C bus 4, 0x40/0x41/0x42) |
| 舵机 | 33路 |

## 启动

```bash
cd ~/robot_head
python pipeline.py
```

### 参数

- `--no-aec` — 关闭回声消除
- `--wav-dir=<目录>` — 保存 TTS 音频

## 关键配置 (config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| ASR_MODEL_TYPE | 14m_fp32 | ASR 模型 |
| LLM_MODEL | lingxin_1.5b.rkllm | LLM 路径 |
| ENABLE_AEC | True | 回声消除 |
| ENERGY_THRESHOLD | 20 | VAD 阈值 |
| ENERGY_THRESHOLD_HIGH | 80 | TTS 播放时 VAD 阈值 |
| NOISE_GATE | 0.003 | 底噪门限 |

## PID 追踪参数 (empathy_track_servo_v3.py)

```python
kp = 0.6    # 比例
ki = 0.001  # 积分
kd = 0.005  # 微分
```

## 跨进程通信

视觉进程写 `/tmp/morpheus_vision.json`，情绪/人脸状态
语音进程写 `/tmp/morpheus_voice.json`，对话文本/说话状态
