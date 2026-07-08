#!/bin/bash
pulseaudio --check 2>/dev/null || pulseaudio --start 2>/dev/null
pactl set-default-sink alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.analog-stereo 2>/dev/null
trap "echo; echo 正在退出...; kill %1 %2 2>/dev/null; exit 0" INT TERM
source /home/elf/miniconda3/etc/profile.d/conda.sh
conda activate robot
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"
MODE="${1:-voice}"
case "$MODE" in
    voice)
        export OMP_NUM_THREADS=3
        export TORCH_NUM_THREADS=3
        exec python3 -u pipeline.py
        ;;
    vision)
        # 视觉进程：限制 PyTorch 单线程避免与 OpenCV/NPU 争抢 CPU
        exec env DISPLAY=:0 OMP_NUM_THREADS=3 TORCH_NUM_THREADS=3 \
            python3 -u face_engine/empathy_track_servo_v3.py "$@"
        ;;
    core)
        echo "=== Core (Voice + Vision) ==="
        export OMP_NUM_THREADS=3
        export TORCH_NUM_THREADS=3
        DISPLAY=:0 taskset -c 0-3 env OMP_NUM_THREADS=3 TORCH_NUM_THREADS=3 \
            python3 -u face_engine/empathy_track_servo_v3.py &
        VISION_PID=$!
        python3 -u pipeline.py
        kill $VISION_PID 2>/dev/null
        ;;
    doa)
        exec python3 -u face_engine/doa_process.py
        ;;

    all)
        echo "=== 全系统 (DOA 引导) ==="
        export OMP_NUM_THREADS=3
        export TORCH_NUM_THREADS=3
        DISPLAY=:0 taskset -c 0-3 env OMP_NUM_THREADS=3 TORCH_NUM_THREADS=3             python3 -u face_engine/empathy_track_servo_v3.py &
        VISION_PID=$!
        python3 -u pipeline.py &
        PIPELINE_PID=$!
        echo "  模型预热中..."
        sleep 5
        python3 -u face_engine/doa_process.py --boot
        echo "  DOA 引导完成, ASR 启动"
        wait $PIPELINE_PID
        kill $VISION_PID 2>/dev/null
        ;;

esac
