#!/home/elf/miniconda3/envs/robot/bin/python
"""DOA — 8MICA直通，有人脸或说话就不动"""
import os,sys,time,json,subprocess as sp
sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from face_engine.mor_servo_dev import PCA9685,angle_to_pulse_us

RATE=48000;CHUNK=int(RATE*0.04);NEEDED=10;THRESH=1500
sv=PCA9685(bus_id=4,address=0x42,freq_hz=50)
def turn(t):
    for p in np.arange(90,t,0.45 if t>90 else -0.45):
        sv.set_servo_pulse_us(0,angle_to_pulse_us(p,0,180,600,2400));time.sleep(0.04)
    sv.set_servo_pulse_us(0,angle_to_pulse_us(t,0,180,600,2400))
    for _ in range(30):
        if busy(): break
        sv.set_servo_pulse_us(0,angle_to_pulse_us(t,0,180,600,2400));time.sleep(0.1)

def busy():
    try:
        v=json.load(open("/tmp/morpheus_vision.json"))
        if v.get("face_detected") and v.get("face_box"): return True
    except: pass
    try:
        v=json.load(open("/tmp/morpheus_voice.json"))
        if v.get("speaking",False): return True
    except: pass
    return False

idle_t=time.time()
print("[DOA] 8MICA直通（不杀PA）")
proc=sp.Popen(["arecord","-D","plughw:4,0","-f","S16_LE","-r","48000","-c","4","-t","raw"],
              stdout=sp.PIPE,stderr=sp.DEVNULL)
try:
    while True:
        if busy():
            idle_t=time.time()
            time.sleep(1); continue
        if time.time()-idle_t<5:
            time.sleep(0.5); continue

        r=proc.stdout.read(CHUNK*4*2)
        if not r or len(r)<CHUNK*4*2:continue
        a=np.frombuffer(r,dtype=np.int16).reshape(-1,4)
        e2,e3=np.max(np.abs(a[:,2])),np.max(np.abs(a[:,3]))
        if max(e2,e3)>THRESH:
            vf=1
            for _ in range(NEEDED):
                r2=proc.stdout.read(CHUNK*4*2)
                if r2:
                    a2=np.frombuffer(r2,dtype=np.int16).reshape(-1,4)
                    if max(np.max(np.abs(a2[:,2])),np.max(np.abs(a2[:,3])))>THRESH:vf+=1
            if vf>=NEEDED and not busy():
                sd="right" if e3-e2<0 else "left" if abs(e3-e2)>150 else "center"
                print(f"[DOA] {sd}")
                turn(55 if sd=="left" else 125 if sd=="right" else 90)
                idle_t=time.time()
except:pass
finally:
    proc.terminate();sv.close()
    print("[DOA] 完成")
