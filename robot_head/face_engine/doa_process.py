#!/home/elf/miniconda3/envs/robot/bin/python
import os,sys,time,json,subprocess as sp,math,shutil
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
from scipy.signal import butter,lfilter,correlate
from face_engine.mor_servo_dev import PCA9685,angle_to_pulse_us
R=48000;CS=0.04;CH=int(R*CS);CN=4;CL,CR=2,3
ET=1200;CF=8;DD=150;MI=5;TC=2.5;SH=8.0
NM,NX,NM2=30,150,90;SA=0x42;SC=0;SS=2.0;PS=None

def _ea(f):
    if not hasattr(_ea,'bf'):
        _ea.bf,_ea.af=butter(5,[300/8000,3400/8000],btype='bandpass')
    a16=f[::3].astype(np.float32)
    cl=lfilter(_ea.bf,_ea.af,a16[:,CL])
    cr=lfilter(_ea.bf,_ea.af,a16[:,CR])
    el=float(np.max(np.abs(cl)));er=float(np.max(np.abs(cr)))
    if max(el,er)<400:return None,el,er
    corr=correlate(cl,cr,mode='full')
    dly=float(np.argmax(corr)-(len(cl)-1))
    ang=float(NM2)+float(np.degrees(np.arcsin(max(-1.0,min(1.0,dly*343.0/(0.035*16000))))))*0.8
    return max(float(NM),min(float(NX),ang)),el,er

class N:
    def __init__(s):
        s.p=PCA9685(bus_id=4,address=SA,freq_hz=50)
        s.ca=s.ta=float(NM2);s._s(s.ca)
    def _s(s,a):
        s.p.set_servo_pulse_us(SC,angle_to_pulse_us(max(NM,min(NX,a)),0,180,600,2400))
    def t(s,a):s.ta=max(NM,min(NX,float(a)))
    def tk(s):
        d=s.ta-s.ca
        if abs(d)<0.3:
            if s.ca!=s.ta:s.ca=s.ta;s._s(s.ca)
            return 0
        st=max(0.2,min(abs(d),SS*max(0.3,abs(d)/(NX-NM)*3)))
        s.ca+=(1 if d>0 else -1)*st;s._s(s.ca);return abs(d)>0.5
    def h(s):s._s(s.ca)
    def rs(s):s.ta=NM2
    def cl(s):s.p.close()

def bu():
    try:
        v=json.load(open('/tmp/morpheus_vision.json'))
        if v.get('face_detected')and v.get('face_box'):return 1
    except:pass
    try:
        v=json.load(open('/tmp/morpheus_voice.json'))
        if v.get('speaking',0)or v.get('status')in('thinking','recognizing'):return 1
    except:pass
    return 0

def vs():
    try:return json.load(open('/tmp/morpheus_vision.json'))
    except:return{}

def psf():
    global PS
    if PS:return PS
    try:
        o=sp.run(['pactl','list','sources','short'],capture_output=1,text=1,timeout=3).stdout
        for l in o.split('\n'):
            if'8MICA'in l or'Yundea'in l:PS=l.split()[1];return PS
    except:pass

def bm():
    _T0 = time.time()
    sr=psf()
    if sr:sp.run(['pactl','suspend-source',sr,'1'],stderr=sp.DEVNULL);print('  PA suspended, ALSA接管')
    if vs().get('face_detected') and os.path.getmtime('/tmp/morpheus_vision.json') > _T0:
        print('  Face OK, skip DOA')
        if sr:sp.run(['pactl','suspend-source',sr,'0'],stderr=sp.DEVNULL)
        return
    nk=N();fn=ca=0;fl=fr=500.0;ta=None;tt=0.0;cb=[]
    pr=sp.Popen(['arecord','-D','plughw:5,0','-f','S16_LE','-r','48000','-c','4','-t','raw'],stdout=sp.PIPE,stderr=sp.DEVNULL)
    print('  DOA boot... speak or make sound')
    try:
        while 1:
            fn+=1;raw=pr.stdout.read(CH*CN*2)
            if not raw or len(raw)<CH*CN*2:continue
            a=np.frombuffer(raw,dtype=np.int16).reshape(-1,CN)
            el=np.max(np.abs(a[:,CL]));er=np.max(np.abs(a[:,CR]))
            if ca<60:
                fl=fl*0.92+el*0.08;fr=fr*0.92+er*0.08;ca+=1
                if ca==60:print('  calib L={:.0f} R={:.0f}'.format(fl,fr))
                continue
            r=_ea(a)
            if r and r[0] is not None:
                ang=r[0];eln=r[1];ern=r[2];df=ern-eln
                if abs(df)<DD:ang2=NM2
                elif df>0:ang2=NM2-min(df/3000.0,1)*50
                else:ang2=NM2+min(abs(df)/3000.0,1)*50
                if max(eln,ern)>1200:
                    cb.append(ang2)
                    if len(cb)>CF:cb.pop(0)
                elif cb:cb.pop(0)
                if len(cb)>=CF:
                    aa=sum(cb)/len(cb)
                    if ta is None or abs(aa-ta)>5 or time.time()-tt>TC:
                        ta=aa;nk.t(ta);tt=time.time();cb=[]
                        print('  >> track {:.0f}deg'.format(aa))
            if fn%10==0:
                if vs().get('face_detected') and os.path.getmtime('/tmp/morpheus_vision.json') > _T0:
                    print('  Face detected! Handing over...')
                    time.sleep(3.0)
                    break
                sys.stdout.write('\r  servo {:.0f}deg face:none'.format(nk.ca));sys.stdout.flush()
            nk.tk()
    except KeyboardInterrupt:print('\n  DOA interrupted')
    finally:
        pr.terminate();time.sleep(0.05);nk.cl()
        print('  Handover to PID track')
        if sr:sp.run(['pactl','suspend-source',sr,'0'],stderr=sp.DEVNULL);print('  PA resumed, ASR starts')

def main():
    os.system('');print('\x1b[?25l\x1b[2J\x1b[H')
    print('  +--------------------------------------------+')
    print('  |   MORPHEUS DOA - Sound Source Radar        |')
    print('  |   ALSA direct | standalone | no conflict    |')
    print('  |   8MICA CH{}(L) CH{}(R) @ {}kHz              |'.format(CL+1,CR+1,R//1000))
    print('  +--------------------------------------------+\n')
    nk=N();fn=0;it=time.time();ta=None;tt=0.0;da=None;cb=[];fl=fr=500.0;ca=0
    pr=sp.Popen(['arecord','-D','plughw:5,0','-f','S16_LE','-r','48000','-c','4','-t','raw'],stdout=sp.PIPE,stderr=sp.DEVNULL)
    try:
        while 1:
            fn+=1;raw=pr.stdout.read(CH*CN*2)
            if not raw or len(raw)<CH*CN*2:time.sleep(0.01);continue
            a=np.frombuffer(raw,dtype=np.int16).reshape(-1,CN)
            el=np.max(np.abs(a[:,CL]));er=np.max(np.abs(a[:,CR]))
            if ca<60:
                fl=fl*0.92+el*0.08;fr=fr*0.92+er*0.08;ca+=1
                if ca%10==0:
                    p=ca*100//60;b='#'*(p//5)+'.'*(20-p//5)
                    sys.stdout.write('\r  \x1b[33mcalib [{}] {}%\x1b[0m'.format(b,p));sys.stdout.flush()
                if ca==60:print('\r  \x1b[92mcalib done L={:.0f} R={:.0f}\x1b[0m'.format(fl,fr))
                continue
            b=bu()
            if b:
                it=time.time();ta=None;cb=[];nk.tk();time.sleep(0.02);df=er-el
                st='\x1b[91m[ yield ]\x1b[0m'
            else:
                wt=time.time()-it
                if wt<MI:
                    st='\x1b[90m[ wait {:.1f}s ]\x1b[0m'.format(max(0,MI-wt));df=er-el
                else:
                    r=_ea(a)
                    if r and r[0] is not None:
                        ang=r[0];eln=r[1];ern=r[2];df=ern-eln;da=ang
                        if abs(df)<DD:ang2=NM2
                        elif df>0:ang2=NM2-min(df/3000.,1)*50
                        else:ang2=NM2+min(abs(df)/3000.,1)*50
                        if max(eln,ern)>1200:
                            cb.append(ang2)
                            if len(cb)>CF:cb.pop(0)
                        elif cb:cb.pop(0)
                        if len(cb)>=CF:
                            aa=sum(cb)/len(cb)
                            if ta is None or abs(aa-ta)>5 or time.time()-tt>TC:
                                ta=aa;nk.t(ta);tt=time.time();cb=[];st='\x1b[92m[ >>> {} ]\x1b[0m'.format(int(aa))
                            else:st='\x1b[92m[ lock {} ]\x1b[0m'.format(int(ta))
                        else:st='\x1b[93m[ conf {}% ]\x1b[0m'.format(int(len(cb)/CF*100))
                    else:
                        st='\x1b[90m[ scan ]\x1b[0m';df=er-el
                        if ta and time.time()-tt<SH:
                            if fn%10==0:nk.h()
                        elif ta:ta=None;nk.rs()
            nk.tk()
            if fn%2==0:
                buf=['\x1b[?25l\x1b[H','+----- DOA RADAR -----+',''];s=nk.ca
                if da is not None:
                    if abs(da-NM2)<5:dt='\x1b[92m[^front]\x1b[0m'
                    elif da<NM2:dt='\x1b[93m[<-left {}]\x1b[0m'.format(int(NM2-da))
                    else:dt='\x1b[91m[->right {}]\x1b[0m'.format(int(da-NM2))
                else:dt='\x1b[90m[-- ----]\x1b[0m'
                buf.append('  {} servo {:.0f}deg {}'.format(dt,s,st))
                for v,l,co in[(er,'R',39),(el,'L',220)]:
                    vv=min(v/8000.,1.);bw=20;fi=int(vv*bw)
                    bar='\x1b[38;5;{}m{}\x1b[0m'.format(co,'#'*fi+'.'*(bw-fi))
                    buf.append('  {} {} {:5.0f}'.format(l,bar,v))
                buf.append('  D:{:+d}'.format(int(df)))
                v2=vs();buf.append('')
                buf.append('  {} {} | Ctrl+C'.format('\x1b[92mface\x1b[0m' if v2.get('face_detected') else '\x1b[90mnone\x1b[0m',v2.get('emotion','--')))
                sys.stdout.write('\n'.join(buf));sys.stdout.flush()
    except KeyboardInterrupt:print('\n\x1b[?25h DOA exit\n')
    finally:pr.terminate();nk.rs();time.sleep(0.3);nk.cl();print('\x1b[?25h',end='');sys.stdout.flush()

if __name__=='__main__':
    if '--boot' in sys.argv:bm()
    else:main()
