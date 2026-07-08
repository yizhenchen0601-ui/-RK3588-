import os,time,json,threading
_T0=time.time()
_CDISP=""
_CFG="1.0s/1.5s"
_lu=""
_lt=0.0
_at="--"
_ac=0
_fr=0
def _m():
    global _fr,_CDISP,_lu,_lt,_at,_ac
    while 1:
        try:
            with open('/tmp/morpheus_voice.json')as f:v=json.load(f)
            if v.get('status')=='idle'and _fr==0:
                _fr=1;sec=int(time.time()-_T0);m,s=divmod(sec,60)
                _CDISP=(str(m)+'m'+str(s)+'s')if m else str(s)+'s'
            ut=v.get('user_text','')
            if ut and ut!=_lu:_ac+=1;_lt=time.time();_lu=ut
            if _lt>0:_at=str(int(time.time()-_lt))+'s'
        except:pass
        time.sleep(0.2)
threading.Thread(target=_m,daemon=1).start()
def gc():return _CDISP if _CDISP else 'loading'
def ga():return {'c':_CFG,'t':_at,'n':_ac}
