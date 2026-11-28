"""Start the stdlib UDP receiver independently of the calling WSL terminal."""
from pathlib import Path
import argparse,json,os,subprocess,sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--hz',type=float,required=True);a=p.parse_args()
    if os.name!='nt':raise RuntimeError('Windows launcher must run with py.exe')
    if not a.session.is_dir()or not(a.session/'camera_udp.py').is_file():raise ValueError('prepared receiver source missing')
    with(a.session/'receiver.log').open('xb')as log:
        child=subprocess.Popen([sys.executable,str(a.session/'camera_udp.py'),'--output',str(a.session/'archive'),'--seconds',str(a.seconds),'--capture-hz',str(a.hz)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,close_fds=True,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    (a.session/'receiver_launcher.json').write_text(json.dumps(dict(windows_pid=child.pid,detached=True),indent=2)+'\n');print(child.pid)
if __name__=='__main__':main()
