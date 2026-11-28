"""Bounded MORAI camera UDP assembly and immutable RGB archive (stdlib only).

Native timestamps are retained verbatim, not asserted to be exposure/UTC time.
Format reference: https://github.com/MORAI-Autonomous/MORAI-SensorExample
Run this receiver on the host targeted by the simulator's camera destination IP.
It never emits control packets. Network camera streaming excludes Capture Mode.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,os,shutil,socket,struct,time
from pathlib import Path


def utc():return datetime.now(timezone.utc).isoformat()
def atomic_json(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def jpeg_size(data):
    """Read JPEG SOF dimensions without a platform imaging dependency."""
    if not data.startswith(b'\xff\xd8')or not data.endswith(b'\xff\xd9'):raise ValueError('incomplete JPEG')
    pos=2
    while pos<len(data):
        if data[pos]!=255:raise ValueError('invalid JPEG marker')
        while pos<len(data)and data[pos]==255:pos+=1
        marker=data[pos];pos+=1
        if marker in [0xd8,0xd9,0x01]or 0xd0<=marker<=0xd7:continue
        if pos+2>len(data):raise ValueError('truncated JPEG marker')
        length=struct.unpack_from('>H',data,pos)[0]
        if length<2 or pos+length>len(data):raise ValueError('truncated JPEG segment')
        if marker in [0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf]:
            if length<8:raise ValueError('invalid JPEG SOF')
            h,w=struct.unpack_from('>HH',data,pos+3);return w,h
        if marker==0xda:break
        pos+=length
    raise ValueError('JPEG dimensions unavailable')


class CameraAssembler:
    def __init__(self,max_bytes=8*1024*1024,timeout=.5):
        self.max_bytes=max_bytes;self.timeout=timeout;self.count=Counter();self.reset()
    def reset(self):self.stamp=None;self.parts=[];self.next=0;self.size=0;self.started=None
    def feed(self,block,now=None):
        now=time.monotonic()if now is None else now;self.count['datagrams']+=1
        if self.started is not None and now-self.started>self.timeout:
            self.count['assembly_timeout']+=1;self.reset()
        if len(block)<21 or block[:3]!=b'MOR' or block[-2:]not in [b'AI',b'EI']:
            self.count['invalid_packet']+=1;self.reset();return None
        sec,nsec,index,size=struct.unpack_from('<IIii',block,3)
        if size<=0 or size>len(block)-21 or index<0 or nsec>=1000000000:
            self.count['invalid_packet']+=1;self.reset();return None
        stamp=(sec,nsec)
        if index==0:
            if self.parts:self.count['abandoned_frame']+=1
            self.reset();self.stamp=stamp;self.started=now
        if self.stamp!=stamp or self.next!=index:
            self.count['sequence_or_stamp_error']+=1;self.reset();return None
        self.parts.append(block[19:19+size]);self.size+=size;self.next+=1
        if self.size>self.max_bytes:
            self.count['oversize_frame']+=1;self.reset();return None
        if block[-2:]!=b'EI':return None
        data=b''.join(self.parts);meta=dict(native_seconds=sec,native_nanoseconds=nsec,fragments=self.next,assembly_seconds=now-self.started)
        self.reset()
        try:meta['width'],meta['height']=jpeg_size(data)
        except ValueError:self.count['invalid_jpeg']+=1;return None
        self.count['complete_frames']+=1;return data,meta


def receive(a):
    if a.seconds<=0 or a.capture_hz<=0 or a.max_bytes<=0 or a.min_free_bytes<0:raise ValueError('invalid receive limits')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);(root/'rgb').mkdir()
    assembler=CameraAssembler();counts=Counter();started=time.monotonic();last_saved=-float('inf');saved=0;total=0;reason='duration';peer=None
    context=dict(schema='camera-rgb-archive-1',started_utc=utc(),transport='morai_udp',camera='front',width=1280,height=720,
                 bind_host=a.host,bind_port=a.port,expected_peer_port=a.peer_port,capture_hz=a.capture_hz,max_seconds=a.seconds,
                 max_bytes=a.max_bytes,min_free_bytes=a.min_free_bytes,original_jpeg_preserved=True,
                 native_timestamp_semantics='raw_unverified_not_exposure_or_utc',uses_ground_truth=False,emits_vehicle_commands=False,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(root/'archive.json',context);sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,4*1024*1024);sock.settimeout(.25)
    try:
        sock.bind((a.host,a.port))
        with(root/'rgb.jsonl').open('a',encoding='utf-8',buffering=1)as manifest:
            while time.monotonic()-started<a.seconds:
                if(root/'STOP').exists():reason='stop_requested';break
                if saved and saved%20==0 and shutil.disk_usage(root).free<a.min_free_bytes:reason='low_disk';break
                try:block,peer=sock.recvfrom(65535)
                except socket.timeout:counts['receive_timeouts']+=1;continue
                if peer[1]!=a.peer_port:counts['unexpected_peer_packets']+=1;continue
                result=assembler.feed(block)
                if result is None:continue
                data,meta=result;now=time.monotonic()
                if(meta['width'],meta['height'])!=(1280,720):counts['wrong_geometry_frames']+=1;continue
                if now-last_saved<1./a.capture_hz:counts['sampling_skipped_frames']+=1;continue
                if total+len(data)>a.max_bytes:reason='archive_byte_limit';break
                if shutil.disk_usage(root).free<a.min_free_bytes:reason='low_disk';break
                sample='front_'+str(saved).zfill(8);name='rgb/'+sample+'.jpg';path=root/name;temp=path.with_suffix('.tmp')
                temp.write_bytes(data);temp.replace(path);digest=hashlib.sha256(data).hexdigest()
                row=dict(sample_id=sample,image=name,view='front',expected_sha256=dict(rgb=digest),received_utc=utc(),
                         receiver_elapsed_seconds=now-started,peer_host=peer[0],peer_port=peer[1],**meta)
                manifest.write(json.dumps(row,allow_nan=False)+'\n');last_saved=now;saved+=1;total+=len(data)
                atomic_json(root/'receiver_status.json',dict(status='running',saved_frames=saved,raw_bytes=total,assembly=dict(assembler.count),counters=dict(counts),updated_utc=utc()))
    except BaseException as e:
        reason='receiver_error';atomic_json(root/'receiver_error.json',dict(type=type(e).__name__,message=str(e),utc=utc()));raise
    finally:
        sock.close();receipt=dict(status='stopped'if reason!='receiver_error'else'failed',reason=reason,saved_frames=saved,raw_bytes=total,seconds=time.monotonic()-started,assembly=dict(assembler.count),counters=dict(counts),stopped_utc=utc(),archive_sha256=hashlib.sha256((root/'archive.json').read_bytes()).hexdigest())
        atomic_json(root/'receiver_status.json',receipt);atomic_json(root/'receiver_done.json',receipt)
    print(json.dumps(receipt))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=9291);p.add_argument('--peer-port',type=int,default=9290);p.add_argument('--seconds',type=float,default=300);p.add_argument('--capture-hz',type=float,default=2);p.add_argument('--max-bytes',type=int,default=1024**3);p.add_argument('--min-free-bytes',type=int,default=5*1024**3);receive(p.parse_args())
if __name__=='__main__':main()
