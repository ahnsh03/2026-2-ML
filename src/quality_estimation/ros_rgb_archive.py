"""Thin ROS1 CompressedImage archive, separated from the CUDA Python runtime.

A single latest slot decouples the ROS callback from disk writes. All deliberate
sampling, sequence gaps and overwritten slots are counted separately. RGB only.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,threading,time,uuid
from .camera_udp import atomic_json,jpeg_size,utc


def main():
    import rospy
    from sensor_msgs.msg import CompressedImage
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--topic',default='/image_jpeg/compressed');p.add_argument('--frame-id',default='Camera-1');p.add_argument('--seconds',type=float,default=300);p.add_argument('--capture-hz',type=float,default=1);p.add_argument('--max-bytes',type=int,default=1024**3);p.add_argument('--min-free-bytes',type=int,default=5*1024**3);a=p.parse_args(rospy.myargv()[1:])
    if a.seconds<=0 or a.capture_hz<=0 or a.max_bytes<=0 or a.min_free_bytes<0:raise ValueError('invalid archive limits')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);(root/'rgb').mkdir()
    from collections import Counter
    counts=Counter();lock=threading.Lock();pending=[None];prior=[None];started=time.monotonic();total=0;saved=0;last_saved=-float('inf');reason='duration';error=None
    archive_id='rgb_'+uuid.uuid4().hex
    context=dict(archive_id=archive_id,schema='camera-rgb-archive-1',transport='ros1_compressed_image',topic=a.topic,frame_id=a.frame_id,camera='front',width=1280,height=720,
                 capture_hz=a.capture_hz,max_seconds=a.seconds,max_bytes=a.max_bytes,min_free_bytes=a.min_free_bytes,original_jpeg_preserved=True,
                 native_timestamp_semantics='ros_header_verbatim_not_calibrated_exposure_utc',uses_ground_truth=False,emits_vehicle_commands=False,
                 started_utc=utc(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(root/'archive.json',context)
    def callback(msg):
        with lock:
            counts['received_messages']+=1;seq=int(msg.header.seq)
            if prior[0]is not None:
                step=(seq-prior[0])%2**32
                if step==0:counts['repeated_sequences']+=1
                elif step>=2**31:counts['sequence_resets']+=1
                else:counts['sequence_gaps']+=step-1
            prior[0]=seq
            if pending[0]is not None:counts['latest_slot_overwrites']+=1
            pending[0]=(msg,utc(),time.monotonic())
    rospy.init_node('ml_shadow_rgb_archive',anonymous=True,disable_signals=True)
    subscriber=rospy.Subscriber(a.topic,CompressedImage,callback,queue_size=1,buff_size=16*1024**2)
    try:
        with(root/'rgb.jsonl').open('x',buffering=1)as stream:
            while time.monotonic()-started<a.seconds and not rospy.is_shutdown():
                if(root/'STOP').exists():reason='stop_requested';break
                with lock:item=pending[0];pending[0]=None
                if item is None:time.sleep(.01);continue
                msg,received,arrival=item
                if arrival-last_saved<1/a.capture_hz:counts['sampling_skipped_messages']+=1;continue
                if msg.header.frame_id!=a.frame_id:counts['wrong_frame_id']+=1;continue
                if 'jpeg'not in msg.format.lower()and 'jpg'not in msg.format.lower():counts['wrong_image_format']+=1;continue
                data=bytes(msg.data)
                try:w,h=jpeg_size(data)
                except ValueError:counts['invalid_jpeg']+=1;continue
                if(w,h)!=(1280,720):counts['wrong_geometry_frames']+=1;continue
                if total+len(data)>a.max_bytes:reason='archive_byte_limit';break
                if shutil.disk_usage(root).free<a.min_free_bytes:reason='low_disk';break
                sample=archive_id+'_'+str(saved).zfill(8);name='rgb/front_'+str(saved).zfill(8)+'.jpg';path=root/name;tmp=path.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(path)
                row=dict(sample_id=sample,image=name,view='front',expected_sha256=dict(rgb=hashlib.sha256(data).hexdigest()),received_utc=received,
                         receiver_elapsed_seconds=arrival-started,width=w,height=h,ros_header_seq=int(msg.header.seq),ros_frame_id=msg.header.frame_id,
                         ros_stamp_seconds=int(msg.header.stamp.secs),ros_stamp_nanoseconds=int(msg.header.stamp.nsecs),format=msg.format)
                stream.write(json.dumps(row,allow_nan=False)+'\n');total+=len(data);saved+=1;last_saved=arrival
                atomic_json(root/'receiver_status.json',dict(status='running',saved_frames=saved,raw_bytes=total,counters=dict(counts),updated_utc=utc()))
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e));reason='receiver_error';raise
    finally:
        subscriber.unregister();receipt=dict(status='failed'if error else'stopped',reason=reason,error=error,saved_frames=saved,raw_bytes=total,counters=dict(counts),seconds=time.monotonic()-started,stopped_utc=utc(),archive_sha256=hashlib.sha256((root/'archive.json').read_bytes()).hexdigest())
        atomic_json(root/'receiver_status.json',receipt);atomic_json(root/'receiver_done.json',receipt)
    print(json.dumps(receipt))
if __name__=='__main__':main()
