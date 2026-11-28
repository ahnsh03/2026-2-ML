"""Controlled ROS transport replay; NEVER a simulator/accuracy dataset.

Publish invalid-camera/geometry probes followed by a fixed public image burst.
Run only on a separate test topic or while the native camera publisher is off.
"""
from pathlib import Path
import argparse,json,time,hashlib
import cv2
import rospy
from sensor_msgs.msg import CompressedImage


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--topic',default='/ml_shadow_probe/compressed');p.add_argument('--image',type=Path,required=True);p.add_argument('--seconds',type=float,default=30);p.add_argument('--hz',type=float,default=40);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args(rospy.myargv()[1:]);assert 0<a.seconds<=120 and 0<a.hz<=100
    if a.receipt.exists():raise ValueError('probe receipt exists')
    image=cv2.imread(str(a.image))
    if image is None or image.shape!=(720,1280,3):raise ValueError('fixed public input required')
    encode=lambda x:cv2.imencode('.jpg',x,[cv2.IMWRITE_JPEG_QUALITY,90])[1].tobytes()
    good=encode(image);wrong=encode(cv2.resize(image,(640,480)))
    rospy.init_node('ml_shadow_transport_probe',anonymous=True);publisher=rospy.Publisher(a.topic,CompressedImage,queue_size=8)
    stop=time.monotonic()+10
    while publisher.get_num_connections()==0:
        if time.monotonic()>stop:raise RuntimeError('no test subscriber')
        time.sleep(.05)
    counts={};start=time.monotonic();seq=0
    def send(label,data,frame='Camera-1'):
        nonlocal seq
        msg=CompressedImage();msg.header.seq=seq;msg.header.stamp=rospy.Time.now();msg.header.frame_id=frame;msg.format='jpeg';msg.data=data;publisher.publish(msg);seq+=1;counts[label]=counts.get(label,0)+1
    for label,data,frame in [('wrong_camera',good,'Camera-2'),('wrong_geometry',wrong,'Camera-1'),('corrupt_image',b'broken','Camera-1')]:
        for _ in range(12):send(label,data,frame);time.sleep(.06)
    burst=time.monotonic()
    while time.monotonic()-burst<a.seconds:
        send('valid_replay',good);time.sleep(1/a.hz)
    time.sleep(.5)
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(dict(transport_test_only=True,exclude_from_training=True,source_image_sha256=hashlib.sha256(a.image.read_bytes()).hexdigest(),jpeg_sha256=hashlib.sha256(good).hexdigest(),topic=a.topic,counts=counts,seconds=time.monotonic()-start),indent=2)+'\n');print(counts)
if __name__=='__main__':main()
