#!/usr/bin/env python3
"""Forward a ROS 1 front CompressedImage topic to GT-free RGB quality inference."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import stat
import threading

import rospy
from sensor_msgs.msg import CompressedImage

from data_collection.compressed_rgb import CompressedRGBSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/image_jpeg/compressed")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fifo", type=Path, help="Existing named pipe for a live stream_review reader")
    parser.add_argument("--limit", type=int, help="End after this many messages during a finite replay")
    args = parser.parse_args(rospy.myargv()[1:])
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.fifo is not None and not stat.S_ISFIFO(args.fifo.stat().st_mode):
        parser.error("--fifo must refer to a named pipe")
    rospy.init_node("quality_front_rgb", anonymous=True)
    lock = threading.Lock()
    stopped = threading.Event()
    errors = []
    with ExitStack() as resources:
        stream = resources.enter_context(args.fifo.open("w")) if args.fifo is not None else None
        writer = resources.enter_context(CompressedRGBSession(args.output_dir, rospy.resolve_name(args.topic)))

        def on_frame(message):
            with lock:
                if stopped.is_set():
                    return
                try:
                    writer.append(message.data, message.format, message.header.seq, message.header.stamp.to_nsec(),
                                  message.header.frame_id, stream)
                    if writer.rows == args.limit:
                        stopped.set()
                except Exception as exc:
                    errors.append(exc)
                    stopped.set()
            if stopped.is_set():
                rospy.signal_shutdown("RGB capture ended")

        subscriber = rospy.Subscriber(args.topic, CompressedImage, on_frame, queue_size=16, buff_size=8*1024*1024)
        try:
            rospy.spin()
        finally:
            subscriber.unregister()
            with lock:
                stopped.set()
        if errors:
            raise errors[0]
        if args.limit is not None and writer.rows != args.limit:
            raise RuntimeError("publisher session ended before the requested message count")


if __name__ == "__main__":
    main()
