#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 이관: 인지 코드 레포 d0922bb `src/data_collection/scripts/capture_mode_collector_node.py` (원작성: 안승현)
"""Automatically trigger all MORAI sensors currently in Capture Mode.

The node never publishes vehicle controls. It therefore works unchanged while
an MPC node publishes /ctrl_cmd or while MORAI is in Game Wheel/Keyboard mode.
"""

from __future__ import print_function

import copy
import os
import threading
import time

import rospy
import rospkg
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus, SaveSensorData

from data_collection.capture_grpc import (
    CaptureGrpcError,
    MoraiCaptureClient,
    resolve_proto_root,
)
from data_collection.capture_mode import (
    CaptureRunWriter,
    CaptureSchedule,
    ctrl_message_to_dict,
    default_capture_data_root,
    ego_message_to_dict,
    make_run_id,
    sanitize_name,
    speed_mps_from_ego,
    utc_now_iso,
)


class RosTopicCaptureClient(object):
    """Publish MORAI's native /SaveSensorData command without touching control."""

    def __init__(self, topic):
        self.topic = topic
        self._publisher = rospy.Publisher(topic, SaveSensorData, queue_size=1)

    def wait_ready(self, timeout_sec):
        deadline = time.monotonic() + float(timeout_sec)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self._publisher.get_num_connections() > 0:
                return
            time.sleep(0.1)
        raise CaptureGrpcError(
            "no MORAI subscriber connected to {} within {:.1f}s".format(
                self.topic, float(timeout_sec)
            )
        )

    def capture(self, custom_name, file_dir=""):
        msg = SaveSensorData()
        msg.is_custom_file_name = True
        msg.custom_file_name = custom_name
        msg.file_dir = file_dir or ""
        started = time.monotonic()
        self._publisher.publish(msg)
        return {
            "success": True,
            "capture_backend": "ros_topic",
            "grpc_status": None,
            "rpc_latency_ms": round((time.monotonic() - started) * 1000.0, 3),
            "morai_sim_time_raw": None,
            "morai_sim_time_unit": None,
            "morai_sim_time_ns": None,
            "transport_note": "ROS publish has no application-level save acknowledgement",
        }

    def close(self):
        self._publisher.unregister()


class CaptureModeCollector(object):
    def __init__(self):
        self.hz = float(rospy.get_param("~capture_hz", 1.0))
        self.max_captures = int(rospy.get_param("~max_captures", 0))
        self.start_delay_sec = float(rospy.get_param("~start_delay_sec", 2.0))
        self.only_when_moving = bool(rospy.get_param("~only_when_moving", False))
        self.min_speed_mps = float(rospy.get_param("~min_speed_mps", 0.2))
        self.require_ego = bool(rospy.get_param("~require_ego", False))
        self.dry_run = bool(rospy.get_param("~dry_run", False))
        self.max_consecutive_failures = int(
            rospy.get_param("~max_consecutive_failures", 3)
        )
        self.control_source_timeout = float(
            rospy.get_param("~control_source_timeout_sec", 1.0)
        )
        self.file_dir = str(rospy.get_param("~morai_file_dir", ""))
        self._lock = threading.Lock()
        self._latest_ego = None
        self._latest_ego_wall = None
        self._latest_ctrl = None
        self._latest_ctrl_wall = None
        self._closed = False

        run_id = rospy.get_param("~run_id", "") or make_run_id("kcity_capture")
        self.run_id = sanitize_name(run_id, fallback="kcity_capture")
        data_root = rospy.get_param("~data_root", "") or default_capture_data_root()
        grpc_address = str(rospy.get_param("~grpc_address", "127.0.0.1"))
        grpc_port = int(rospy.get_param("~grpc_port", 7789))
        rpc_timeout = float(rospy.get_param("~rpc_timeout_sec", 2.0))
        self.backend_requested = str(rospy.get_param("~backend", "auto")).lower()
        if self.backend_requested not in ("auto", "grpc", "ros_topic"):
            raise ValueError("backend must be auto, grpc, or ros_topic")
        save_topic = str(rospy.get_param("~save_topic", "/SaveSensorData"))
        connect_timeout = float(rospy.get_param("~connect_timeout_sec", 5.0))

        metadata = {
            "capture_hz": self.hz,
            "max_captures": self.max_captures,
            "only_when_moving": self.only_when_moving,
            "min_speed_mps": self.min_speed_mps,
            "require_ego": self.require_ego,
            "grpc_target": "{}:{}".format(grpc_address, grpc_port),
            "capture_backend_requested": self.backend_requested,
            "save_topic": save_topic,
            "morai_file_dir": self.file_dir,
            "dry_run": self.dry_run,
            "ego_topic": str(rospy.get_param("~ego_topic", "/Ego_topic")),
            "ctrl_topic": str(rospy.get_param("~ctrl_topic", "/ctrl_cmd")),
            "capture_policy": "wall_clock_fixed_rate_no_catchup",
            "control_independent": True,
            "notes": "Capture Mode GT lives in MORAI SensorData; this run stores trigger/state lineage.",
        }
        self.writer = CaptureRunWriter(data_root, self.run_id, metadata)
        self.schedule = CaptureSchedule(self.hz, self.start_delay_sec)
        self.client = None

        ego_topic = metadata["ego_topic"]
        ctrl_topic = metadata["ctrl_topic"]
        self._ego_sub = rospy.Subscriber(
            ego_topic, EgoVehicleStatus, self._on_ego, queue_size=1
        )
        self._ctrl_sub = rospy.Subscriber(ctrl_topic, CtrlCmd, self._on_ctrl, queue_size=1)

        if self.hz > 2.0:
            rospy.logwarn(
                "capture_hz=%.2f exceeds the currently validated range; "
                "run 100-frame drop/file-integrity QA before keeping this data",
                self.hz,
            )

        self.backend_active = "dry_run"
        if not self.dry_run:
            try:
                grpc_error = None
                if self.backend_requested in ("auto", "grpc"):
                    try:
                        package_path = rospkg.RosPack().get_path("data_collection")
                        proto_root = resolve_proto_root(
                            package_path, str(rospy.get_param("~grpc_proto_root", ""))
                        )
                        self.client = MoraiCaptureClient(
                            grpc_address,
                            grpc_port,
                            proto_root,
                            rpc_timeout_sec=rpc_timeout,
                        )
                        self.client.wait_ready(connect_timeout)
                        self.backend_active = "grpc"
                    except Exception as exc:
                        grpc_error = exc
                        if self.client is not None:
                            self.client.close()
                            self.client = None
                        if self.backend_requested == "grpc":
                            raise
                        rospy.logwarn(
                            "gRPC Capture unavailable (%s); trying ROS topic %s",
                            exc,
                            save_topic,
                        )
                if self.client is None and self.backend_requested in ("auto", "ros_topic"):
                    self.client = RosTopicCaptureClient(save_topic)
                    try:
                        self.client.wait_ready(connect_timeout)
                    except Exception as ros_error:
                        if grpc_error is not None:
                            raise CaptureGrpcError(
                                "neither Capture transport is ready; gRPC: {}; ROS: {}".format(
                                    grpc_error, ros_error
                                )
                            )
                        raise
                    self.backend_active = "ros_topic"
            except Exception:
                if self.client is not None:
                    self.client.close()
                self.writer.close()
                raise

        rospy.on_shutdown(self.close)
        rospy.loginfo(
            "Capture Mode auto collector ready: %.2f Hz, run=%s, backend=%s, manifest=%s, dry_run=%s",
            self.hz,
            self.run_id,
            self.backend_active,
            self.writer.manifest_path,
            self.dry_run,
        )
        rospy.loginfo(
            "This node publishes no control commands; MPC and Game Wheel are both supported."
        )

    def _on_ego(self, msg):
        now = time.time()
        value = ego_message_to_dict(msg)
        with self._lock:
            self._latest_ego = value
            self._latest_ego_wall = now

    def _on_ctrl(self, msg):
        now = time.time()
        value = ctrl_message_to_dict(msg)
        with self._lock:
            self._latest_ctrl = value
            self._latest_ctrl_wall = now

    def _snapshot(self):
        now = time.time()
        with self._lock:
            ego = copy.deepcopy(self._latest_ego)
            ctrl = copy.deepcopy(self._latest_ctrl)
            ego_wall = self._latest_ego_wall
            ctrl_wall = self._latest_ctrl_wall
        ego_age = None if ego_wall is None else max(0.0, now - ego_wall)
        ctrl_age = None if ctrl_wall is None else max(0.0, now - ctrl_wall)
        ros_ctrl_recent = ctrl_age is not None and ctrl_age <= self.control_source_timeout
        return {
            "ego": ego,
            "ego_age_sec": None if ego_age is None else round(ego_age, 4),
            "ctrl_cmd": ctrl if ros_ctrl_recent else None,
            "ctrl_cmd_age_sec": None if ctrl_age is None else round(ctrl_age, 4),
            "control_source": "ros_ctrl" if ros_ctrl_recent else "manual_or_unknown",
        }

    def _motion_allows_capture(self, snapshot):
        ego = snapshot.get("ego")
        if ego is None:
            return not self.require_ego and not self.only_when_moving
        if self.only_when_moving and speed_mps_from_ego(ego) < self.min_speed_mps:
            return False
        return True

    def run(self):
        seq = 0
        consecutive_failures = 0
        while not rospy.is_shutdown():
            if self.max_captures > 0 and self.writer.succeeded >= self.max_captures:
                rospy.loginfo("max_captures=%d reached", self.max_captures)
                break
            remaining = self.schedule.remaining()
            if remaining > 0.0:
                time.sleep(min(remaining, 0.1))
                continue

            snapshot = self._snapshot()
            if not self._motion_allows_capture(snapshot):
                self.writer.mark_motion_skip()
                self.schedule.advance()
                rospy.loginfo_throttle(
                    5.0,
                    "capture waiting for ego motion (require_ego=%s min_speed=%.2f m/s)",
                    self.require_ego,
                    self.min_speed_mps,
                )
                continue

            custom_name = "{}_{:06d}".format(self.run_id, seq)
            seq += 1
            record = {
                "sequence": seq - 1,
                "custom_name": custom_name,
                "requested_at_utc": utc_now_iso(),
                "request_wall_time_ns": time.time_ns(),
                "ros_time_ns": int(rospy.Time.now().to_nsec()),
                "state": snapshot,
                "success": False,
            }
            try:
                if self.dry_run:
                    result = {
                        "success": True,
                        "capture_backend": "dry_run",
                        "grpc_status": None,
                        "rpc_latency_ms": 0.0,
                        "morai_sim_time_raw": None,
                        "morai_sim_time_unit": None,
                        "morai_sim_time_ns": None,
                        "dry_run": True,
                    }
                else:
                    result = self.client.capture(custom_name, self.file_dir)
                    result.setdefault("capture_backend", self.backend_active)
                record.update(result)
            except Exception as exc:
                record["error"] = "{}: {}".format(type(exc).__name__, exc)
                rospy.logerr("capture %s failed: %s", custom_name, record["error"])

            record["completed_at_utc"] = utc_now_iso()
            record["completed_wall_time_ns"] = time.time_ns()
            self.writer.write_capture(record)
            if record.get("success"):
                consecutive_failures = 0
                rospy.loginfo(
                    "capture ok seq=%06d sim_raw=%s unit=%s rpc=%.1fms source=%s",
                    seq - 1,
                    record.get("morai_sim_time_raw"),
                    record.get("morai_sim_time_unit"),
                    float(record.get("rpc_latency_ms") or 0.0),
                    snapshot["control_source"],
                )
            else:
                consecutive_failures += 1
                rospy.logerr(
                    "capture rejected seq=%06d status=%s consecutive_failures=%d",
                    seq - 1,
                    record.get("grpc_status"),
                    consecutive_failures,
                )
                if consecutive_failures >= self.max_consecutive_failures:
                    rospy.signal_shutdown("capture RPC failed repeatedly")
                    break
            self.schedule.advance()

        self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.client is not None:
            self.client.close()
        self.writer.close()
        rospy.loginfo(
            "capture run closed: root=%s attempted=%d success=%d failed=%d motion_skips=%d",
            self.writer.root,
            self.writer.attempted,
            self.writer.succeeded,
            self.writer.failed,
            self.writer.skipped_motion,
        )


def main():
    rospy.init_node("capture_mode_collector")
    try:
        collector = CaptureModeCollector()
        collector.run()
    except (CaptureGrpcError, ValueError, OSError) as exc:
        rospy.logfatal("Capture Mode collector could not start: %s", exc)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
