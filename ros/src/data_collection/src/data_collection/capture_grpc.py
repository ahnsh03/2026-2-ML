# -*- coding: utf-8 -*-
# 이관: 인지 코드 레포 d0922bb `src/data_collection/src/data_collection/capture_grpc.py` (원작성: 안승현)
"""Small, strict MORAI gRPC client used by the Capture Mode collector."""

from __future__ import print_function

import os
import sys
import time


class CaptureGrpcError(RuntimeError):
    pass


def resolve_proto_root(package_path, configured=""):
    """Find the generated MORAI Python protobuf package without copying it."""
    candidates = []
    if configured:
        candidates.append(configured)
    if os.environ.get("MORAI_GRPC_PROTO_ROOT"):
        candidates.append(os.environ["MORAI_GRPC_PROTO_ROOT"])
    # MORAI가 생성한 proto는 공개 레포에 넣지 않는다. 레포 밖에 두고 경로만 넘긴다.

    for candidate in candidates:
        path = os.path.abspath(os.path.expanduser(candidate))
        marker = os.path.join(path, "morai", "sensor", "sensor_pb2_grpc.py")
        if os.path.isfile(marker):
            return path
    raise CaptureGrpcError(
        "MORAI generated proto not found; set ~grpc_proto_root or MORAI_GRPC_PROTO_ROOT"
    )


class MoraiCaptureClient(object):
    def __init__(self, address, port, proto_root, rpc_timeout_sec=2.0):
        if proto_root not in sys.path:
            sys.path.insert(0, proto_root)
        try:
            import grpc
            from morai.common.type_pb2 import Empty
            from morai.sensor.sensor_data_save_config_pb2 import SensorDataSaveConfig
            from morai.sensor.sensor_pb2_grpc import SensorStub
            from morai.simulator.simulator_pb2_grpc import SimulatorStub
        except ImportError as exc:
            raise CaptureGrpcError(
                "gRPC runtime/protobuf import failed: {}. Use the ROS Docker image "
                "(docker/ros-noetic) or install docker/ros-noetic/requirements-grpc.txt".format(exc)
            )

        self._grpc = grpc
        self._empty_type = Empty
        self._request_type = SensorDataSaveConfig
        self._timeout = float(rpc_timeout_sec)
        target = "{}:{}".format(address, int(port))
        self._channel = grpc.insecure_channel(
            target,
            options=[("grpc.max_receive_message_length", 32 * 1024 * 1024)],
        )
        self._sensor = SensorStub(self._channel)
        self._simulator = SimulatorStub(self._channel)
        self.target = target

    def wait_ready(self, timeout_sec):
        try:
            self._grpc.channel_ready_future(self._channel).result(timeout=float(timeout_sec))
        except Exception as exc:
            raise CaptureGrpcError(
                "MORAI gRPC server {} not ready within {:.1f}s: {}".format(
                    self.target, float(timeout_sec), exc
                )
            )

    def _sim_time_raw(self):
        try:
            response = self._simulator.GetTimestamp(
                self._empty_type(), timeout=self._timeout
            )
            return int(response.value)
        except Exception:
            return None

    def capture(self, custom_name, file_dir=""):
        request = self._request_type()
        request.is_custom_file_name = True
        request.custom_file_name = custom_name
        request.file_dir = file_dir or ""
        sim_time_raw = self._sim_time_raw()
        started = time.monotonic()
        response = self._sensor.SaveSensorData(request, timeout=self._timeout)
        latency_ms = (time.monotonic() - started) * 1000.0
        status = int(response.status)
        return {
            "success": status == 1,
            "grpc_status": status,
            "grpc_description": str(getattr(response, "description", "")),
            "grpc_custom_message": str(getattr(response, "custom_message", "")),
            "rpc_latency_ms": round(latency_ms, 3),
            "morai_sim_time_raw": sim_time_raw,
            "morai_sim_time_unit": "us" if sim_time_raw is not None else None,
            "morai_sim_time_ns": (
                int(sim_time_raw) * 1000 if sim_time_raw is not None else None
            ),
        }

    def close(self):
        self._channel.close()
