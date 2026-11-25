"""Stationary RGB/GT release-transfer capture with explicit file-backed completion.

Some simulator releases write all sensor files but never complete SaveSensorData.
Only a parked, stationary, unchanged scene can use file-backed acceptance below.
This is not a moving-frame synchronization workaround or a control-interface test.
"""

from __future__ import annotations
import argparse, hashlib, json, math, random, signal, time
from datetime import datetime, timezone
from pathlib import Path
import grpc
from PIL import Image
from google.protobuf.json_format import MessageToDict
from data_collection.capture_grpc import MoraiCaptureClient
from morai.actor.actor_pb2_grpc import ActorStub
from morai.actor.actor_get_pb2 import GetAllActorsFilter
from morai.actor.actor_enum_pb2 import GEAR_MODE_P, VEHICLE_CONTROL_KEYBOARD
from morai.actor.actor_set_pb2 import (
    EnableActor,
    SetVelocityParam,
    SetTransformParam,
    VehicleGear,
    VehicleControlModeParam,
)
from morai.common.object_info_pb2 import ObjectInfo
from morai.common.type_pb2 import Empty
from morai.environment.environment_pb2_grpc import EnvironmentStub

now = lambda: datetime.now(timezone.utc)
message = lambda m: MessageToDict(
    m, preserving_proto_field_name=True, including_default_value_fields=True
)


def check(response):
    if response.status != 1:
        raise RuntimeError(str(response))


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def distance(a, b):
    return math.sqrt(
        sum((getattr(a.location, k) - getattr(b.location, k)) ** 2 for k in "xyz")
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--proto-dir", required=True)
    ap.add_argument("--host", default="host.docker.internal")
    ap.add_argument("--port", type=int, default=7789)
    ap.add_argument("--simulator-file-dir", required=True)
    ap.add_argument("--capture-rpc-timeout", type=float, default=3.0)
    a = ap.parse_args()
    plan = json.loads(a.plan.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    write(
        a.output / "collection_context.json",
        dict(
            plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            simulator_release=plan["simulator_release"],
            capture_rpc_timeout=a.capture_rpc_timeout,
            static_control="keyboard_park_without_actor_pause",
        ),
    )
    channel = grpc.insecure_channel(f"{a.host}:{a.port}")
    actor = ActorStub(channel)
    env = EnvironmentStub(channel)
    if a.capture_rpc_timeout <= 0:
        raise ValueError("positive capture RPC timeout required")
    client = MoraiCaptureClient(
        a.host,
        a.port,
        a.proto_dir,
        rpc_timeout_sec=a.capture_rpc_timeout,
        read_sim_time=False,
    )
    ego = ObjectInfo(object_type=1)
    ego.id.value = "Ego"
    initial = actor.GetActorState(ego, timeout=10)
    mode = actor.GetVehicleControlMode(ego, timeout=10)
    w0 = env.GetWeather(Empty(), timeout=10)
    h0 = env.GetTime(Empty(), timeout=10)
    actors = actor.GetAllActorsState(
        GetAllActorsFilter(vehicle=True, pedestrian=True, obstacle=True), timeout=10
    )
    assert (
        mode.mode == VEHICLE_CONTROL_KEYBOARD
        and initial.vehicle_state.gear_mode == GEAR_MODE_P
    )
    assert (
        len(actors.states) == 1
        and math.sqrt(sum(getattr(initial.global_velocity, k) ** 2 for k in "xyz"))
        < 0.1
    )
    write(
        a.output / "before.json",
        dict(
            utc=now().isoformat(),
            ego=message(initial),
            mode=message(mode),
            weather=message(w0),
            time=message(h0),
        ),
    )

    def hold():
        check(
            actor.SetVelocity(SetVelocityParam(actor_info=ego, velocity=0), timeout=10)
        )
        check(
            actor.SetVehicleGear(
                VehicleGear(actor_info=ego, gear=GEAR_MODE_P), timeout=10
            )
        )

    def move(pose):
        hold()
        request = SetTransformParam(actor_info=ego)
        for key in "xyz":
            setattr(request.transform.location, key, pose["location"][key])
            setattr(request.transform.rotation, key, pose["rotation"][key])
        desired = type(request.transform)()
        desired.CopyFrom(request.transform)
        for _ in range(5):
            check(actor.SetTransform(request, timeout=10))
            # Teleportation can switch this release to Auto/D. Restore keyboard
            # control and Park AFTER the teleport, before checking/capturing.
            check(
                actor.SetVehicleControlMode(
                    VehicleControlModeParam(
                        actor_info=ego, mode=VEHICLE_CONTROL_KEYBOARD
                    ),
                    timeout=10,
                )
            )
            hold()
            time.sleep(0.15)
            actual = actor.GetActorState(ego, timeout=10).transform
            yaw = (actual.rotation.z - desired.rotation.z + 180) % 360 - 180
            if distance(actual, desired) < 0.07 and abs(yaw) < 0.2:
                return
            request.transform.rotation.z -= yaw
        raise RuntimeError("pose did not settle")

    def guard():
        if now() >= datetime.fromisoformat(plan["hard_stop_utc"]):
            raise TimeoutError("collection deadline")

    def state():
        return dict(
            ego=message(actor.GetActorState(ego, timeout=10)),
            mode=message(actor.GetVehicleControlMode(ego, timeout=10)),
            weather=message(env.GetWeather(Empty(), timeout=10)),
            time=message(env.GetTime(Empty(), timeout=10)),
        )

    def settled_state():
        stop = time.monotonic() + 5
        while True:
            value = state()
            speed = math.sqrt(
                sum(value["ego"]["global_velocity"][k] ** 2 for k in "xyz")
            )
            if speed < 0.02:
                return value
            if time.monotonic() > stop:
                write(a.output / "unsettled_state.json", value)
                raise RuntimeError("parked vehicle did not settle below0.02m/s")
            time.sleep(0.25)

    def set_condition(setter, request, getter, field, value):
        try:
            check(setter(request, timeout=30))
            return "rpc_success"
        except grpc.RpcError as exc:
            if exc.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
                raise
            # Setting is idempotent, but do not resend before checking actual state.
            observed = getter(Empty(), timeout=30)
            if getattr(observed, field) != value:
                raise RuntimeError(
                    "condition RPC timed out and readback differs"
                ) from exc
            return "rpc_deadline_verified_readback"

    def interrupted(signum, frame):
        raise InterruptedError(str(signum))

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    completed = 0
    try:
        poses = list(plan["poses"])
        random.Random(plan["seed"]).shuffle(poses)
        for site in poses:
            guard()
            move(site["pose"])
            conditions = site.get(
                "conditions",
                [(w, h) for w in plan["weather_ids"] for h in plan["hours"]],
            )
            random.Random(plan["seed"] + site["id"]).shuffle(conditions)
            for weather, hour in conditions:
                guard()
                # Keep the physics loop running: some 26R1 transitions block
                # while paused. Park and before/after stability are mandatory.
                condition_receipt = dict(
                    weather=set_condition(
                        env.SetWeather,
                        type(w0)(type=weather, instantly=True),
                        env.GetWeather,
                        "type",
                        weather,
                    ),
                    time=set_condition(
                        env.SetTime,
                        type(h0)(hour=hour, instantly=True, data_only=False),
                        env.GetTime,
                        "hour",
                        hour,
                    ),
                )
                time.sleep(plan["settle_seconds"])
                before = settled_state()
                assert (
                    before["weather"]["type"] == plan["weather_names"][str(weather)]
                    and before["time"]["hour"] == hour
                )
                sid = f"release26_site{site['id']:02d}_w{weather}_h{hour:02d}"
                name = sid + "_" + now().strftime("%Y%m%dT%H%M%S%f")
                files = {
                    kind: a.output / "CAMERA_1" / (name + "_" + kind + ".png")
                    for kind in ("Intensity", "Semantic", "Depth", "Instance")
                }
                assert not any(p.exists() for p in files.values())
                started = now().isoformat()
                # Isolate each incomplete save RPC on its own channel. Never
                # accumulate timed-out calls on a long-lived capture channel.
                client.close()
                client = MoraiCaptureClient(
                    a.host,
                    a.port,
                    a.proto_dir,
                    rpc_timeout_sec=a.capture_rpc_timeout,
                    read_sim_time=False,
                )
                try:
                    try:
                        result = client.capture(name, a.simulator_file_dir)
                    except grpc.RpcError as exc:
                        if exc.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
                            raise
                        result = dict(
                            success=False,
                            rpc_code=str(exc.code()),
                            rpc_error=exc.details(),
                        )
                finally:
                    client.close()
                stop = time.monotonic() + 10
                previous = None
                while True:
                    sizes = [
                        p.stat().st_size if p.exists() else 0 for p in files.values()
                    ]
                    if min(sizes) > 0 and sizes == previous:
                        break
                    if time.monotonic() > stop:
                        raise TimeoutError("incomplete sensor files")
                    previous = sizes
                    time.sleep(0.2)
                after = state()
                with (a.output / "attempts.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            dict(
                                sample_id=sid,
                                name=name,
                                requested=site["pose"],
                                before=before,
                                after=after,
                                capture=result,
                                files={
                                    kind: str(path.relative_to(a.output))
                                    for kind, path in files.items()
                                },
                                started_utc=started,
                                completed_utc=now().isoformat(),
                                accepted=False,
                            ),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                assert (
                    before["weather"] == after["weather"]
                    and before["time"] == after["time"]
                )
                for observed in (before, after):
                    p = observed["ego"]["transform"]
                    reference = site["pose"]
                    assert (
                        math.dist(
                            [p["location"][k] for k in "xyz"],
                            [reference["location"][k] for k in "xyz"],
                        )
                        < 0.07
                    )
                    assert (
                        abs(
                            (p["rotation"]["z"] - reference["rotation"]["z"] + 180)
                            % 360
                            - 180
                        )
                        < 0.2
                    )
                    assert (
                        observed["ego"]["vehicle_state"]["gear_mode"] == "GEAR_MODE_P"
                    )
                    assert observed["mode"]["mode"] == "VEHICLE_CONTROL_KEYBOARD"
                    assert (
                        math.sqrt(
                            sum(
                                observed["ego"]["global_velocity"][k] ** 2
                                for k in "xyz"
                            )
                        )
                        < 0.02
                    )
                assert (
                    math.dist(
                        [before["ego"]["transform"]["location"][k] for k in "xyz"],
                        [after["ego"]["transform"]["location"][k] for k in "xyz"],
                    )
                    < 0.005
                )
                receipts = {}
                for kind, path in files.items():
                    with Image.open(path) as im:
                        assert im.size == (1280, 720)
                        im.verify()
                    receipts[kind] = dict(
                        path=str(path.relative_to(a.output)),
                        bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        mtime_ns=path.stat().st_mtime_ns,
                    )
                row = dict(
                    sample_id=sid,
                    name=name,
                    view="front",
                    site=f"release26_site{site['id']:02d}",
                    weather=weather,
                    hour=hour,
                    simulator_release=plan["simulator_release"],
                    requested=site["pose"],
                    before=before,
                    after=after,
                    started_utc=started,
                    completed_utc=now().isoformat(),
                    capture=result,
                    completion_basis="rpc_and_verified_files"
                    if result["success"]
                    else "verified_files_static_scene_rpc_deadline",
                    files=receipts,
                    condition_receipt=condition_receipt,
                    static_control="keyboard_park_without_actor_pause",
                )
                with (a.output / "manifest.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                completed += 1
            print("completed site", site["id"], "frames", completed, flush=True)
        assert completed == plan["expected_frames"]
        (a.output / "COMPLETE").write_text(now().isoformat() + "\n")
    finally:
        restored = {}
        for label, action in [
            ("pose", lambda: move(message(initial.transform))),
            (
                "resume",
                lambda: check(
                    actor.SetPause(
                        EnableActor(actor_info=ego, enable=False), timeout=10
                    )
                ),
            ),
            (
                "weather",
                lambda: check(
                    env.SetWeather(type(w0)(type=w0.type, instantly=True), timeout=15)
                ),
            ),
            (
                "time",
                lambda: check(
                    env.SetTime(
                        type(h0)(hour=h0.hour, instantly=True, data_only=False),
                        timeout=15,
                    )
                ),
            ),
            (
                "control",
                lambda: check(
                    actor.SetVehicleControlMode(
                        VehicleControlModeParam(actor_info=ego, mode=mode.mode),
                        timeout=10,
                    )
                ),
            ),
        ]:
            try:
                action()
                restored[label] = True
            except Exception as exc:
                restored[label] = repr(exc)
        restored.update(utc=now().isoformat(), completed_frames=completed)
        try:
            restored["observed"] = state()
        except Exception as exc:
            restored["readback_error"] = repr(exc)
        write(a.output / "after.json", restored)
        client.close()
        channel.close()
        assert all(
            restored[k] is True
            for k in ("weather", "time", "pose", "resume", "control")
        ), restored


if __name__ == "__main__":
    main()
