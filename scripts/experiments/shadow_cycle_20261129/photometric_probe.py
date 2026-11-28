"""Bounded parked time-of-day settling probe; never accuracy or training data."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'roi9_drive_20261126'))
from collect import Session, Empty, GEAR_MODE_P, VEHICLE_CONTROL_KEYBOARD, norm, message, utc
from common.bridge import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--endpoint', default='host.docker.internal:7789')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    session = Session(SimpleNamespace(endpoint=a.endpoint, udp_relay=None,
                                      native_velocity_divisor=3.6))
    state = session.state()
    mode = session.actor.GetVehicleControlMode(session.ego, timeout=5)
    weather = session.environment.GetWeather(Empty(), timeout=5)
    clock = session.environment.GetTime(Empty(), timeout=5)
    if state.vehicle_state.gear_mode != GEAR_MODE_P or mode.mode != VEHICLE_CONTROL_KEYBOARD or norm(state.global_velocity) / 3.6 > .02:
        raise ValueError('requires standstill Keyboard/Park')
    initial = dict(ego=message(state), time=message(clock), weather_type=int(weather.type))
    (a.output / 'before.json').write_text(json.dumps(initial, indent=2) + '\n')
    (a.output / 'EXCLUDE_FROM_DATASETS.json').write_text(json.dumps(dict(
        purpose='Photometric settling diagnostic at one parked pose',
        exclude_from_accuracy_confirmation=True, exclude_from_training=True,
        no_GT=True, source_sha256=sha256_file(Path(__file__))), indent=2) + '\n')
    deadline = time.monotonic() + 35
    while not (a.archive / 'rgb.jsonl').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('ROS RGB archive not ready')
        time.sleep(.2)
    failure, events = None, []
    try:
        with (a.output / 'events.jsonl').open('x', buffering=1) as log:
            for hour in [13, 5, 6, 7, 13]:
                began = utc()
                receipt = session.set_condition(0, hour)
                event = dict(requested_utc=began, settled_utc=utc(), hour=hour,
                             sdk_readback=receipt, weather=message(session.environment.GetWeather(Empty(), timeout=5)))
                events.append(event)
                log.write(json.dumps(event) + '\n')
                print('Parked photometric hour', hour, flush=True)
                time.sleep(17)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        restored = session.restore(initial)
        session.channel.close()
        result = dict(finished_utc=utc(), events=len(events), failure=failure,
                      restoration=restored, source_sha256=sha256_file(Path(__file__)),
                      scope='One parked pose; RGB only; no frame-exposure UTC calibration or general visibility claim.')
        (a.output / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
        if not restored.get('verified'):
            raise RuntimeError('photometric probe restoration failed')


if __name__ == '__main__':
    main()
