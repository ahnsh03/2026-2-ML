"""Restore Keyboard/Park after the final UDP packet; never move or reset a scene."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math
from collect import Session,message,norm,VEHICLE_CONTROL_KEYBOARD,GEAR_MODE_P


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['relay-root','native-receipt','initial-state','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--endpoint',default='host.docker.internal:7789');a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous restoration receipt')
    stopped=a.relay_root/'relay_stopped.json'
    if not stopped.exists():raise ValueError('Stop the UDP sender before restoring SDK mode')
    native=read(a.native_receipt);age=(datetime.now(timezone.utc)-datetime.fromisoformat(native['utc'].replace('Z','+00:00'))).total_seconds()
    if not 0<=age<15 or native['gear']!=1 or abs(native['signed_speed_kph'])>.1 or native['control_commands_sent']!=0:
        raise ValueError('Fresh read-only native Park/standstill receipt required')
    initial=read(a.initial_state)
    if initial['mode']['mode']!='VEHICLE_CONTROL_KEYBOARD':raise ValueError('This recovery only restores a previously verified Keyboard session')
    a.udp_relay=None;session=Session(a)
    before=session.state();before_mode=session.actor.GetVehicleControlMode(session.ego,timeout=5)
    if before.vehicle_state.gear_mode!=GEAR_MODE_P or norm(before.global_velocity)/3.6>.02:
        raise ValueError('SDK vehicle is not parked/stationary')
    expected=initial['ego']['transform']['location'];xy=math.hypot(before.transform.location.x-expected['x'],before.transform.location.y-expected['y'])
    if xy>.1:raise ValueError('Pose already differs; do not hide this by relocating the vehicle')
    # A relay's final failsafe packet can select AUTO again. SDK mode restoration
    # must be the last control-mode writer, after the sender has stopped.
    session.hold();session.mode(VEHICLE_CONTROL_KEYBOARD)
    after=session.state();mode=session.actor.GetVehicleControlMode(session.ego,timeout=5)
    from morai.common.type_pb2 import Empty
    weather=session.environment.GetWeather(Empty(),timeout=5);clock=session.environment.GetTime(Empty(),timeout=5)
    verified=(mode.mode==VEHICLE_CONTROL_KEYBOARD and after.vehicle_state.gear_mode==GEAR_MODE_P
        and norm(after.global_velocity)/3.6<.02 and weather.type==initial['weather_type']and clock.hour==initial['time']['hour'])
    result=dict(status='verified'if verified else'failed',utc=datetime.now(timezone.utc).isoformat(),before_mode=message(before_mode),after_mode=message(mode),gear=after.vehicle_state.gear_mode,speed_mps=norm(after.global_velocity)/3.6,initial_pose_error_m=xy,weather=message(weather),time=message(clock),relay_stopped_sha256=sha(stopped),native_receipt_sha256=sha(a.native_receipt),initial_state_sha256=sha(a.initial_state),source_sha256=sha(Path(__file__)),scene_or_pose_reset=False,scope='Final SDK Keyboard/Park restoration after all UDP senders stop; zero velocity only, no throttle/steering drive or weather/time/pose change.')
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');session.channel.close()
    if not verified:raise RuntimeError('Final SDK restoration was not verified')
    print('Final Keyboard/Park restoration verified')


if __name__=='__main__':main()
