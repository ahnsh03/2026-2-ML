"""Bounded26R1 drive for RGB-shadow integration; no GT capture or quality control."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import argparse,json,math,signal,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'roi9_drive_20261126'))
from collect import Session,geometry,norm,message,check,utc,GEAR_MODE_D,GEAR_MODE_P,VEHICLE_CONTROL_AUTO_MODE,VEHICLE_CONTROL_KEYBOARD,Empty
from common.bridge import sha256_file

def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','output','udp-relay']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--endpoint',default='host.docker.internal:7789');p.add_argument('--hard-stop-utc',required=True);p.add_argument('--maximum-routes',type=int,default=6);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads(a.plan.read_text());args=SimpleNamespace(endpoint=a.endpoint,udp_relay=a.udp_relay,native_velocity_divisor=3.6)
    session=Session(args);weather=session.environment.GetWeather(Empty(),timeout=5);clock=session.environment.GetTime(Empty(),timeout=5);state=session.state();mode=session.actor.GetVehicleControlMode(session.ego,timeout=5)
    if mode.mode!=VEHICLE_CONTROL_KEYBOARD or state.vehicle_state.gear_mode!=GEAR_MODE_P or norm(state.global_velocity)/3.6>.02:raise ValueError('requires Keyboard/Park/standstill')
    initial=dict(ego=message(state),time=message(clock),weather_type=int(weather.type),mode=message(mode));save(a.output/'before.json',initial)
    save(a.output/'design.json',dict(source_sha256=sha256_file(Path(__file__)),plan_sha256=sha256_file(a.plan),source_session_sha256=sha256_file(Path(__file__).resolve().parents[1]/'roi9_drive_20261126/collect.py'),target_kph=plan['target_kph'],wire_steering_inverted_once=True,steering_range_deg=plan['steering_range_deg'],uses_quality_for_vehicle_control=False,captures_paired_GT=False,scope='Runtime drive verification. Frame/vehicle clocks are not exposure-synchronized; not independent accuracy/training data.'))
    stopped=False;results=[];failure=None
    def interrupt(sig,frame):raise InterruptedError(str(sig))
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,interrupt)
    try:
        for route in plan['routes'][:a.maximum_routes]:
            s0=session.place(route['start_pose']);points=np.asarray(route['points']);_,_,stations=geometry(points,[s0.transform.location.x,s0.transform.location.y]);session.mode(VEHICLE_CONTROL_AUTO_MODE);session.gear(GEAR_MODE_D);session.control()
            began=time.monotonic();controls=[];last_collision=0
            with(a.output/(route['route_id']+'_controls.jsonl')).open('x',buffering=1)as log:
                while True:
                    tick=time.monotonic()
                    if datetime.now(timezone.utc)>=datetime.fromisoformat(a.hard_stop_utc):raise TimeoutError('drive deadline')
                    if tick-began>45:raise TimeoutError('route timeout')
                    state=session.state();xy=np.array([state.transform.location.x,state.transform.location.y]);station,cte,_=geometry(points,xy);speed=norm(state.global_velocity)/3.6
                    if cte>.8 or speed>1.9:raise RuntimeError('tracking/speed bound')
                    if station>=stations[-1]-1.3:break
                    if tick-began>10 and station<1:raise RuntimeError('no measured progress')
                    if tick-last_collision>1:
                        actual=session.actor.GetActorState(session.ego,timeout=.4)
                        if any(obj and obj!='Ego'for obj in actual.vehicle_state.collision_objects):raise RuntimeError('SDK collision')
                        last_collision=tick
                    goal=min(station+4.,float(stations[-1]));target=np.array([np.interp(goal,stations,points[:,j])for j in [0,1]]);bearing=math.atan2(*(target-xy)[::-1]);yaw=math.radians(state.transform.rotation.z);alpha=(bearing-yaw+math.pi)%(2*math.pi)-math.pi
                    delta=math.atan2(2*2.7*math.sin(alpha),max(np.linalg.norm(target-xy),1.5));steer=float(np.clip(delta/math.radians(plan['steering_range_deg']),-.5,.5));target_speed=plan['target_kph']/3.6
                    throttle=float(np.clip(.14+.16*(target_speed-speed),0,.3));brake=.25 if speed>target_speed+.35 else 0.;session.control(throttle=throttle if not brake else 0,brake=brake,steer=steer)
                    r=dict(utc=utc(),elapsed=tick-began,station_m=station,cte_m=cte,speed_mps=speed,steering_left_positive=steer,state=message(state),udp_state=session.last_udp_state)
                    controls.append(r);log.write(json.dumps(r)+'\n');time.sleep(max(0,.06-(time.monotonic()-tick)))
            session.hold();end=session.state();result=dict(route_id=route['route_id'],control_rows=len(controls),duration_seconds=time.monotonic()-began,distance_m=station,maximum_cte_m=max(r['cte_m']for r in controls),maximum_speed_mps=max(r['speed_mps']for r in controls),end_state=message(end),completed=True);results.append(result);save(a.output/'routes_partial.json',results);print(route['route_id'],round(station,1),'m',flush=True)
    except BaseException as e:failure=dict(type=type(e).__name__,message=str(e));raise
    finally:
        first=session.restore(initial);save(a.output/'restore_before_relay_stop.json',first);(a.udp_relay/'STOP').write_text(utc()+'\n')
        deadline=time.monotonic()+5
        while not(a.udp_relay/'relay_stopped.json').exists()and time.monotonic()<deadline:time.sleep(.05)
        session.channel.close()
        if not(a.udp_relay/'relay_stopped.json').exists():raise RuntimeError('relay did not stop; inspect before any new drive')
        # Last relay packets can change control mode. Restore via fresh SDK after stop.
        final=Session(SimpleNamespace(endpoint=a.endpoint,udp_relay=None,native_velocity_divisor=3.6));restoration=final.restore(initial);final.channel.close()
        record=dict(finished_utc=utc(),routes=results,failure=failure,restoration=restoration,relay_stopped_sha256=sha256_file(a.udp_relay/'relay_stopped.json'))
        save(a.output/'completion.json',record)
        if not restoration.get('verified'):raise RuntimeError('final Keyboard/Park restoration failed')
        if failure is None:(a.output/'_SUCCESS').write_text(sha256_file(a.output/'completion.json')+'\n')
if __name__=='__main__':main()
