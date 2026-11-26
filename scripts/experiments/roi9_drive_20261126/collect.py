"""Bounded low-speed continuous drives with asynchronous four-channel capture.

Native frame timestamps/SaveSensorData acknowledgements are NOT synthesized.
File-backed moving samples retain request/readback timing envelopes for a separate
RGB/Semantic alignment audit. Do not reuse stationary-capture acceptance claims.
"""
from __future__ import annotations
import argparse, hashlib, json, math, shutil, signal, time
from datetime import datetime, timezone
from pathlib import Path
import grpc
import numpy as np
from PIL import Image
from google.protobuf.json_format import MessageToDict
from morai.actor.actor_pb2_grpc import ActorStub
from morai.actor.actor_get_pb2 import GetAllActorsFilter
from morai.actor.actor_control_pb2 import VehicleCtrlCmd
from morai.actor.actor_set_pb2 import SetTransformParam, SetVelocityParam, VehicleGear, VehicleControlModeParam, EnableActor
from morai.actor.actor_enum_pb2 import VEHICLE_CONTROL_AUTO_MODE, VEHICLE_CONTROL_KEYBOARD, GEAR_MODE_D, GEAR_MODE_P
from morai.common.object_info_pb2 import ObjectInfo
from morai.common.type_pb2 import Empty
from morai.environment.environment_pb2_grpc import EnvironmentStub
from morai.sensor.sensor_pb2_grpc import SensorStub
from morai.sensor.sensor_data_save_config_pb2 import SensorDataSaveConfig

SOURCE_BYTES=Path(__file__).read_bytes()
SOURCE_SHA256=hashlib.sha256(SOURCE_BYTES).hexdigest()

def utc():return datetime.now(timezone.utc).isoformat()
def message(m):return MessageToDict(m,preserving_proto_field_name=True,including_default_value_fields=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def append(path,value):
 with path.open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
def norm(v):return math.sqrt(sum(getattr(v,k)**2 for k in 'xyz'))
def check(r):
 if r.status!=1:raise RuntimeError(str(r))
 return message(r)
def geometry(points,xy):
 p=np.asarray(points);d=np.diff(p[:,:2],axis=0);s=np.r_[0,np.cumsum(np.linalg.norm(d,axis=1))]
 t=np.clip(np.einsum('ij,ij->i',np.asarray(xy)-p[:-1,:2],d)/np.maximum(np.einsum('ij,ij->i',d,d),1e-12),0,1)
 q=p[:-1,:2]+t[:,None]*d;distance=np.linalg.norm(q-np.asarray(xy),axis=1);i=int(distance.argmin())
 return float(s[i]+t[i]*(s[i+1]-s[i])),float(distance[i]),s

class Session:
 def __init__(self,args):
  self.args=args;self.channel=grpc.insecure_channel(args.endpoint);grpc.channel_ready_future(self.channel).result(timeout=5)
  self.actor=ActorStub(self.channel);self.environment=EnvironmentStub(self.channel);self.ego=ObjectInfo(object_type=1);self.ego.id.value='Ego'
  self.state_type=type(self.actor.GetActorState(self.ego,timeout=3));self.last_udp_state=None
 def state(self):
  if not self.args.udp_relay:return self.actor.GetActorState(self.ego,timeout=1)
  p=self.args.udp_relay/'state.json';stop=time.monotonic()+.2
  while True:
   try:raw=json.loads(p.read_text(encoding='utf-8-sig'));break
   except (OSError,json.JSONDecodeError):
    if time.monotonic()>=stop:raise
    time.sleep(.01)
  if abs(time.time()-raw['received_unix_ms']/1000)>.5:raise RuntimeError('UDP state stale')
  self.last_udp_state=raw;s=self.state_type();s.actor_info.CopyFrom(self.ego)
  for axis in 'xyz':
   setattr(s.transform.location,axis,raw['position'][axis]);setattr(s.transform.rotation,axis,raw['rotation'][axis]);setattr(s.global_velocity,axis,raw['velocity_kph'][axis])
  s.vehicle_state.gear_mode=raw['gear'];s.vehicle_state.current_link_info.id.value=raw['link_id'];return s
 def mode(self,v):return check(self.actor.SetVehicleControlMode(VehicleControlModeParam(actor_info=self.ego,mode=v),timeout=5))
 def gear(self,v):return check(self.actor.SetVehicleGear(VehicleGear(actor_info=self.ego,gear=v),timeout=5))
 def relay_command(self,armed,throttle=0,brake=1,steer=0,gear=4):
  path=self.args.udp_relay/'command.json';temp=path.with_suffix('.tmp.driver')
  save(temp,dict(sequence=time.monotonic_ns(),armed=armed,throttle=throttle,brake=brake,steer=steer,gear=gear,utc=utc()))
  stop=time.monotonic()+.25
  while True:
   try:temp.replace(path);break
   except PermissionError:
    if time.monotonic()>=stop:raise
    time.sleep(.01)
 def control(self,throttle=0,brake=1,steer=0):
  if self.args.udp_relay:self.relay_command(True,throttle,brake,steer);return dict(queued=True,transport='Windows loopback UDP26R1 with independent0.6s watchdog')
  return check(self.actor.ControlVehicle(VehicleCtrlCmd(actor_info=self.ego,long_cmd_type=1,throttle=throttle,brake=brake,steer=steer),timeout=.6))
 def hold(self):
  if self.args.udp_relay:self.relay_command(False,gear=1);time.sleep(.25)
  self.gear(GEAR_MODE_P);self.mode(VEHICLE_CONTROL_KEYBOARD)
  check(self.actor.SetVelocity(SetVelocityParam(actor_info=self.ego,velocity=0),timeout=5))
 def place(self,pose):
  self.hold();req=SetTransformParam(actor_info=self.ego)
  for axis in 'xyz':setattr(req.transform.location,axis,pose['location'][axis]);setattr(req.transform.rotation,axis,pose['rotation'][axis])
  for _ in range(5):
   check(self.actor.SetTransform(req,timeout=5));self.hold();time.sleep(.25)
   s=self.state();yaw=(s.transform.rotation.z-pose['rotation']['z']+180)%360-180
   xy=math.hypot(s.transform.location.x-pose['location']['x'],s.transform.location.y-pose['location']['y'])
   if abs(yaw)<.2 and xy<.1:break
   req.transform.rotation.z-=yaw
  else:raise RuntimeError('placement did not match start XY/yaw')
  for _ in range(30):
   s=self.state()
   if norm(s.global_velocity)<.02:return s
   time.sleep(.15)
  raise RuntimeError('placement did not settle')
 def set_condition(self,w,h):
  oldw=self.environment.GetWeather(Empty(),timeout=5);oldh=self.environment.GetTime(Empty(),timeout=5)
  receipt=[]
  for setter,request,getter,field,target in [(self.environment.SetWeather,type(oldw)(type=w,instantly=True),self.environment.GetWeather,'type',w),(self.environment.SetTime,type(oldh)(hour=h,instantly=True,data_only=False),self.environment.GetTime,'hour',h)]:
   began=time.monotonic();code='acknowledged';observations=[]
   try:check(setter(request,timeout=10))
   except grpc.RpcError as e:
    if e.code()!=grpc.StatusCode.DEADLINE_EXCEEDED:raise
    code='rpc_deadline'
   stop=time.monotonic()+5
   while True:
    observed=getattr(getter(Empty(),timeout=5),field);observations.append(dict(elapsed=time.monotonic()-began,value=observed))
    if observed==target:break
    if time.monotonic()>stop:raise RuntimeError(f'condition readback differs: field={field}, requested={target}, observed={observed}')
    time.sleep(.2)
   receipt.append(dict(field=field,requested=target,rpc=code,observations=observations))
  time.sleep(1.5)
  return receipt
 def restore(self,initial):
  result={}
  try:self.control()
  except BaseException as exc:result['stop_command_error']=repr(exc)
  if self.args.udp_relay:
   self.relay_command(False,gear=1);time.sleep(.3)
  try:
   self.hold();self.set_condition(initial['weather_type'],initial['time']['hour'])
   s=self.place(initial['ego']['transform']);m=self.actor.GetVehicleControlMode(self.ego,timeout=5)
   result=dict(ego=message(s),mode=message(m),weather=message(self.environment.GetWeather(Empty(),timeout=5)),time=message(self.environment.GetTime(Empty(),timeout=5)))
   result['xy_error_m']=math.hypot(s.transform.location.x-initial['ego']['transform']['location']['x'],s.transform.location.y-initial['ego']['transform']['location']['y'])
   result['yaw_error_deg']=(s.transform.rotation.z-initial['ego']['transform']['rotation']['z']+180)%360-180
   result['native_velocity_norm']=norm(s.global_velocity)
   result['native_velocity_divisor']=float(getattr(self.args,'native_velocity_divisor',3.6))
   result['speed_mps']=result['native_velocity_norm']/result['native_velocity_divisor']
   result['verified']=result['xy_error_m']<.1 and abs(result['yaw_error_deg'])<.2 and result['speed_mps']<.02 and m.mode==VEHICLE_CONTROL_KEYBOARD and s.vehicle_state.gear_mode==GEAR_MODE_P
  except BaseException as exc:result['error']=repr(exc);result['verified']=False
  return result

def run_drive(session,route,w,h,args):
 runid=f"{args.prefix}_{route['route_id']}_w{w}_h{h:02d}";dest=args.output/'runs'/runid
 if (dest/'_SUCCESS').exists():
  summary=json.loads((dest/'summary.json').read_text());assert summary['plan_sha256']==args.plan_sha256
  assert json.loads((dest/'design.json').read_text())['plan_sha256']==args.plan_sha256
  return summary
 for prefix in args.reuse_prefix:
  reused=args.output/'runs'/f"{prefix}_{route['route_id']}_w{w}_h{h:02d}"
  if (reused/'_SUCCESS').exists():
   summary=json.loads((reused/'summary.json').read_text());assert summary['plan_sha256']==args.plan_sha256
   assert json.loads((reused/'design.json').read_text())['plan_sha256']==args.plan_sha256
   print('reuse completed',summary['run_id'],flush=True);return summary
 dest.mkdir(parents=True,exist_ok=False)
 (dest/'collector_source.py').write_bytes(SOURCE_BYTES)
 (dest/'plan_snapshot.json').write_bytes(args.plan_snapshot_bytes)
 save(dest/'design.json',dict(created_utc=utc(),source_sha256=digest(dest/'collector_source.py'),plan_sha256=args.plan_sha256,route=route,weather=w,hour=h,target_kph=args.target_kph,capture_interval_s=args.capture_interval,steering_range_deg=args.steering_range_deg,native_velocity_divisor=args.native_velocity_divisor,transport='WindowsUDP26R1+watchdog' if args.udp_relay else 'native_gRPC',scope='Actual moving capture. Camera exposure timestamps unavailable; request and per-file readbacks retained. UDP state timestamp is vehicle telemetry, not camera exposure. Direct ROS and UDP collision feedback are not tested.'))
 save(dest/'condition_receipt.json',session.set_condition(w,h));s0=session.place(route['start_pose']);save(dest/'before.json',message(s0))
 points=np.asarray(route['points']);_,_,stations=geometry(points,[s0.transform.location.x,s0.transform.location.y])
 session.mode(VEHICLE_CONTROL_AUTO_MODE);session.gear(GEAR_MODE_D)
 for _ in range(4):session.control();time.sleep(.05)
 started=time.monotonic();last_capture=-np.inf;captures=[];controls=[];end_reason=None
 while True:
  tick=time.monotonic()
  if datetime.now(timezone.utc)>=datetime.fromisoformat(args.hard_stop_utc):raise TimeoutError('collection hard stop')
  if tick-started>75:raise RuntimeError('drive timeout')
  if shutil.disk_usage(args.output).free<20*1024**3:raise RuntimeError('E free below20GiB')
  s=session.state();position=np.array([s.transform.location.x,s.transform.location.y]);station,cte,_=geometry(points,position);speed=norm(s.global_velocity)/args.native_velocity_divisor
  if cte>1 or speed>2.2:raise RuntimeError(f'tracking/speed bound: cte={cte}, speed={speed}')
  if [x for x in s.vehicle_state.collision_objects if x and x!='Ego']:raise RuntimeError('native collision')
  if stations[-1]-station<1.3:end_reason='route_complete';break
  if tick-started>10 and station<1:raise RuntimeError('no measured progress')
  lookahead=4.;goal=min(station+lookahead,float(stations[-1]));target=np.array([np.interp(goal,stations,points[:,i]) for i in [0,1]])
  yaw=math.radians(s.transform.rotation.z);bearing=math.atan2(*(target-position)[::-1]);alpha=(bearing-yaw+math.pi)%(2*math.pi)-math.pi
  delta=math.atan2(2*2.7*math.sin(alpha),max(np.linalg.norm(target-position),1.5))
  steer=float(np.clip(delta/math.radians(args.steering_range_deg),-.5,.5));target_speed=args.target_kph/3.6
  throttle=float(np.clip(.14+.16*(target_speed-speed),0,.3));brake=.25 if speed>target_speed+.35 else 0.
  session.control(throttle=throttle if not brake else 0,brake=brake,steer=steer)
  record=dict(utc=utc(),elapsed=tick-started,monotonic=tick,state=message(s),station_m=station,cte_m=cte,speed_mps=speed,command=dict(throttle=throttle if not brake else 0,brake=brake,steer=steer),udp_state=session.last_udp_state)
  controls.append(record);append(dest/'control.jsonl',record)
  if station>=1 and speed>.2 and tick-last_capture>=args.capture_interval and stations[-1]-station>3:
   name=runid+f'_f{len(captures):04d}';channel=grpc.insecure_channel(args.endpoint)
   request=SensorDataSaveConfig(is_custom_file_name=True,custom_file_name=name,file_dir=args.simulator_file_dir)
   capture=dict(sample_id=name,requested_utc=utc(),requested_monotonic=time.monotonic(),requested_state=record,first_seen={},channel=channel,call=SensorStub(channel).SaveSensorData.future(request,timeout=1))
   captures.append(capture);last_capture=tick
   append(dest/'capture_intents.jsonl',{k:v for k,v in capture.items() if k not in ['channel','call']})
  for capture in captures:
   for suffix in ['Intensity','Semantic','Depth','Instance']:
    file=args.output/'captures/CAMERA_1'/(capture['sample_id']+'_'+suffix+'.png')
    if suffix not in capture['first_seen'] and file.exists():capture['first_seen'][suffix]=dict(observed_utc=utc(),monotonic=time.monotonic(),bytes=file.stat().st_size,state=record)
   if time.monotonic()-capture['requested_monotonic']>1. and len(capture['first_seen'])<4:raise RuntimeError('moving capture files overdue; stop with independent UDP brake')
  time.sleep(max(0,.06-(time.monotonic()-tick)))
 session.control();session.hold();time.sleep(.4)
 rows=[];incomplete=0
 for capture in captures:
  call=capture.pop('call');channel=capture.pop('channel')
  try:capture['rpc_response']=message(call.result())
  except grpc.RpcError as e:capture['rpc_code']=str(e.code());capture['rpc_details']=e.details()
  finally:channel.close()
  capture['files']={};capture['accepted']=False;capture['exposure_timestamp']=None
  try:
   for suffix in ['Intensity','Semantic','Depth','Instance']:
    file=args.output/'captures/CAMERA_1'/(capture['sample_id']+'_'+suffix+'.png');size=file.stat().st_size
    with Image.open(file) as image:
     assert image.size==(1280,720);image.verify()
    assert size==file.stat().st_size and size>0
    capture['files'][suffix]=dict(path=str(file.relative_to(args.output)),sha256=digest(file),bytes=size)
   assert set(capture['first_seen'])=={'Intensity','Semantic','Depth','Instance'}
   latency=max(v['monotonic']-capture['requested_monotonic'] for v in capture['first_seen'].values())
   if latency>1.:raise ValueError('four files not observed within bounded moving envelope')
   actual=capture['first_seen']['Intensity']['state']['state']['transform']['location']
   capture.update(accepted=True,maximum_first_seen_latency_s=latency,quality_role=route['role'],alignment_acceptance='Pending separate cross-modal alignment/sensitivity audit; common request prefix and bounded file presence are not exposure synchronization proof.')
   rows.append(dict(sample_id=capture['sample_id'],image=args.output.name+'/'+capture['files']['Intensity']['path'],semantic=args.output.name+'/'+capture['files']['Semantic']['path'],expected_sha256=dict(rgb=capture['files']['Intensity']['sha256'],semantic=capture['files']['Semantic']['sha256']),view='front',group=route['route_id'],site=route['route_id'],cohort=route['role'],weather=w,hour=h,xy=[actual[k] for k in 'xy']))
  except BaseException as e:capture['rejection']=repr(e);incomplete+=1
  append(dest/'manifest.jsonl',capture)
 if not rows:raise RuntimeError('no complete moving samples')
 env=session.environment
 assert env.GetWeather(Empty(),timeout=5).type==w and env.GetTime(Empty(),timeout=5).hour==h
 summary=dict(run_id=runid,role=route['role'],route_id=route['route_id'],requested_link=route['link'],weather=w,hour=h,frames=len(rows),rejected=incomplete,control_rows=len(controls),duration_s=time.monotonic()-started,maximum_cte_m=max(r['cte_m'] for r in controls),maximum_speed_mps=max(r['speed_mps'] for r in controls),start_station_m=controls[0]['station_m'],last_station_m=controls[-1]['station_m'],end_reason=end_reason,plan_sha256=args.plan_sha256,source_sha256=SOURCE_SHA256)
 save(dest/'index.json',{runid:dict(simulator_release='26.R1',rows=rows)});save(dest/'summary.json',summary);(dest/'_SUCCESS').write_text(utc()+'\n');print(json.dumps(summary),flush=True);return summary

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--simulator-file-dir',required=True);p.add_argument('--endpoint',default='host.docker.internal:7789');p.add_argument('--prefix',required=True)
 p.add_argument('--routes',nargs='+');p.add_argument('--roles',nargs='+');p.add_argument('--condition',nargs=2,type=int,action='append');p.add_argument('--target-kph',type=float,default=4);p.add_argument('--steering-range-deg',type=float,default=20);p.add_argument('--native-velocity-divisor',type=float,choices=[1.,3.6],default=3.6);p.add_argument('--capture-interval',type=float,default=1.2);p.add_argument('--hard-stop-utc',default='2026-11-26T05:00:00+00:00')
 p.add_argument('--reuse-prefix',nargs='*',default=[])
 p.add_argument('--udp-relay',type=Path);p.add_argument('--max-new-runs',type=int,default=0)
 a=p.parse_args();assert 0<a.target_kph<=5 and .8<=a.capture_interval<=4 and 10<=a.steering_range_deg<=45
 a.plan_snapshot_bytes=a.plan.read_bytes();a.plan_sha256=hashlib.sha256(a.plan_snapshot_bytes).hexdigest()
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'runs').mkdir(exist_ok=True)
 session=Session(a);weather=session.environment.GetWeather(Empty(),timeout=5);initial=dict(utc=utc(),ego=message(session.state()),mode=message(session.actor.GetVehicleControlMode(session.ego,timeout=5)),weather=message(weather),weather_type=weather.type,time=message(session.environment.GetTime(Empty(),timeout=5)))
 actors=session.actor.GetAllActorsState(GetAllActorsFilter(vehicle=True,pedestrian=True,obstacle=True),timeout=5)
 assert initial['mode']['mode']=='VEHICLE_CONTROL_KEYBOARD' and initial['ego']['vehicle_state']['gear_mode']=='GEAR_MODE_P' and len(actors.states)==1
 def interrupted(*_):raise InterruptedError('collection interrupted')
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,interrupted)
 before=a.output/(a.prefix+'_before.json');assert not before.exists();save(before,initial)
 completed=[];failure=None;new_runs=0
 try:
  check(session.actor.SetPause(EnableActor(actor_info=session.ego,enable=False),timeout=5))
  plan=json.loads(a.plan_snapshot_bytes)
  for route in plan['routes']:
   if a.routes and route['route_id'] not in a.routes:continue
   if a.roles and route['role'] not in a.roles:continue
   for w,h in (a.condition or route['conditions']):
    result=run_drive(session,route,w,h,a);completed.append(result)
    if result['run_id'].startswith(a.prefix+'_'):new_runs+=1
    if a.max_new_runs and new_runs>=a.max_new_runs:return
 except BaseException as e:failure=repr(e);raise
 finally:
  restored=session.restore(initial);save(a.output/(a.prefix+'_completion.json'),dict(completed_runs=completed,failure=failure,restoration=restored,finished_utc=utc()))
  if not restored.get('verified'):raise RuntimeError('simulator restoration requires inspection')
if __name__=='__main__':main()
