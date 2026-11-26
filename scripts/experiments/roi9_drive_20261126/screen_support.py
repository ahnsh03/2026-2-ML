from pathlib import Path
from types import SimpleNamespace
import argparse,json,sys,time,math,hashlib
import grpc,cv2,numpy as np
from PIL import Image
from collect import Session,message,save,SensorStub,SensorDataSaveConfig,Empty
from quality_estimation.spatial_features import SpatialProfile
parser=argparse.ArgumentParser(description='Select geometry by native GT support only; no perception inference or prediction errors.')
for name in ['output','plan','udp-relay']:parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--candidate',type=int,required=True);parser.add_argument('--endpoint',default='host.docker.internal:7789');parser.add_argument('--simulator-file-dir',required=True);a=parser.parse_args()
S=a.output;root=S/'screen';root.mkdir(parents=True,exist_ok=True)
routes=json.loads(a.plan.read_text())['routes'];route=routes[a.candidate];dest=root/f'candidate_{a.candidate:02d}';dest.mkdir(exist_ok=False)
s=Session(SimpleNamespace(endpoint=a.endpoint,udp_relay=a.udp_relay))
w=s.environment.GetWeather(Empty(),timeout=5)
initial=dict(ego=message(s.state()),mode=message(s.actor.GetVehicleControlMode(s.ego,timeout=5)),weather=message(w),weather_type=w.type,time=message(s.environment.GetTime(Empty(),timeout=5)))
save(dest/'before.json',initial);records=[]
try:
 s.set_condition(0,13);points=np.asarray(route['points']);stations=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1))];regions=SpatialProfile.load().native[::2,::2]
 for j,station in enumerate([8.,16.,24.]):
  xyz=np.array([np.interp(station,stations,points[:,axis]) for axis in range(3)]);ahead=np.array([np.interp(station+1,stations,points[:,axis]) for axis in range(3)])
  pose=dict(location=dict(zip('xyz',[float(xyz[0]),float(xyz[1]),float(xyz[2]+.35)])),rotation=dict(x=0,y=0,z=math.degrees(math.atan2(ahead[1]-xyz[1],ahead[0]-xyz[0]))));s.place(pose);time.sleep(.4)
  name=f'markedscreen_c{a.candidate:02d}_p{j}';request=SensorDataSaveConfig(is_custom_file_name=True,custom_file_name=name,file_dir=a.simulator_file_dir);rpc='acknowledged'
  try:SensorStub(s.channel).SaveSensorData(request,timeout=1)
  except grpc.RpcError as error:
   if error.code()!=grpc.StatusCode.DEADLINE_EXCEEDED:raise
   rpc=str(error.code())
  files={};deadline=time.monotonic()+5
  for suffix in ['Intensity','Semantic','Depth','Instance']:
   path=root/'captures/CAMERA_1'/(name+'_'+suffix+'.png')
   while not path.exists():
    if time.monotonic()>deadline:raise RuntimeError('static support-screen files missing')
    time.sleep(.05)
   with Image.open(path) as im:assert im.size==(1280,720);im.verify()
   files[suffix]=dict(path=str(path.relative_to(S)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  sem=cv2.imread(str(S/files['Semantic']['path']));semantic=cv2.resize(sem,(640,360),interpolation=cv2.INTER_NEAREST);lane=np.all(semantic==255,2)|np.all(semantic==[0,255,255],2)
  counts=[int((lane&(regions==i)).sum()) for i in range(9)];records.append(dict(point=j,station_m=station,pose=pose,files=files,rpc=rpc,whole_gt_pixels=sum(counts),near_gt_pixels=sum(counts[6:8]),far_gt_pixels=sum(counts[4:6])))
 qualified=all(r['whole_gt_pixels']>=500 and r['near_gt_pixels']>=100 and r['far_gt_pixels']>=100 for r in records)
 save(dest/'support.json',dict(candidate=a.candidate,route=route,qualified=qualified,records=records,scope='GT support only at three fixed Sunny13 poses. No perception inference or quality error inspected; static geometry screening, not moving evidence.'))
 print('candidate',a.candidate,'qualified',qualified,[(r['near_gt_pixels'],r['far_gt_pixels']) for r in records],flush=True)
finally:
 restored=s.restore(initial);save(dest/'restoration.json',restored)
 if not restored.get('verified'):raise RuntimeError('support-screen restoration failed')
