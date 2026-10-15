import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import cv2,numpy as np,unittest,tempfile,json,datetime as dt
from trainwatch import core,export
class FixTests(unittest.TestCase):
 def test_rotated_mask_and_sampling(self):
  p=core.Profile(track_rect=[140,130,500,210],track_angle=-28,ref_line=[[250,60],[390,300]])
  det=core.Detector(p,core.work_crop(p,640,360))
  self.assertGreater(det.ref_mask.sum(),30)
  self.assertIsNotNone(det.track_maps)
  self.assertGreater(det.ref_idx,20)
  self.assertLess(det.ref_idx,340)
  self.assertTrue(np.all(det.track_mask[det.ref_mask]))
  self.assertEqual(core.Profile.from_dict(json.loads(p.to_json())).track_angle,-28)
 def test_passing_process_progress_and_cancel(self):
  from trainwatch import selftest
  with tempfile.TemporaryDirectory() as tmp:
   f=str(pathlib.Path(tmp)/'passing.avi');selftest.make_video(f,seconds=12,trains=((5,1),))
   p=core.Profile(passing_only=True);updates=[]
   core.process([f],[0],p,progress=lambda a,b:updates.append(a))
   self.assertTrue(updates)
   calls=[]
   core.process([f],[0],p,cancel=lambda:(calls.append(True) or True))
   self.assertEqual(len(calls),1)
 def test_passing_background_drift_and_thirteen_second_train(self):
  p=core.Profile(passing_only=True,occluded_track=True)
  crop=core.work_crop(p,640,360);det=core.Detector(p,crop)
  x0,y0,x1,y1=crop;bg=np.full((360,640),100,np.uint8)
  for y in range(140,240,8): cv2.line(bg,(150,y),(550,y),130,1)
  det.bg=cv2.GaussianBlur(bg[y0:y1,x0:x1],(5,5),0).astype('float32')
  for i in range(401):
   t=i*.2;im=np.clip(bg.astype('float32')+t*.5,0,255).astype('uint8')
   for entry,length in [(5,160),(45,1040)]:
    head=int(350+(t-entry)*80);tail=head-length
    if head>=0 and tail<640:
     cv2.rectangle(im,(tail,168),(head,214),230,-1)
     for wx in range(tail+10,head,25): cv2.rectangle(im,(wx,176),(wx+12,197),40,-1)
   im=cv2.GaussianBlur(im,(5,5),0)
   det.feed(t,im[y0:y1,x0:x1],0,'synthetic',t)
  ev=[e for e in det.finish() if e['valid']]
  self.assertEqual(len(ev),2,[(e['start'],e['end']) for e in ev])
  for e,a,b in zip(ev,[5,45],[7,58]):
   self.assertLess(abs(e['start']-a),.5)
   self.assertLess(abs(e['end']-b),.5)
 def test_completed_pass_static_tail_is_not_train_occupancy(self):
  # Regression: train motion ends at 7 s but stationary illumination residual lasts to 13 s.
  for passing,expected in [(True,7),(False,13)]:
   p=core.Profile(passing_only=passing);det=core.Detector(p,core.work_crop(p,640,360))
   c=dict(start=5.,end=13.,scores=[(i/5,25,'v',i/5) for i in range(25,66)],
          tail=[(i/5,25,'v',i/5) for i in range(25,76)],
          ref_motion_tail=[(i/5,4 if i<=35 else (1.2 if i in (45,60) else .1),'v',i/5) for i in range(25,76)])
   det._refine(c);self.assertAlmostEqual(c['end'],expected)
 def test_rotated_train_entry_and_exit(self):
  p=core.Profile(track_rect=[100,140,540,220],track_angle=-27)
  center=np.float32([320,180]);a=np.deg2rad(-27)
  R=np.float32([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
  p.ref_line=((np.float32([[320,140],[320,220]])-center)@R.T+center).tolist()
  crop=core.work_crop(p,640,360);det=core.Detector(p,crop)
  x0,y0,x1,y1=crop;det.bg=np.full((y1-y0,x1-x0),100,np.float32)
  M=cv2.getRotationMatrix2D((320,180),27,1)
  for i in range(121):
   t=i/10;im=np.full((360,640),100,np.uint8)
   head=int(320+(t-5)*80);tail=head-160
   cv2.rectangle(im,(tail,158),(head,214),230,-1)
   im=cv2.warpAffine(im,M,(640,360),borderValue=100)
   det.feed(t,im[y0:y1,x0:x1],0,'synthetic',t)
  ev=[e for e in det.finish() if e['valid']]
  self.assertEqual(len(ev),1,ev)
  self.assertLess(abs(ev[0]['start']-5),.3)
  self.assertLess(abs(ev[0]['end']-7),.3)
 def test_rotated_direction_labels(self):
  p=core.Profile(track_angle=90)
  self.assertEqual(p.direction_label(1),'往下')
  self.assertEqual(p.direction_label(-1),'往上')
 def test_reference_outside_track_rejected(self):
  p=core.Profile(ref_line=[[0,0],[20,0]])
  with self.assertRaises(ValueError): core.Detector(p,core.work_crop(p,640,360))
 def test_real_camera_guard_small_and_large_shift(self):
  cv2.setNumThreads(1)
  rng=np.random.default_rng(818)
  bg=cv2.GaussianBlur(rng.integers(0,256,(360,640,3),dtype=np.uint8),(5,5),0)
  for dx,expected in [(9,False),(32,True)]:
   guard=core.CameraGuard(core.Profile())
   moved=cv2.warpAffine(bg,np.float32([[1,0,dx],[0,1,0]]),(640,360))
   p=core.Profile(camera_anchors=[core.make_anchor(bg)])
   self.assertEqual(core.plan_anchor_save(p,moved,False)['verdict'],'moved' if expected else 'same')
   hits=[]
   for i in range(81):
    t=i*.5;h=guard.check(t,bg if t<10 else moved,50,False,'test',t)
    if h: hits.append(h)
   self.assertEqual(bool(hits),expected,(dx,hits))
 def test_minor_and_major_geometry(self):
  def r(n): return dict(n=300,off=np.ones((7,2),np.float32)*[n,0])
  self.assertEqual(core.geo_verdict(r(9),tolerance=core.Profile().camera_tolerance),'same')
  self.assertEqual(core.geo_verdict(r(30),tolerance=core.Profile().camera_tolerance),'moved')
  self.assertEqual(core.geo_verdict(None,tolerance=20),'uncertain')
 def test_unmeasured_camera_does_not_interrupt_train(self):
  p=core.Profile();det=core.Detector(p,core.work_crop(p,640,360));det._frame=None;det._sat=50
  det.guard.check=lambda *a:(5,'f',5,'uncertain','uncertain')
  det.cur={'marker':1};self.assertFalse(det._camera_check(10,None,1,'f',10))
  self.assertEqual(det.camera_events,[]);self.assertEqual(det.cur,{'marker':1})
 def test_txt_merge_encodings_sort_dedup_bytes(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=pathlib.Path(tmp);a='2030/01/02,08:00:00,08:00:03,10,FR,,A';b='2030/01/02,08:10:00,08:10:03,10,FR,,A'
   (d/'a').write_bytes((b+'\n\n'+a+'\n').encode('utf-16'))
   (d/'b').write_bytes(('\ufeff'+a+'\r\n').encode('utf-8'))
   ls,n=export.merge_noise_lines([str(d/'a'),str(d/'b')]);self.assertEqual(ls,[a,b]);self.assertEqual(n,1)
   export.write_noise_lines(str(d/'out'),ls);self.assertEqual((d/'out').read_bytes(),(a+'\r\n'+b+'\r\n').encode('ascii'))
   (d/'bad').write_text(a+b,encoding='ascii')
   with self.assertRaises(ValueError): export.merge_noise_lines([str(d/'bad')])
 def test_txt_bad_date_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=pathlib.Path(tmp)/'a';p.write_text('2030/02/30,08:00:00,08:00:03,10,FR,,A',encoding='ascii')
   with self.assertRaises(ValueError): export.merge_noise_lines([str(p)])
if __name__=='__main__': unittest.main(verbosity=2)
