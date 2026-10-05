"""Checks causal input, legacy trajectory, eclipse geometry, and recorded split."""
import json,numpy as np
from swmpc_sim import Scenario,run,metrics,QPMPC,PAR
from tim import TIM,TIMPredictor
from orbit_env import orbit_track,cylindrical_sunlight
from run_all import SIG,PID,EVENTS
tr=orbit_track(np.datetime64('2024-05-09T22:00'),6*3600,60.,h_mean=400e3)
p=TIMPredictor(TIM.load(400),tr,400e3,strict_solar=True)
a=p.predict_steps(60,36,5);p.X[61:,1:4]+=100
assert np.array_equal(a,p.predict_steps(60,36,5))
tr={'r_eci':np.array([[-7e6,0,0],[7e6,0,0],[-7e6,7e6,0]]),'sun_unit':np.tile([1.,0,0],(3,1))}
assert np.array_equal(cylindrical_sunlight(tr),[0,1,1])
q=QPMPC(PAR);args=(0,1350,10,1,np.full(36,.03),np.full(36,.2))
a=q.solve(*args);b=q.solve(*args,power=np.full(36,900.),load=np.full(36,600.))
assert np.array_equal(a['z'],b['z']) and a['n_con']==222
sc=Scenario('May2024',EVENTS['May2024'],400e3)
o=run(sc,'swmpc',seed=0,sigma_prof=SIG['400km']['sigma_lam_p95'],pid_wn=PID['400km']['wn'])
ref=json.load(open('../data/results_metrics.json'))['results']['May2024_400km']['swmpc'][0]
for k in ('rms','fuel_g','minE','min_x','n_floor_viol'):assert abs(metrics(o)[k]-ref[k])<1e-7,(k,metrics(o)[k],ref[k])
r=json.load(open('../data/operational_coverage.json'))
assert len(r['rows'])==90 and len(r['calibration_dates'])==8 and len(r['test_dates'])==7
cal=set(r['calibration_dates']);test=set(r['test_dates'].values());assert cal.isdisjoint(test)
assert max(cal)<min(t for t in test if t[:4]!='2015')
trace=np.load('../data/operational_forecast_traces.npz')
profiles=json.load(open('../data/operational_sigma_profile.json'))
for h in (400,550):
 a=np.concatenate([trace[x['trace_key']+'_weighted'] for x in r['rows'] if x['phase']=='calibration' and x['h_km']==h])
 assert np.allclose(np.percentile(np.abs(a),95,axis=0),profiles[f'{h}km']['sigma_lam_p95'])
for row in r['rows']:
 a=trace[row['trace_key']+'_weighted'];m=trace[row['trace_key']+'_margin']
 assert a.shape==m.shape and a.shape[1]==36
 assert abs(np.mean(np.abs(a)<=m)-row['coverage_effective'])<1e-12
print('PASS causal solar persistence, shadow geometry, vector-power identity, legacy seed agreement, independent calibration/test split and coverage reconstruction.')
