"""Independent checks of R7 design, accounting, endpoints and reported claims."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from operational_calibration import ORIGINAL, PID
from orbit_env import cylindrical_sunlight, orbit_track
from swmpc_sim import G0, PAR, Scenario, run

rows=json.loads(Path('../data/common_recovery_campaign.json').read_text())
digest=hashlib.sha256(Path('../R7_COMPARISON_PROTOCOL.md').read_bytes()).hexdigest()
expected={(e,h,l,ph,c,s) for e in ORIGINAL for h in (400.,550.) for l in (False,True)
          for ph in (0.,180.) for c in ('pid_lowdrag_preview','swmpc','swmpc_nosw') for s in range(5)}
keys=[(r['event'],r['h_km'],r['payload'],r['phase_deg'],r['ctrl'],r['seed']) for r in rows]
assert len(rows)==len(set(keys))==360 and set(keys)==expected
endpoints={}
for e,h,l,ph,c,s in expected:
    if (e,h,ph) in endpoints: continue
    t0=np.datetime64(ORIGINAL[e])-np.timedelta64(12*3600,'s')
    tr=orbit_track(t0,112*3600,60.,h_mean=h*1000,u0_deg=ph)
    lit=cylindrical_sunlight(tr)
    good=lit[:len(lit)//5*5].reshape(-1,5).all(axis=1)
    # Compute endpoint from adjacent step flags, independently of runner loop.
    candidates=np.flatnonzero(good[:-1]&~good[1:])+1
    step=candidates[candidates>=108*12][0]
    endpoints[(e,h,ph)]=step/12.-12.
for r in rows:
    assert r['protocol_sha256']==digest
    assert r['completion_event_h']==endpoints[(r['event'],r['h_km'],r['phase_deg'])]
    f=r['full'];pre=r['before_recovery'];rec=r['recovery']
    assert f['E_start_Wh']==1350. and f['E_end_Wh']==1500.
    expected_load=600*(r['completion_event_h']+12)+(1200 if r['payload'] else 0)
    assert abs(f['total_load_Wh']-expected_load)<1e-8
    assert f['powered_regime_valid'] and f['n_floor_viol']==f['n_energy_floor_viol']==0
    assert f['unserved_bus_Wh']==0 and f['qp_failures_all']==0
    assert np.isclose(f['fuel_g'],pre['fuel_g']+rec['fuel_g'],atol=1e-9,rtol=0)
    assert rec['qp_attempts_all']==f['qp_attempts_all'] # metrics labels refer to all supplied trajectory steps
    # The recovery metric includes all trajectory QP counts; actual recovery
    # attempts are independently checked in the saved designated traces below.
    for window in ('full','event_window','before_recovery','recovery'):
        m=r[window]
        balance=m['total_generated_Wh']-m['total_load_Wh']-m['total_propulsion_Wh']-m['total_shunt']+m['total_unserved_Wh']
        assert abs(m['E_end_Wh']-m['E_start_Wh']-balance)<1e-8
        prop=m['fuel_g']*1e-3*(PAR['Isp']*G0)**2/(2*PAR['eta_T']*3600)
        assert abs(prop-m['total_propulsion_Wh'])<1e-8
    # No additional terminal-energy requirement: 1008 QPs before recovery.
    assert f['qp_attempts_all']==(0 if r['ctrl']=='pid_lowdrag_preview' else 1008)
bykey=dict(zip(keys,rows))
pairs=pd.read_csv('../data/common_recovery_pairs.csv')
assert len(pairs)==120
for p in pairs.to_dict('records'):
    k=(p['event'],p['h_km'],p['payload'],p['phase_deg'])
    a=bykey[(*k,'pid_lowdrag_preview',p['seed'])]['full']
    b=bykey[(*k,'swmpc',p['seed'])]['full']
    assert p['resource_matched'] and abs(a['x_end']-b['x_end'])<=1
    expected_saving=100*(a['fuel_g']-b['fuel_g'])/a['fuel_g']
    assert abs(p['saving_full_pct']-expected_saving)<1e-10
    # Paired form of the exact energy identity: identical endpoints and loads.
    assert abs(a['total_load_Wh']-b['total_load_Wh'])<1e-8
    left=(a['fuel_g']-b['fuel_g'])*1e-3*(PAR['Isp']*G0)**2/(7200*PAR['eta_T'])
    right=a['total_generated_Wh']-b['total_generated_Wh']-a['total_shunt']+b['total_shunt']
    assert abs(left-right)<1e-8
assert pairs.resource_matched.sum()==120 and (pairs.saving_full_pct>0).sum()==40
for f in Path('../data/common_recovery_traces').glob('*.npz'):
    t=np.load(f);after=t['t_h']>=72
    assert t['qp_attempt'][after].sum()==0
    assert t['A'][after][-1]==PAR['A_nom']
    assert t['E'][-1]==1500
print('PASS: 360 unique fixed-design runs; geometry-only completion; 120/120 PI/MPC pairs qualify; equal endpoints and zero sampled floor violations; all energy and fuel balances <1e-8 Wh; recovery is non-MPC.')

# Repeated campaign: endpoint measurement was corrected after plant propagation.
repeat=Path('../provenance/numerical_repeat/common_recovery_per_seed.csv')
if repeat.exists():
    old=pd.read_csv(repeat).set_index(['event','h_km','payload','phase_deg','ctrl','seed']).sort_index()
    new=pd.read_csv('../data/common_recovery_per_seed.csv').set_index(old.index.names).sort_index()
    for col in ('full_fuel_g','full_E_end_Wh','event_window_rms','full_minE'):
        assert np.array_equal(old[col].to_numpy(),new[col].to_numpy()),col
    print('PASS: all 360 repeated runs preserve fuel, event RMS, final battery and minimum energy exactly; the endpoint-altitude metric now uses the post-propagation state.')

# An optional exact numerical regression against the frozen R6 simulator.
frozen=Path('../provenance/swmpc_sim_R6.py')
if frozen.exists():
    spec=importlib.util.spec_from_file_location('frozen_r6',frozen)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    sc=Scenario('legacy_check',ORIGINAL['May2024'],400e3)
    sigma=json.loads(Path('../data/sigma_profile.json').read_text())['400km']['sigma_lam_p95']
    kw=dict(seed=0,sigma_prof=sigma,pid_wn=PID['400km']['wn'])
    a=old.run(sc,'swmpc',**kw);b=run(sc,'swmpc',**kw)
    for key in a:
        if isinstance(a[key],np.ndarray) and key not in ('solve_ms','solve_ms_all','loop_ms_all'):
            assert np.array_equal(a[key],b[key],equal_nan=True),key
    print('PASS: disabling the recovery option preserves every non-timing R6 output array exactly for the May2024/400km/seed0 regression case in the same environment.')
