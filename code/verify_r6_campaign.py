"""Independently audit R6 pairing, qualification, claims and legacy identity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from swmpc_sim import Scenario
from operational_calibration import ORIGINAL
from orbit_env import orbit_track, cylindrical_sunlight
from paired_resource_campaign import scenario
rows=json.loads(Path('../data/paired_resource_campaign.json').read_text())
assert len(rows)==144
keys=[(r['event'],r['h_km'],r['payload'],r['phase_deg'],r['ctrl'],r['seed']) for r in rows]
assert len(set(keys))==144
expected={(e,h,l,ph,c,s) for e in ORIGINAL for h in (400.,550.) for l in (False,True)
          for ph in (0.,180.) for c in ('pid_lowdrag_preview','swmpc') for s in range(3)}
assert set(keys)==expected
for r in rows:
    m=r['full']
    assert m['E_start_Wh']==1350.
    assert m['powered_regime_valid'] and m['n_floor_viol']==m['n_energy_floor_viol']==0
    assert m['unserved_bus_Wh']==0.
    sc=scenario(r['event'],r['h_km']*1000,r['phase_deg'])
    t=sc.tr['t']/3600-sc.spinup_h
    bus=600.+200.*((t>=30)&(t<36)) if r['payload'] else np.full(len(t),600.)
    q=(1500.*cylindrical_sunlight(sc.tr)-bus)[:int(sc.dur_h*60)]/60.
    cumulative=np.r_[0.,np.cumsum(q)]
    # Independent closed-form cap envelope, avoiding the campaign recurrence.
    upper=min(1350.+cumulative[-1],1500.+cumulative[-1]-cumulative.max())
    assert np.isclose(upper,r['terminal_upper_bound_Wh'],rtol=0,atol=1e-7)
    assert m['E_end_Wh']<=upper+1e-6
    if r['ctrl']=='swmpc' and upper<1450:
        assert m['qp_failures_all']==36
pairs=pd.read_csv('../data/paired_resource_pairs.csv')
bykey={k:r for k,r in zip(keys,rows)}
count=0;fail=0
for row in pairs.to_dict('records'):
    k=(row['event'],row['h_km'],row['payload'],row['phase_deg'])
    pi=bykey[(*k,'pid_lowdrag_preview',row['seed'])]['full']
    mp=bykey[(*k,'swmpc',row['seed'])]['full']
    saving=100*(pi['fuel_g']-mp['fuel_g'])/pi['fuel_g']
    assert np.isclose(saving,row['saving_full_pct'],rtol=0,atol=1e-10)
    match=abs(pi['E_end_Wh']-mp['E_end_Wh'])<=5 and min(pi['E_end_Wh'],mp['E_end_Wh'])>=1445
    assert match==row['resource_matched']
    count+=match;fail+=mp['qp_failures_all']
assert count==14 and fail==1308
assert sum(r['ctrl']=='swmpc' and r['terminal_upper_bound_Wh']<1450 for r in rows)==36
assert (pairs.mpc_E_end_Wh>=1445).sum()==36
assert len(pairs[pairs.resource_matched & (pairs.saving_full_pct>0)])==2
# Default orbital phase leaves the exact legacy track unchanged.
sc=Scenario('phase_check',ORIGINAL['Feb2022'],400e3)
track=orbit_track(sc.t0,sc.dur_h*3600+6*3600,60.,h_mean=400e3)
for k in ('lat','lon','alt','r_eci','sun_unit'):
    assert np.array_equal(sc.tr[k],track[k])
print("PASS: numerical campaign design, resources, failures and comparisons.")
