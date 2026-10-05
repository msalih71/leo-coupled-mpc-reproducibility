"""R6 fixed-protocol paired comparison; see ../R6_COMPARISON_PROTOCOL.md."""
import json
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from swmpc_sim import Scenario, run, metrics
from orbit_env import cylindrical_sunlight
from operational_calibration import ORIGINAL, PID

@lru_cache(None)
def scenario(event, altitude, phase):
    return Scenario(event, ORIGINAL[event], altitude, orbit_phase_deg=phase)

def terminal_energy_upper_bound(sc, generation, load):
    # Optimistic physical envelope: full-area generation, zero propulsion draw.
    # Same 60-s sampling and storage cap as the plant; monotone in initial energy.
    energy=1350.
    for gen,bus in zip(generation[:int(sc.dur_h*60)],load[:int(sc.dur_h*60)]):
        energy=min(1500.,energy+(gen-bus)/60.)
    return float(energy)

def job(args):
    event, altitude, payload, phase, controller, seed = args
    sc = scenario(event, altitude, phase)
    time_h = sc.tr['t']/3600-sc.spinup_h
    sunlight = cylindrical_sunlight(sc.tr)
    load = 600.+200.*((time_h>=30)&(time_h<36)) if payload else np.full(len(time_h),600.)
    profile = json.loads(Path('../data/operational_sigma_profile.json').read_text())[f'{altitude/1000:.0f}km']['sigma_lam_p95']
    kw = {'terminal_energy':(72.,1450.)} if controller=='swmpc' else {}
    out = run(sc,controller,seed=seed,sigma_prof=profile,pid_wn=PID[f'{altitude/1000:.0f}km']['wn'],
              strict_solar=True,generation_W=1500.*sunlight,load_W=load,**kw)
    return dict(event=event,h_km=altitude/1000,payload=payload,phase_deg=phase,ctrl=controller,seed=seed,
                sunlight_fraction=float(sunlight[(time_h>=0)&(time_h<72)].mean()),
                terminal_upper_bound_Wh=terminal_energy_upper_bound(sc,1500.*sunlight,load),
                full=metrics(out,t_from=-12.),event_window=metrics(out,t_from=0.))

def summarize(rows):
    records=[]
    for row in rows:
        record={k:v for k,v in row.items() if k not in ('full','event_window')}
        for window in ('full','event_window'):
            record.update({window+'_'+k:v for k,v in row[window].items()})
        records.append(record)
    df=pd.DataFrame(records)
    df.to_csv('../data/paired_resource_per_seed.csv',index=False)
    keys=['event','h_km','payload','phase_deg','seed']
    pi=df[df.ctrl=='pid_lowdrag_preview'].set_index(keys)
    mpc=df[df.ctrl=='swmpc'].set_index(keys)
    assert pi.index.equals(mpc.index)
    pairs=pi.index.to_frame(index=False)
    for ctrl,sub in [('pi',pi),('mpc',mpc)]:
        for metric in ['fuel_g','rms','E_start_Wh','E_end_Wh','minE','n_floor_viol','n_energy_floor_viol','powered_regime_valid','qp_failures_all','unserved_bus_Wh']:
            pairs[ctrl+'_'+metric]=sub['full_'+metric].to_numpy()
        pairs[ctrl+'_event_start_Wh']=sub['event_window_E_start_Wh'].to_numpy()
    pairs['endpoint_difference_Wh']=abs(pairs.pi_E_end_Wh-pairs.mpc_E_end_Wh)
    pairs['saving_full_pct']=100*(pairs.pi_fuel_g-pairs.mpc_fuel_g)/pairs.pi_fuel_g
    pairs['rms_ratio']=pairs.mpc_rms/pairs.pi_rms
    pairs['resource_matched']=(pairs.endpoint_difference_Wh<=5)&(pairs.pi_E_end_Wh>=1445)&(pairs.mpc_E_end_Wh>=1445)
    for ctrl in ('pi','mpc'):
        pairs['resource_matched'] &= pairs[ctrl+'_powered_regime_valid'] & (pairs[ctrl+'_n_floor_viol']==0) & (pairs[ctrl+'_n_energy_floor_viol']==0)
    pairs.to_csv('../data/paired_resource_pairs.csv',index=False)
    summary=pairs.groupby(['event','h_km','payload','phase_deg']).agg(
        n_pairs=('seed','size'),n_matched=('resource_matched','sum'),saving_mean_pct=('saving_full_pct','mean'),
        saving_min_pct=('saving_full_pct','min'),saving_max_pct=('saving_full_pct','max'),
        pi_rms_m=('pi_rms','mean'),mpc_rms_m=('mpc_rms','mean'),
        endpoint_difference_max_Wh=('endpoint_difference_Wh','max'),pi_minE_Wh=('pi_minE','min'),mpc_minE_Wh=('mpc_minE','min'))
    summary.to_csv('../data/paired_resource_summary.csv')
    print(summary.to_string(),flush=True)
    print('QUALIFYING PAIRS',int(pairs.resource_matched.sum()),'/',len(pairs),flush=True)

if __name__=='__main__':
    jobs=[(ev,h,load,phase,c,s) for ev in ORIGINAL for h in (400e3,550e3) for load in (False,True)
          for phase in (0.,180.) for c in ('pid_lowdrag_preview','swmpc') for s in range(3)]
    rows=[]
    with Pool(2) as pool:
        for row in pool.imap(job,jobs,chunksize=1):
            rows.append(row)
            Path('../data/paired_resource_checkpoint.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
            if len(rows)%12==0:print('Completed',len(rows),'/',len(jobs),flush=True)
    Path('../data/paired_resource_campaign.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    summarize(rows)
