"""R4 paired baseline/power/Cd audit; fixed seeds 0,1,2; no retuning.
Outputs are separate from the original ten-seed nominal/timing experiment.
"""
import json, os
from multiprocessing import Pool
from functools import lru_cache
from pathlib import Path
from swmpc_sim import Scenario, run, metrics
from run_all import EVENTS, SIG, PID

@lru_cache(None)
def scenario(ev, h):
    return Scenario(ev, EVENTS[ev], h)

def one(args):
    kind, ev, h, ctrl, seed, par = args
    tag = f'{h/1e3:.0f}km'
    o = run(scenario(ev, h), ctrl, seed=seed, par=par,
            sigma_prof=SIG[tag]['sigma_lam_p95'], pid_wn=PID[tag]['wn'])
    return dict(kind=kind,event=ev,h_km=h/1e3,ctrl=ctrl,seed=seed,par=par,metrics=metrics(o))

if __name__ == '__main__':
    jobs=[]
    for ev in EVENTS:
        for h in (400e3,550e3):
            for c in ('pid_ff','pid_lowdrag','swmpc'):
                for seed in range(3): jobs.append(('paired',ev,h,c,seed,{}))
    for gen in (750.0,600.0):
        for c in ('pid_ff','pid_lowdrag','swmpc'):
            for seed in range(3): jobs.append(('power','May2024',400e3,c,seed,{'P_gen':gen}))
    for cd in (1.8,2.6):
        for c in ('pid_ff','pid_lowdrag','swmpc'):
            for seed in range(3): jobs.append(('Cd','May2024',400e3,c,seed,{'Cd':cd}))
    with Pool(int(os.environ.get('NPROC',2))) as pool:
        out=pool.map(one,jobs,chunksize=3)
    Path('../data/review_extensions.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    old=json.load(open('../data/results_metrics.json'))['results']
    for row in out:
        if row['kind']=='paired' and row['ctrl']!='pid_lowdrag':
            ref=old[f"{row['event']}_{row['h_km']:.0f}km"][row['ctrl']][row['seed']]
            for k in ('rms','fuel_g','minE','min_x','n_floor_viol'):
                assert abs(row['metrics'][k]-ref[k])<1e-7,(row,k,ref[k])
    print(f'Finished {len(out)} runs; nominal metrics agree with archived seeds.')
