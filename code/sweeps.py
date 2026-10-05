"""Sensitivity studies: (1) power-preference weight R_A (fuel vs generation loss),
(2) tightening level (sigma scaled by 0, 0.5, 1), (3) plant-model mismatch with
NRLMSIS 2.1 as truth, (4) forecast latency."""
import numpy as np, json
from multiprocessing import Pool
from swmpc_sim import Scenario, run, metrics
from run_all import EVENTS, SIG, PID, storm_window, CTRLS

SEEDS = range(5)
def one(args):
    kind, ev, h, ctrl, val, truth = args
    scn = Scenario(ev, EVENTS[ev], h, truth_version=truth)
    tag = f'{h/1e3:.0f}km'; sig = np.array(SIG[tag]['sigma_lam_p95'])
    par = {}; lat = 3600.0
    if kind == 'RA': par = {'R_A': val}
    if kind == 'sigma': sig = sig*val
    if kind == 'latency': lat = val
    ms = [metrics(run(scn, ctrl, seed=s, sigma_prof=sig, pid_wn=PID[tag]['wn'], par=par, latency_s=lat,
                      use_sw=(ctrl != 'swmpc_nosw')), storm=storm_window(EVENTS[ev])) for s in SEEDS]
    return args, {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}

if __name__ == '__main__':
    jobs = []
    for ev in ('Mar2015', 'May2024'):
        for h in (550e3, 400e3):
            for v in (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0):
                jobs.append(('RA', ev, h, 'swmpc', v, 0))
    for ev in ('Mar2015', 'May2024'):
        for v in (0.0, 0.5, 1.0):
            for c in ('mpc_thrust', 'swmpc'):
                jobs.append(('sigma', ev, 400e3, c, v, 0))
    for ev in ('Mar2015', 'May2024'):
        for h in (550e3, 400e3):
            for c in CTRLS:
                jobs.append(('msis21', ev, h, c, 0, 2.1))
    for v in (0.0, 3600.0, 3*3600.0, 6*3600.0):
        jobs.append(('latency', 'May2024', 400e3, 'swmpc', v, 0))
    with Pool(2) as pool:
        out = pool.map(one, jobs)
    res = [dict(kind=a[0], event=a[1], h_km=a[2]/1e3, ctrl=a[3], value=a[4], truth=a[5], metrics=m) for a, m in out]
    json.dump(res, open('../data/sensitivity.json', 'w'), indent=1)
    for r in res:
        m = r['metrics']
        print(r['kind'], r['event'], r['h_km'], r['ctrl'], r['value'], 'rms %.2f max %.2f minx %.2f fuel %.2f viol %.1f gloss %.3f minE %.0f' % (
            m['rms'], m['maxabs'], m['min_x'], m['fuel_g'], m['n_floor_viol'], m['mean_gen_loss'], m['minE']))
