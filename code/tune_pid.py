"""Tune the PI natural frequency on a CALIBRATION storm (7-10 Sep 2017, G4),
separately for each altitude; the same gains are then used for pid and pid_ff."""
import numpy as np, json
from swmpc_sim import Scenario, run, metrics
res = {}
for h in (550e3, 400e3):
    scn = Scenario('Sep2017', '2017-09-07T00:00', h)
    best = None
    for Th in (0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0):
        r = np.mean([metrics(run(scn, 'pid', seed=s, pid_wn=1/(Th*3600)))['rms'] for s in range(3)])
        r2 = np.mean([metrics(run(scn, 'pid_ff', seed=s, pid_wn=1/(Th*3600)))['rms'] for s in range(3)])
        print(h, Th, round(r, 3), round(r2, 3))
        if best is None or r + r2 < best[0]: best = (r + r2, Th)
    res[f'{h/1e3:.0f}km'] = dict(T_n_h=best[1], wn=1/(best[1]*3600))
json.dump(res, open('../data/pid_tuning.json', 'w'), indent=1); print(res)
