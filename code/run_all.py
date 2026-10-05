"""Batch evaluation: 3 observed storms x 2 altitudes x 6 controllers x N_SEEDS
measurement-noise realisations.  Outputs ../data/results_metrics.json and
../data/results_timeseries.pkl (seed 0)."""
import numpy as np, json, pickle, sys, os
from multiprocessing import Pool
from swmpc_sim import Scenario, run, metrics, PAR
from tim import SW

EVENTS = {'Feb2022': '2022-02-02T00:00', 'Mar2015': '2015-03-16T00:00', 'May2024': '2024-05-09T12:00'}
ALTS = (550e3, 400e3)
CTRLS = ['pid', 'pid_ff', 'rule', 'mpc_thrust', 'swmpc', 'swmpc_nosw']
N_SEEDS = int(os.environ.get('N_SEEDS', 10))
SIG = json.load(open('../data/sigma_profile.json'))
PID = json.load(open('../data/pid_tuning.json'))

def storm_window(t0):
    t = np.datetime64(t0) + np.arange(0, 72*3600, 10800)*np.timedelta64(1, 's')
    ap = SW.ap_at(t); idx = np.where(ap >= 39)[0]
    return (max(0.0, idx[0]*3 - 3.0), min(72.0, idx[-1]*3 + 3 + 12.0))

_cache = {}
def job(args):
    ev, h, ctrl, seed, extra = args
    key = (ev, h, extra.get('truth', 0))
    if key not in _cache:
        _cache[key] = Scenario(ev, EVENTS[ev], h, truth_version=extra.get('truth', 0))
    scn = _cache[key]
    tag = f'{h/1e3:.0f}km'
    o = run(scn, ctrl, seed=seed, sigma_prof=SIG[tag]['sigma_lam_p95'], pid_wn=PID[tag]['wn'],
            use_sw=(ctrl != 'swmpc_nosw'), par=extra.get('par'))
    m = metrics(o, storm=storm_window(EVENTS[ev]))
    return args, m, (o if seed == 0 else None)

if __name__ == '__main__':
    jobs = [(ev, h, c, s, {}) for ev in EVENTS for h in ALTS for c in CTRLS for s in range(N_SEEDS)]
    with Pool(int(os.environ.get('NPROC', 4))) as pool:
        out = pool.map(job, jobs, chunksize=4)
    res = {}; ts = {}
    for (ev, h, c, s, _), m, o in out:
        res.setdefault(f'{ev}_{h/1e3:.0f}km', {}).setdefault(c, []).append(m)
        if o is not None: ts[(ev, h, c)] = o
    json.dump(dict(results=res, storm_windows={ev: storm_window(t) for ev, t in EVENTS.items()},
                   n_seeds=N_SEEDS, par={k: (float(v) if np.isscalar(v) else v) for k, v in PAR.items()}),
              open('../data/results_metrics.json', 'w'), indent=1)
    pickle.dump(ts, open('../data/results_timeseries.pkl', 'wb'))
    for sc, d in res.items():
        print('==', sc)
        for c, ms in d.items():
            f = lambda k: (np.mean([m[k] for m in ms]), np.std([m[k] for m in ms]))
            print('  %-11s rms %6.2f±%.2f  max %6.2f  minx %7.2f  fuel %7.2f±%.2f  viol %5.1f  minA %.2f  minE %6.0f  gloss %.2f' % (
                c, *f('rms'), f('maxabs')[0], f('min_x')[0], *f('fuel_g'), f('n_floor_viol')[0], f('minA')[0], f('minE')[0], f('mean_gen_loss')[0]))
