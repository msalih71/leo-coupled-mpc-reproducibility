"""R7 fixed common-recovery comparison; protocol is ../R7_COMPARISON_PROTOCOL.md."""
import hashlib
import json
import os
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from operational_calibration import ORIGINAL, PID
from orbit_env import cylindrical_sunlight, orbit_track
from swmpc_sim import PAR, Scenario, metrics, run

CONTROLLERS = ('pid_lowdrag_preview', 'swmpc', 'swmpc_nosw')
SEEDS = tuple(range(5))
PROTOCOL_HASH = hashlib.sha256(Path('../R7_COMPARISON_PROTOCOL.md').read_bytes()).hexdigest()


def completion_hour(event, altitude, phase):
    t0 = np.datetime64(ORIGINAL[event]) - np.timedelta64(12*3600, 's')
    tr = orbit_track(t0, 112*3600, 60., h_mean=altitude, u0_deg=phase)
    sunlight = cylindrical_sunlight(tr)
    for step in range(108*12, 112*12):
        # step is an endpoint, not the timestamp of the preceding command.
        if (np.all(sunlight[(step-1)*5:step*5] == 1) and
                not np.all(sunlight[step*5:(step+1)*5] == 1)):
            return step/12. - 12.
    raise RuntimeError('No fully sunlit completion step in predefined search window')


@lru_cache(None)
def scenario(event, altitude, phase):
    end = completion_hour(event, altitude, phase)
    return Scenario(event, ORIGINAL[event], altitude, dur_h=12.+end,
                    orbit_phase_deg=phase)


def truncated_metrics(out, end_h, start_h):
    n = int(np.count_nonzero(out['t_h'] < end_h))
    cut = {k: (v[:n] if isinstance(v, np.ndarray) and len(v) == len(out['t_h']) else v)
           for k, v in out.items()}
    attempted = cut['qp_attempt'].astype(bool)
    cut['solve_ms_all'] = cut['solve_ms'][attempted]
    cut['loop_ms_all'] = out['loop_ms_all'][:n]
    return metrics(cut, t_from=start_h)


def job(args):
    event, altitude, payload, phase, controller, seed = args
    sc = scenario(event, altitude, phase)
    time_h = sc.tr['t']/3600-sc.spinup_h
    sun = cylindrical_sunlight(sc.tr)
    load = 600.+200.*((time_h >= 30)&(time_h < 36)) if payload else np.full(len(time_h), 600.)
    sigma = json.loads(Path('../data/operational_sigma_profile.json').read_text())[f'{altitude/1000:.0f}km']['sigma_lam_p95']
    out = run(sc, controller, seed=seed, sigma_prof=sigma,
              pid_wn=PID[f'{altitude/1000:.0f}km']['wn'], strict_solar=True,
              use_sw=controller != 'swmpc_nosw', generation_W=1500.*sun,
              load_W=load, recovery_start_h=72.)
    row = dict(protocol_sha256=PROTOCOL_HASH, event=event, h_km=altitude/1000, payload=payload,
               phase_deg=phase, ctrl=controller, seed=seed,
               completion_event_h=completion_hour(event, altitude, phase),
               full=metrics(out, t_from=-12.),
               event_window=truncated_metrics(out, 72., 0.),
               before_recovery=truncated_metrics(out, 72., -12.),
               recovery=metrics(out, t_from=72.))
    # Only geometry-designated examples; all metrics are retained for every run.
    if seed == 0 and not payload and ((event == 'May2024' and altitude == 400e3 and phase == 0.) or
                                     (event == 'Mar2015' and altitude == 550e3 and phase == 180.)):
        name = f'{event}_{altitude/1000:.0f}_{phase:.0f}_{controller}'
        folder = Path('../data/common_recovery_traces'); folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/(name+'.npz'), **{k: out[k] for k in
            ('t_h', 'x', 'x_after', 'A', 'E', 'u', 'fuel', 'qp_attempt', 'qp_ok')})
    return row


def qualified(a, b):
    valid = all(x['powered_regime_valid'] and x['n_floor_viol'] == 0 and
                x['n_energy_floor_viol'] == 0 and x['unserved_bus_Wh'] == 0 for x in (a, b))
    return bool(valid and min(a['E_end_Wh'], b['E_end_Wh']) >= 1495. and
                abs(a['E_end_Wh']-b['E_end_Wh']) <= 5. and
                abs(a['x_end']-b['x_end']) <= 1.)


def summarize(rows):
    records=[]
    for row in rows:
        record={k:v for k,v in row.items() if k not in ('full', 'event_window', 'before_recovery', 'recovery')}
        for window in ('full', 'event_window', 'before_recovery', 'recovery'):
            record.update({window+'_'+k:v for k,v in row[window].items()})
        records.append(record)
    pd.DataFrame(records).to_csv('../data/common_recovery_per_seed.csv', index=False)
    bykey={(r['event'], r['h_km'], r['payload'], r['phase_deg'], r['ctrl'], r['seed']):r for r in rows}
    pairs=[]; ablations=[]
    for r in rows:
        if r['ctrl'] != 'swmpc': continue
        k=(r['event'],r['h_km'],r['payload'],r['phase_deg'])
        pi=bykey[(*k, 'pid_lowdrag_preview', r['seed'])]
        no=bykey[(*k, 'swmpc_nosw', r['seed'])]
        base={x:r[x] for x in ('event','h_km','payload','phase_deg','seed','completion_event_h')}
        pair=dict(base, resource_matched=qualified(pi['full'], r['full']),
                  saving_full_pct=100*(pi['full']['fuel_g']-r['full']['fuel_g'])/pi['full']['fuel_g'],
                  endpoint_difference_Wh=abs(pi['full']['E_end_Wh']-r['full']['E_end_Wh']),
                  endpoint_altitude_difference_m=abs(pi['full']['x_end']-r['full']['x_end']))
        for label, row in (('pi', pi), ('mpc', r)):
            for window in ('full','event_window','before_recovery','recovery'):
                for metric in ('fuel_g','rms','minE','E_start_Wh','E_end_Wh','x_end','qp_attempts_all','qp_failures_all'):
                    pair[label+'_'+window+'_'+metric]=row[window][metric]
        pairs.append(pair)
        ablations.append(dict(base, resource_matched=qualified(no['full'], r['full']),
             weather_fuel_saving_pct=100*(no['full']['fuel_g']-r['full']['fuel_g'])/no['full']['fuel_g'],
             weather_rms_improvement_pct=100*(no['event_window']['rms']-r['event_window']['rms'])/no['event_window']['rms'],
             mpc_event_rms=r['event_window']['rms'], frozen_event_rms=no['event_window']['rms']))
    pairs=pd.DataFrame(pairs); ablations=pd.DataFrame(ablations)
    pairs.to_csv('../data/common_recovery_pairs.csv',index=False)
    ablations.to_csv('../data/common_recovery_ablation.csv',index=False)
    summary=pairs.groupby(['event','h_km','payload','phase_deg']).agg(
        n_pairs=('seed','size'),n_matched=('resource_matched','sum'),
        saving_mean_pct=('saving_full_pct','mean'),saving_min_pct=('saving_full_pct','min'),saving_max_pct=('saving_full_pct','max'),
        pi_event_rms=('pi_event_window_rms','mean'),mpc_event_rms=('mpc_event_window_rms','mean'),
        pi_recovery_fuel_g=('pi_recovery_fuel_g','mean'),mpc_recovery_fuel_g=('mpc_recovery_fuel_g','mean'),
        pi_minE=('pi_full_minE','min'),mpc_minE=('mpc_full_minE','min'),
        endpoint_difference_max_Wh=('endpoint_difference_Wh','max'),
        endpoint_altitude_difference_max_m=('endpoint_altitude_difference_m','max'),
        qp_failures=('mpc_full_qp_failures_all','sum'))
    summary.to_csv('../data/common_recovery_summary.csv')
    print(summary.to_string(),flush=True)
    print('MATCHED',int(pairs.resource_matched.sum()),'/',len(pairs),flush=True)
    print('ABLATION RANGE',ablations.weather_fuel_saving_pct.min(),ablations.weather_fuel_saving_pct.max(),flush=True)


if __name__ == '__main__':
    protocol=Path('../R7_COMPARISON_PROTOCOL.md')
    Path('../R7_PROTOCOL_SHA256.txt').write_text(hashlib.sha256(protocol.read_bytes()).hexdigest()+'  R7_COMPARISON_PROTOCOL.md\n')
    jobs=[(e,h,l,ph,c,s) for e in ORIGINAL for h in (400e3,550e3) for l in (False,True)
          for ph in (0.,180.) for c in CONTROLLERS for s in SEEDS]
    cp=Path('../data/common_recovery_checkpoint.json')
    rows=json.loads(cp.read_text()) if cp.exists() else []
    assert all(r.get('protocol_sha256') == PROTOCOL_HASH and
               r['completion_event_h'] == completion_hour(r['event'],r['h_km']*1000,r['phase_deg'])
               for r in rows), 'Stale checkpoint: use the same protocol and completion rule'
    done={(r['event'],r['h_km']*1000,r['payload'],r['phase_deg'],r['ctrl'],r['seed']) for r in rows}
    todo=[j for j in jobs if j not in done]
    with Pool(int(os.environ.get('NPROC','4'))) as pool:
        for row in pool.imap(job,todo,chunksize=1):
            rows.append(row); cp.write_text(json.dumps(rows,indent=2,allow_nan=False))
            if len(rows)%15==0: print('Completed',len(rows),'/',len(jobs),flush=True)
    rows.sort(key=lambda r:jobs.index((r['event'],r['h_km']*1000,r['payload'],r['phase_deg'],r['ctrl'],r['seed'])))
    Path('../data/common_recovery_campaign.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    summarize(rows)
