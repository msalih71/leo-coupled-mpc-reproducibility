"""Derive the forecast-uncertainty profile sigma_c(k) (95th percentile of the
cumulative relative density-prediction error with online bias correction) from
storm windows INSIDE the calibration period (2011-2019, test storms excluded)."""
import numpy as np, json
from tim import TIM, SW
from orbit_env import orbit_track, msis_density
from validate_tim import horizon_errors

def storm_days(n=15, start='2011-01-01', end='2020-01-01'):
    d0 = np.datetime64(start); d1 = np.datetime64(end)
    days = np.arange(d0, d1, np.timedelta64(1, 'D'))
    idx = (days - SW.day0[0]).astype(int)
    Ap = SW.ap.reshape(-1, 8).mean(1)[idx]
    order = np.argsort(-Ap); chosen = []
    for j in order:
        dd = days[j]
        if np.datetime64('2015-03-10') <= dd < np.datetime64('2015-03-25'): continue
        if all(abs((dd - c).astype(int)) > 3 for c in chosen): chosen.append(dd)
        if len(chosen) == n: break
    return sorted(chosen)

if __name__ == '__main__':
    res = {}
    days = storm_days()
    print([str(d) for d in days])
    for h in (550e3, 400e3):
        tim = TIM.load(h/1e3); allc, allp, alll = [], [], []
        for d in days:
            tr = orbit_track(d.astype('datetime64[m]') - np.timedelta64(24, 'h'), 72*3600, 60.0, h_mean=h)
            r = msis_density(tr, 0)
            er, ec, eq = horizon_errors(tim, tr, r, h, stride=10)
            allc.append(eq); allp.append(ec); alll.append(horizon_errors.lam_errs)
        eq = np.concatenate(allc); ec = np.concatenate(allp); el = np.concatenate(alll)
        res[f'{h/1e3:.0f}km'] = dict(sigma_cum_p95=np.percentile(np.abs(eq), 95, axis=0).tolist(),
                                     sigma_cum_p99=np.percentile(np.abs(eq), 99, axis=0).tolist(),
                                     sigma_step_p95=np.percentile(np.abs(ec), 95, axis=0).tolist(),
                                     sigma_lam_p95=np.percentile(np.abs(el), 95, axis=0).tolist(),
                                     sigma_lam_p99=np.percentile(np.abs(el), 99, axis=0).tolist(), lam=0.8,
                                     n=int(eq.shape[0]), storm_days=[str(x) for x in days])
        print(h, 'cum p95 k=1,6,12,24,36:', [round(res[f'{h/1e3:.0f}km']['sigma_cum_p95'][k-1], 3) for k in (1, 6, 12, 24, 36)],
              'p99:', [round(res[f'{h/1e3:.0f}km']['sigma_cum_p99'][k-1], 3) for k in (1, 12, 36)],
              'lam p95:', [round(res[f'{h/1e3:.0f}km']['sigma_lam_p95'][k-1], 3) for k in (1, 6, 12, 24, 36)])
    json.dump(res, open('../data/sigma_profile.json', 'w'), indent=1)
