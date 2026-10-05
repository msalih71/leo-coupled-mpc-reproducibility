"""Empirical coverage of the sigma_k bound on the three test storms."""
import numpy as np, json
from tim import TIM
from orbit_env import orbit_track, msis_density
from validate_tim import horizon_errors, EVENTS
sig = json.load(open('../data/sigma_profile.json'))
cov = {}
for h in (550e3, 400e3):
    tim = TIM.load(h/1e3); s = np.array(sig[f'{h/1e3:.0f}km']['sigma_lam_p95'])
    for name, t0 in EVENTS.items():
        tr = orbit_track(np.datetime64(t0), 72*3600, 60.0, h_mean=h)
        for truth in (0, 2.1):
            r = msis_density(tr, truth)
            horizon_errors(tim, tr, r, h, stride=5)
            c = float(np.mean(np.abs(horizon_errors.lam_errs) <= s[None, :]))
            cov[f'{name}_{h/1e3:.0f}km_msis{truth}'] = c
            print(name, h, truth, 'coverage %.3f' % c)
json.dump(cov, open('../data/sigma_coverage_test.json', 'w'), indent=1)
