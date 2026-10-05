"""Validation of the revised TIM and of the original-submission TIM against
NRLMSISE-00 and NRLMSIS 2.1 on three out-of-sample storm windows."""
import numpy as np, json
from tim import TIM, TIMPredictor, SW
from orbit_env import orbit_track, msis_density

EVENTS = {'Feb2022_G1G2': '2022-02-02T00:00', 'Mar2015_G4': '2015-03-16T00:00',
          'May2024_G5': '2024-05-09T12:00'}
DT, DUR = 60.0, 72*3600.0
from orbit_env import MU, RE_MEAN

def original_tim(tr):
    """TIM of the original submission (rho0=1.5e-12, f_storm, f_temp, bulge in mission time)."""
    t = tr['t']; ap_all = SW.ap_at(tr['dt64'])
    def fstorm(a):
        return np.where(a < 20, 1.0, np.where(a < 50, 1+0.5*(a-20)/30, np.where(a < 100, 1.5+0.5*(a-50)/50,
               np.where(a < 200, 2+3*(a-100)/100, 5.0))))
    apd = SW.ap_at(tr['dt64'] - np.timedelta64(7200, 's'))
    T = np.zeros_like(t); Tk = 0.0
    for i in range(len(t)):
        Tk = np.clip(Tk + (-Tk/(6*3600) + 9e-5*apd[i])*DT, 0, 500); T[i] = Tk
    F, _ = SW.f107_daily(tr['dt64'])
    rho = 1.5e-12*fstorm(apd)*(1+0.3*np.cos(2*np.pi*t/5760-np.pi/4))*(1+0.002*(F-70))*(1+5e-4*T)
    return np.clip(rho, 0.75e-12, 1.5e-11)

def stats(ratio):
    lr = np.log(ratio)
    return dict(mean_log=float(lr.mean()), rms_log=float(np.sqrt((lr**2).mean())),
                p95_abs_rel=float(np.percentile(np.abs(ratio-1), 95)),
                median_ratio=float(np.median(ratio)))

def horizon_errors(tim, tr, rho_true, h_ref, n_steps=36, sub=5, stride=5, lam=0.8, harmonic=True):
    """Relative error of kappa-corrected, causal predictions vs lead step.
    kappa = mean(rho_true)/mean(rho_hat) over the preceding orbit (idealised
    drag-derived density; the closed-loop estimator is noisier)."""
    P = TIMPredictor(tim, tr, h_ref)
    T_ORB = 2*np.pi*np.sqrt((RE_MEAN + h_ref)**3/MU)
    orbit = int(round(T_ORB/DT))
    errs_raw, errs_cor, errs_cum, errs_lam = [], [], [], []
    W = np.array([[lam**(k-j) if j <= k else 0.0 for j in range(n_steps)] for k in range(n_steps)])
    for i in range(orbit + P.hist_n//4, len(tr['t']) - n_steps*sub, stride):
        # kappa from last orbit (hindcast prediction at those samples)
        _, T = P.thermal(i - orbit, orbit)
        lr = P.logrho_geo[i-orbit:i] + tim.storm(T[(i-orbit)-max(0, i-orbit-P.hist_n):][:orbit], P.X[i-orbit:i, 1])
        pred = P.predict_steps(i, n_steps, sub)
        if harmonic:   # orbit-periodic correction kappa0 + kc cos(th) + ks sin(th), LS over last orbit
            th = 2*np.pi*tr['t']/T_ORB
            ph = np.exp(lr)
            M = np.column_stack([ph, ph*np.cos(th[i-orbit:i]), ph*np.sin(th[i-orbit:i])])
            kv, *_ = np.linalg.lstsq(M, rho_true[i-orbit:i], rcond=None)
            thf = th[i:i+n_steps*sub]
            kappa = np.maximum(0.1, (kv[0] + kv[1]*np.cos(thf) + kv[2]*np.sin(thf))).reshape(n_steps, sub).mean(1)
        else:
            kappa = rho_true[i-orbit:i].mean()/np.exp(lr).mean()
        truth = rho_true[i:i+n_steps*sub].reshape(n_steps, sub).mean(1)
        errs_raw.append(truth/pred - 1); errs_cor.append(truth/(kappa*pred) - 1)
        errs_cum.append(np.cumsum(truth)/np.cumsum(kappa*pred) - 1)
        errs_lam.append((W @ (truth - kappa*pred))/(W @ (kappa*pred)))
    horizon_errors.lam_errs = np.array(errs_lam)
    return np.array(errs_raw), np.array(errs_cor), np.array(errs_cum)

if __name__ == '__main__':
    out = {}
    series = {}
    for h in (550e3, 400e3):
        tim = TIM.load(h/1e3)
        for name, t0 in EVENTS.items():
            tr = orbit_track(np.datetime64(t0), DUR, DT, h_mean=h)
            r00 = msis_density(tr, 0); r21 = msis_density(tr, 2.1)
            rh, T = tim.hindcast(tr, h)
            key = f'{name}_{h/1e3:.0f}km'
            d = dict(TIM_vs_MSIS00=stats(rh/r00), TIM_vs_MSIS21=stats(rh/r21),
                     MSIS00_vs_MSIS21=stats(r00/r21),
                     peak_MSIS00=float(r00.max()), peak_TIM=float(rh.max()), peak_MSIS21=float(r21.max()),
                     ap_max=float(SW.ap_at(tr['dt64']).max()))
            if h == 550e3:
                ro = original_tim(tr)
                d['ORIGINAL_TIM_vs_MSIS00'] = stats(ro/r00); d['peak_ORIGINAL_TIM'] = float(ro.max())
                series[name] = dict(t_h=tr['t']/3600, msis00=r00, msis21=r21, tim=rh, orig=ro,
                                    ap=SW.ap_at(tr['dt64']))
            er, ec, eq = horizon_errors(tim, tr, r00, h)
            d['horizon_p95_abs_cum_kappa'] = np.percentile(np.abs(eq), 95, axis=0).tolist()
            d['horizon_p95_abs_raw'] = np.percentile(np.abs(er), 95, axis=0).tolist()
            d['horizon_p95_abs_kappa'] = np.percentile(np.abs(ec), 95, axis=0).tolist()
            d['horizon_median_abs_kappa'] = np.median(np.abs(ec), axis=0).tolist()
            out[key] = d
            print(key, {k: (round(v['rms_log'], 3) if isinstance(v, dict) else v) for k, v in d.items() if not k.startswith('horizon')},
                  'p95 kappa lead1/12/36: %.3f %.3f %.3f' % (d['horizon_p95_abs_kappa'][0], d['horizon_p95_abs_kappa'][11], d['horizon_p95_abs_kappa'][35]),
                  'raw: %.3f %.3f' % (d['horizon_p95_abs_raw'][0], d['horizon_p95_abs_raw'][35]),
                  'cum: %.3f %.3f %.3f' % tuple(d['horizon_p95_abs_cum_kappa'][j] for j in (0, 11, 35)))
    json.dump(out, open('../data/tim_validation.json', 'w'), indent=1)
    np.save('../data/tim_validation_series.npy', series, allow_pickle=True)
