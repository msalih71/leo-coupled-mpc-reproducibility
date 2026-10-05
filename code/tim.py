"""Enhanced Thermal-Inertia Model (TIM), revised.

ln rho_hat = c0 + c1 x + c1q x^2 + c2 (F - F81)/100 + (a_s + a_sx x) ln(1 + gamma*T)
           + cH (h_gd - h_ref)/1 km + b_c cos(lat)cos(dec)cos(HA)
           + b_s cos(lat)cos(dec)sin(HA) + b_z sin(lat)sin(dec)
           + a1c cos(w) + a1s sin(w) + a2c cos(2w) + a2s sin(2w)
x = (F81 - 120)/100,  w = 2 pi doy/365.25 (annual/semiannual terms)
dT/dt = -T/tau_c + ap(t - tau_d)           [T in nT h]

F   : previous-day observed F10.7, F81: trailing 81-day mean (causal inputs)
HA  : local solar hour angle of the satellite (0 = noon); the three bulge
      terms are the log-linear expansion of a Jacchia-type bulge
      cos(psi) = sin(lat) sin(dec) + cos(lat) cos(dec) cos(HA - HA_apex).
All inputs are available onboard from the predicted orbit and uplinked indices.
Linear coefficients are fitted by least squares; (tau_c, tau_d, gamma) by grid.
"""
import numpy as np, json, os
from spacewx import SpaceWx
from orbit_env import orbit_track, msis_density

SW = SpaceWx()
LIN_NAMES = ['c0', 'c1', 'c1q', 'c2', 'cH', 'b_c', 'b_s', 'b_z', 'a1c', 'a1s', 'a2c', 'a2s']

def thermal_state(t_s, ap_series, tau_c_h, T0=None):
    """Exact ZOH discretisation of dT/dt = -T/tau_c + ap on grid t_s."""
    tau = tau_c_h*3600.0
    T = np.empty_like(ap_series, dtype=float)
    Tk = ap_series[0]*tau_c_h if T0 is None else T0     # start in equilibrium
    dt = np.diff(t_s, prepend=t_s[0])
    for i in range(len(ap_series)):
        e = np.exp(-dt[i]/tau)
        Tk = e*Tk + (1-e)*tau_c_h*ap_series[i - 1 if i > 0 else 0]
        T[i] = Tk
    return T

def geo_features(tr, h_ref):
    lat, dec, HA = tr['lat'], tr['dec'], tr['hour_angle']
    F, F81 = SW.f107_daily(tr['dt64'])
    doy = (tr['dt64'].astype('datetime64[s]') - tr['dt64'].astype('datetime64[Y]')).astype(float)/86400.0
    w = 2*np.pi*doy/365.25
    x1 = (F81-120)/100.0
    return np.stack([np.ones_like(lat), x1, x1**2, (F-F81)/100.0,
                     (tr['alt']-h_ref)/1e3,
                     np.cos(lat)*np.cos(dec)*np.cos(HA), np.cos(lat)*np.cos(dec)*np.sin(HA),
                     np.sin(lat)*np.sin(dec),
                     np.cos(w), np.sin(w), np.cos(2*w), np.sin(2*w)], -1)

class TIM:
    def __init__(self, params):
        self.p = params
        self.w = np.array([params[k] for k in LIN_NAMES])

    @classmethod
    def load(cls, h_km):
        fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', f'tim_params_{h_km:.0f}km.json')
        return cls(json.load(open(fn))['params'])

    def storm(self, T, x):
        return (self.p['a_s'] + self.p['a_sx']*x)*np.log1p(self.p['gamma']*T)

    def log_rho(self, X, T):
        return X @ self.w + self.storm(T, X[:, 1])

    def hindcast(self, tr, h_ref):
        X = geo_features(tr, h_ref)
        ap = SW.ap_at(tr['dt64'] - np.timedelta64(int(self.p['tau_d_h']*3600), 's'))
        T = thermal_state(tr['t'], ap, self.p['tau_c_h'])
        return np.exp(self.log_rho(X, T)), T

def calibrate(h_ref, years=('2011-01-01', '2020-01-01'), dt=1800.0,
              exclude=(('2015-03-10', '2015-03-25'),)):
    t0 = np.datetime64(years[0]); dur = (np.datetime64(years[1]) - t0)/np.timedelta64(1, 's')
    tr = orbit_track(t0, dur, dt, h_mean=h_ref)
    rho = msis_density(tr, 0)
    X = geo_features(tr, h_ref)
    mask = np.ones(len(rho), bool)
    for a, b in exclude:
        mask &= ~((tr['dt64'] >= np.datetime64(a)) & (tr['dt64'] < np.datetime64(b)))
    y = np.log(rho)
    best = None
    for tau_d in [0.0, 1.0, 2.0, 3.0, 4.0]:
        ap = SW.ap_at(tr['dt64'] - np.timedelta64(int(tau_d*3600), 's'))
        for tau_c in [2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]:
            T = thermal_state(tr['t'], ap, tau_c)
            for gamma in np.geomspace(1e-4, 1.0, 33):
                g = np.log1p(gamma*T)
                Xa = np.column_stack([X, g, g*X[:, 1]])
                w, *_ = np.linalg.lstsq(Xa[mask], y[mask], rcond=None)
                r = (y - Xa @ w)[mask]
                sse = float(r @ r)
                if best is None or sse < best[0]:
                    best = (sse, tau_d, tau_c, gamma, w)
    sse, tau_d, tau_c, gamma, w = best
    params = dict(zip(LIN_NAMES + ['a_s', 'a_sx'], map(float, w)))
    params.update(tau_d_h=tau_d, tau_c_h=tau_c, gamma=float(gamma), h_ref_m=h_ref)
    rms = np.sqrt(sse/mask.sum())
    return params, dict(n_samples=int(mask.sum()), rms_log_resid=float(rms),
                        calib_period=list(years), cadence_s=dt,
                        excluded=[list(e) for e in exclude])

if __name__ == '__main__':
    import time
    for h in (550e3, 400e3):
        t = time.time()
        p, info = calibrate(h)
        print(h, p, info, '%.0fs' % (time.time()-t))
        fn = os.path.join('..', 'data', f'tim_params_{h/1e3:.0f}km.json')
        json.dump(dict(params=p, calibration=info), open(fn, 'w'), indent=2)

# ---------------------------------------------------------------------------
# Causal onboard predictor
# ---------------------------------------------------------------------------
from scipy.signal import lfilter

class TIMPredictor:
    """Onboard use: at time t_now predicts step-averaged density over a horizon
    using only ap values available at t_now (latency `lat_s` after the end of
    each 3-h interval; unknown intervals held at the last known value)."""
    def __init__(self, tim, track, h_ref, lat_s=3600.0, hist_h=48.0, strict_solar=False):
        self.tim, self.tr, self.h_ref, self.lat_s = tim, track, h_ref, lat_s
        self.strict_solar = strict_solar
        self.dt = track['t'][1] - track['t'][0]
        self.X = geo_features(track, h_ref)
        self.logrho_geo = self.X @ tim.w
        self.hist_n = int(hist_h*3600/self.dt)
        e = np.exp(-self.dt/(tim.p['tau_c_h']*3600)); self.e = e
        self.tau_d = np.timedelta64(int(tim.p['tau_d_h']*3600), 's')

    def thermal(self, i_now, n_ahead):
        """Thermal state on grid i_now-hist .. i_now+n_ahead using causal ap."""
        i0 = max(0, i_now - self.hist_n); i1 = min(len(self.tr['t']), i_now + n_ahead + 1)
        t_now = self.tr['dt64'][i_now] if getattr(self, 'freeze_t', None) is None else self.freeze_t
        tq = self.tr['dt64'][i0:i1] - self.tau_d
        ap = SW.ap_known(tq, t_now, self.lat_s)
        tc = self.tim.p['tau_c_h']
        zi = [self.e*ap[0]*tc]      # equilibrium start
        T, _ = lfilter([0.0, (1-self.e)*tc], [1.0, -self.e], ap, zi=zi)
        return i0, T

    def predict_steps(self, i_now, n_steps, sub):
        """Mean density over each of n_steps control steps of `sub` samples."""
        n_ahead = n_steps*sub
        i0, T = self.thermal(i_now, n_ahead)
        idx = np.arange(i_now, i_now + n_ahead)
        idx = np.minimum(idx, len(self.tr['t'])-1)
        Xf = self.X[idx].copy()
        if self.strict_solar:
            # Future daily solar data are unavailable at issue time. Persist
            # today's available previous-day F10.7/trailing-mean features.
            source_i = i_now if getattr(self, 'freeze_t', None) is None else int(np.searchsorted(self.tr['dt64'], self.freeze_t))
            Xf[:, 1:4] = self.X[source_i, 1:4]
        lr = Xf @ self.tim.w + self.tim.storm(T[np.minimum(idx - i0, len(T)-1)], Xf[:, 1])
        return np.exp(lr).reshape(n_steps, sub).mean(1)
