"""Causal access to observed 3-hourly ap and daily F10.7 (CelesTrak SW-All)."""
import numpy as np, os
from sw_data import CSV, build_csv

class SpaceWx:
    def __init__(self):
        if not os.path.exists(CSV): build_csv()
        raw = np.genfromtxt(CSV, delimiter=',', skip_header=1, dtype=None, encoding=None)
        self.day0 = np.array([r[0] for r in raw], dtype='datetime64[D]')
        ap = np.array([[r[i] for i in range(12, 20)] for r in raw], float)
        self.ap = ap.ravel()                                  # 3-h ap, index j -> [day0+3h*j)
        self.t_ap = (self.day0[0].astype('datetime64[s]') + np.arange(self.ap.size)*np.timedelta64(3*3600, 's'))
        self.f107 = np.array([r[24] for r in raw], float)       # observed
        self.f107_last81 = np.array([r[28] for r in raw], float)
        self.f107_ctr81 = np.array([r[27] for r in raw], float)

    def _sec(self, t64):
        return (np.asarray(t64).astype('datetime64[s]') - self.t_ap[0]).astype(float)

    def ap_at(self, t64):
        """ap value of the 3-h interval containing t64 (non-causal hindcast value)."""
        j = np.floor(self._sec(t64)/10800.0).astype(int)
        return self.ap[j]

    def ap_known(self, t_query64, t_now64, latency_s=0.0):
        """ap at t_query as available at t_now: intervals that ended at least
        `latency_s` before t_now are known; later ones are replaced by the last
        known value (persistence forecast)."""
        jq = np.floor(self._sec(t_query64)/10800.0).astype(int)
        j_last = int(np.floor((self._sec(t_now64) - latency_s)/10800.0)) - 1  # last complete interval
        return self.ap[np.minimum(jq, j_last)]

    def f107_daily(self, t64):
        d = (np.asarray(t64).astype('datetime64[D]') - self.day0[0]).astype(int)
        return self.f107[d - 1], self.f107_last81[d - 1]       # previous-day value, trailing 81-day
