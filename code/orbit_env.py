"""Orbit geometry and 'truth' thermosphere along a circular LEO orbit.

Truth density = NRLMSISE-00 (pymsis, version=0) or NRLMSIS 2.1 (version=2.1)
driven by observed F10.7 / 3-hourly ap (storm-time ap mode), evaluated at the geodetic position of a
circular 53-deg orbit with J2 nodal regression.  All geometry uses standard
low-precision formulas (Vallado 2013): GMST (IAU-82), solar position
(Astronomical Almanac low-precision), WGS-84 geodetic conversion."""
import numpy as np
from sw_data import setup_pymsis

MU = 3.986004418e14; RE_MEAN = 6_371_000.0; RE_EQ = 6_378_137.0
F_WGS = 1/298.257223563; J2 = 1.08262668e-3; OMEGA_E = 7.2921150e-5

def jd_from_dt64(t):
    return (t - np.datetime64('2000-01-01T12:00:00')) / np.timedelta64(1, 's') / 86400.0 + 2451545.0

def gmst(jd):
    T = (jd - 2451545.0) / 36525.0
    s = 67310.54841 + (876600*3600 + 8640184.812866)*T + 0.093104*T**2 - 6.2e-6*T**3
    return np.mod(np.deg2rad(s / 240.0), 2*np.pi)

def sun_eci(jd):
    n = jd - 2451545.0
    L = np.deg2rad(np.mod(280.460 + 0.9856474*n, 360)); g = np.deg2rad(np.mod(357.528 + 0.9856003*n, 360))
    lam = L + np.deg2rad(1.915)*np.sin(g) + np.deg2rad(0.020)*np.sin(2*g)
    eps = np.deg2rad(23.439 - 4e-7*n)
    return np.stack([np.cos(lam), np.cos(eps)*np.sin(lam), np.sin(eps)*np.sin(lam)], -1)

def ecef_to_geodetic(x, y, z):
    e2 = F_WGS*(2-F_WGS); lon = np.arctan2(y, x); p = np.hypot(x, y)
    lat = np.arctan2(z, p*(1-e2))
    for _ in range(6):
        N = RE_EQ/np.sqrt(1-e2*np.sin(lat)**2)
        h = p/np.cos(lat) - N
        lat = np.arctan2(z, p*(1 - e2*N/(N+h)))
    N = RE_EQ/np.sqrt(1-e2*np.sin(lat)**2)
    return lat, lon, p/np.cos(lat) - N

def orbit_track(t0, dur_s, dt_s, h_mean=550e3, inc_deg=53.0, raan0_deg=0.0, u0_deg=0.0):
    """Returns dict of arrays sampled every dt_s over dur_s from datetime64 t0."""
    t = np.arange(0.0, dur_s + 0.5*dt_s, dt_s)
    a = RE_MEAN + h_mean; n = np.sqrt(MU/a**3); inc = np.deg2rad(inc_deg)
    raan_dot = -1.5*n*J2*(RE_EQ/a)**2*np.cos(inc)
    u_dot = n   # J2 drift of the argument of latitude neglected (secular, <0.1% of n)
    raan = np.deg2rad(raan0_deg) + raan_dot*t; u = np.deg2rad(u0_deg) + u_dot*t
    r_eci = a*np.stack([np.cos(raan)*np.cos(u) - np.sin(raan)*np.sin(u)*np.cos(inc),
                        np.sin(raan)*np.cos(u) + np.cos(raan)*np.sin(u)*np.cos(inc),
                        np.sin(u)*np.sin(inc)], -1)
    v_eci = a*u_dot*np.stack([-np.cos(raan)*np.sin(u) - np.sin(raan)*np.cos(u)*np.cos(inc),
                              -np.sin(raan)*np.sin(u) + np.cos(raan)*np.cos(u)*np.cos(inc),
                              np.cos(u)*np.sin(inc)], -1)
    dts = t0 + (t*1e3).astype('timedelta64[ms]')
    jd = jd_from_dt64(dts); th = gmst(jd)
    x = np.cos(th)*r_eci[:, 0] + np.sin(th)*r_eci[:, 1]
    y = -np.sin(th)*r_eci[:, 0] + np.cos(th)*r_eci[:, 1]
    z = r_eci[:, 2]
    lat, lon, alt = ecef_to_geodetic(x, y, z)
    s = sun_eci(jd)
    dec = np.arcsin(s[:, 2]); ra_sun = np.arctan2(s[:, 1], s[:, 0])
    ra_sat = np.arctan2(r_eci[:, 1], r_eci[:, 0])
    hour_angle = np.mod(ra_sat - ra_sun + np.pi, 2*np.pi) - np.pi   # 0 at local noon
    lst_h = np.mod(12 + np.rad2deg(hour_angle)/15.0, 24)
    w = np.cross(np.array([0, 0, OMEGA_E]), r_eci)
    vrel = v_eci - w
    corot = np.sum(vrel**2, -1)/np.sum(v_eci**2, -1)       # |v_rel|^2 / |v|^2
    return dict(t=t, dt64=dts, lat=lat, lon=lon, alt=alt, lst_h=lst_h, dec=dec,
                hour_angle=hour_angle, corot=corot, a=a, v=a*u_dot, r_eci=r_eci, sun_unit=s)

def msis_density(track, version=0, dalt_km=0.0):
    setup_pymsis()
    from pymsis import msis
    out = msis.calculate(track['dt64'].astype('datetime64[ms]'), np.rad2deg(track['lon']),
                         np.rad2deg(track['lat']), track['alt']/1e3 + dalt_km,
                         version=version, geomagnetic_activity=-1)   # storm-time (3-hourly ap) mode
    return np.asarray(out[..., 0]).ravel()

if __name__ == '__main__':
    import time
    tr = orbit_track(np.datetime64('2015-03-16T00:00'), 72*3600, 60.0)
    t = time.time(); rho = msis_density(tr); print('msis time', time.time()-t, len(rho))
    print('alt range km', tr['alt'].min()/1e3, tr['alt'].max()/1e3, 'corot', tr['corot'].min(), tr['corot'].max())
    print('rho quiet/peak', rho[:1440].mean(), rho.max())


def cylindrical_sunlight(track):
    """Binary illumination: parallel sunlight, spherical Earth, no penumbra."""
    r=track['r_eci']; s=track['sun_unit']
    along=np.einsum('ij,ij->i',r,s)
    distance2=np.einsum('ij,ij->i',r,r)-along**2
    return (~((along<0)&(distance2<RE_EQ**2))).astype(float)
