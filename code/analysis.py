"""Derived numbers quoted in the manuscript (written to ../data/analysis.json)."""
import numpy as np, json, pickle, platform, subprocess
from swmpc_sim import PAR, Scenario, G0
from run_all import EVENTS
R = json.load(open('../data/results_metrics.json'))['results']
SIG = json.load(open('../data/sigma_profile.json'))
out = {}
# ---- terminal-set condition (stationary density = peak 5-min truth density) ----
p = PAR
for ev, t0 in EVENTS.items():
    for h in (550e3, 400e3):
        scn = Scenario(ev, t0, h)
        n = int(84*3600/60)
        r5 = scn.rho[:n - n % 5].reshape(-1, 5).mean(1)
        rho_pk = r5.max(); sbar = max(SIG[f'{h/1e3:.0f}km']['sigma_lam_p95'])
        Dunit = 0.5*rho_pk*p['Cd']*scn.v_ref**2*scn.tr['corot'].mean()     # N per m^2
        feas = []
        for A in np.linspace(p['A_min'], p['A_nom'], 801):
            uw = (1+2*sbar)*Dunit*A    # terminal thrust: (1+sigma) D cancels worst-case drag, + K*delta_inf = sigma D
            ok = uw <= p['u_max'] and p['P_gen']*(A-p['A_min'])/(p['A_nom']-p['A_min']) >= p['P_load'] + p['kp_power']*uw
            if ok: feas.append(A)
        out[f'terminal_{ev}_{h/1e3:.0f}km'] = dict(rho_peak_5min=float(rho_pk), sigma_N=sbar,
            D_peak_Anom_mN=float(Dunit*p['A_nom']*1e3), terminal_set_nonempty=bool(feas),
            A_range=[float(min(feas)), float(max(feas))] if feas else None)
# ---- fuel decomposition: SW-MPC vs thrust-only MPC ---------------------------
for sc, d in R.items():
    def mean(c, k): return float(np.mean([m[k] for m in d[c]]))
    I_sp_g0 = p['Isp']*G0
    dec = {}
    for c in d:
        dec[c] = dict(fuel_g=mean(c, 'fuel_g'), drag_imp_Ns=mean(c, 'drag_imp'), thr_imp_Ns=mean(c, 'thr_imp'),
                      net_Ns=mean(c, 'thr_imp') - mean(c, 'drag_imp'), x_end=mean(c, 'x_end'))
    df = dec['mpc_thrust']['fuel_g'] - dec['swmpc']['fuel_g']
    dd = (dec['mpc_thrust']['drag_imp_Ns'] - dec['swmpc']['drag_imp_Ns'])/I_sp_g0*1e3
    dn = (dec['mpc_thrust']['net_Ns'] - dec['swmpc']['net_Ns'])/I_sp_g0*1e3
    out[f'fuel_{sc}'] = dict(per_ctrl=dec, saving_vs_thrustMPC_g=df, from_drag_impulse_g=dd, from_net_altitude_term_g=dn,
                             saving_vs_thrustMPC_pct=100*df/dec['mpc_thrust']['fuel_g'],
                             saving_vs_pid_pct=100*(dec['pid']['fuel_g']-dec['swmpc']['fuel_g'])/dec['pid']['fuel_g'],
                             rule_vs_pid_pct=100*(dec['pid']['fuel_g']-dec['rule']['fuel_g'])/dec['pid']['fuel_g'])
# ---- timing ------------------------------------------------------------------
sm = [m for sc in R.values() for c in ('mpc_thrust', 'swmpc', 'swmpc_nosw') for m in sc[c]]
out['timing'] = dict(solve_median_ms=float(np.median([m['solve_med_ms'] for m in sm])),
                     solve_p99_ms_max=float(np.max([m['solve_p99_ms'] for m in sm])),
                     solve_max_ms=float(np.max([m['solve_max_ms'] for m in sm])),
                     loop_median_ms=float(np.median([m['loop_med_ms'] for m in sm])),
                     loop_p99_ms_max=float(np.max([m['loop_p99_ms'] for m in sm])),
                     loop_max_ms=float(np.max([m['loop_max_ms'] for m in sm])),
                     cpu=subprocess.run(['bash', '-c', "lscpu | grep 'Model name' | sed 's/Model name:\\s*//'"], capture_output=True, text=True).stdout.strip(),
                     python=platform.python_version())
json.dump(out, open('../data/analysis.json', 'w'), indent=1)
for k, v in out.items():
    if k.startswith('fuel_'):
        print(k, {kk: round(vv, 3) for kk, vv in v.items() if kk != 'per_ctrl'})
    else: print(k, v)
