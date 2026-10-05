"""Generate LaTeX tables (numbers taken directly from the JSON outputs)."""
import json, numpy as np, pandas as pd
D = '../data/'
V = json.load(open(D + 'tim_validation.json')); COV = json.load(open(D + 'sigma_coverage_test.json'))
R = json.load(open(D + 'results_metrics.json'))['results']; S = json.load(open(D + 'sensitivity.json'))
T5 = json.load(open(D + 'tim_params_550km.json')); T4 = json.load(open(D + 'tim_params_400km.json'))
out = {}
# --- TIM parameters
names = [('c0', '$c_0$'), ('c1', '$c_1$'), ('c1q', '$c_{1q}$'), ('c2', '$c_2$'), ('cH', '$c_H$ [km$^{-1}$]'),
         ('b_c', '$b_c$'), ('b_s', '$b_s$'), ('b_z', '$b_z$'), ('a1c', '$a_{1c}$'), ('a1s', '$a_{1s}$'), ('a2c', '$a_{2c}$'),
         ('a2s', '$a_{2s}$'), ('a_s', '$a_s$'), ('a_sx', '$a_{sx}$'), ('gamma', '$\\gamma$ [(nT\\,h)$^{-1}$]'),
         ('tau_c_h', '$\\tau_c$ [h]'), ('tau_d_h', '$\\tau_d$ [h]')]
rows = [f'{lab} & {T5["params"][k]:.4g} & {T4["params"][k]:.4g} \\\\' for k, lab in names]
rows.append(f'RMS log residual & {T5["calibration"]["rms_log_resid"]:.3f} & {T4["calibration"]["rms_log_resid"]:.3f} \\\\')
out['tab_tim'] = '\n'.join(rows)
# --- validation
ev = [('Feb2022_G1G2', 'Feb 2022 (G1--G2)'), ('Mar2015_G4', 'Mar 2015 (G4)'), ('May2024_G5', 'May 2024 (G5)')]
rows = []
for h in ('550', '400'):
    for k, lab in ev:
        d = V[f'{k}_{h}km']
        rows.append(f'{h} & {lab} & {d["TIM_vs_MSIS00"]["rms_log"]:.2f} & {d["TIM_vs_MSIS21"]["rms_log"]:.2f} & '
                    f'{d["MSIS00_vs_MSIS21"]["rms_log"]:.2f} & {d["peak_TIM"]/d["peak_MSIS00"]:.2f} & '
                    f'{100*COV[f"{k}_{h}000.0_msis0".replace("000.0", "")] if False else 100*COV[k+"_"+h+"km_msis0"]:.1f} \\\\')
    if h == '550':
        rows.append('\\midrule')
orig = [V[f'{k}_550km']['ORIGINAL_TIM_vs_MSIS00']['rms_log'] for k, _ in ev]
opk = [V[f'{k}_550km']['peak_ORIGINAL_TIM']/V[f'{k}_550km']['peak_MSIS00'] for k, _ in ev]
rows.append('\\midrule')
rows.append(f'550 & Original TIM$^\\dagger$ & {min(orig):.2f}--{max(orig):.2f} & --- & --- & {min(opk):.1f}--{max(opk):.1f} & --- \\\\')
out['tab_valid'] = '\n'.join(rows)
# --- independent Swarm POD comparison (the observational supplement)
obs = pd.read_csv('../external_validation/three_model_5min_comparison.csv')
assert len(obs) == 6 and set(obs['n']) == {864}
rows = []
for ev, label in [('Feb2022','Feb. 2022'),('Mar2015','Mar. 2015'),('May2024','May 2024')]:
    for sat in ('A','B'):
        r = obs[(obs.event == ev) & (obs.satellite == sat)].iloc[0]
        rows.append(f'{label} & {sat} & {r.alt_median_km:.0f} & '
                    f'{r.TIM_rms_log:.3f} / {r.TIM_median_ratio:.2f} & '
                    f'{r.NRLMSISE00_rms_log:.3f} / {r.NRLMSISE00_median_ratio:.2f} & '
                    f'{r.JB2008_rms_log:.3f} / {r.JB2008_median_ratio:.2f} \\\\')
out['tab_swarm'] = '\n'.join(rows)
# --- closed loop
lab = dict(pid='PI', pid_ff='PI+FF', rule='Rule', mpc_thrust='MPC-T', swmpc='\\textbf{SW-MPC}', swmpc_nosw='SW-MPC$^{-}$')
for h in ('550', '400'):
    rows = []
    for e, el in (('Feb2022', 'G1--G2'), ('Mar2015', 'G4'), ('May2024', 'G5')):
        sc = R[f'{e}_{h}km']; f0 = np.mean([m['fuel_g'] for m in sc['pid']])
        first = True
        for c in ('pid', 'pid_ff', 'rule', 'mpc_thrust', 'swmpc', 'swmpc_nosw'):
            ms = sc[c]; g = lambda k: np.mean([m[k] for m in ms]); sd = lambda k: np.std([m[k] for m in ms])
            fu = g('fuel_g')
            rows.append(('\\multirow{6}{*}{' + el + '}' if first else '') + f' & {lab[c]} & {g("rms"):.2f}$\\pm${sd("rms"):.2f} & '
                        f'{g("min_x"):.1f} & {g("maxabs"):.1f} & {fu:.2f} & {100*(fu-f0)/f0:+.1f} & {100*g("mean_gen_loss"):.0f} & '
                        f'{g("minE")/1e3:.2f} & {int(round(g("n_floor_viol")))} \\\\')
            first = False
        rows.append('\\midrule')
    out[f'tab_cl_{h}'] = '\n'.join(rows[:-1])
# --- sensitivity
rows = []
def get(kind, evn, h, c, v=None):
    for r in S:
        if r['kind'] == kind and r['event'] == evn and r['h_km'] == h and r['ctrl'] == c and (v is None or r['value'] == v):
            return r['metrics']
for v in (0.0, 0.5, 1.0):
    for c, cl in (('mpc_thrust', 'MPC-T'), ('swmpc', 'SW-MPC')):
        m = get('sigma', 'May2024', 400.0, c, v)
        rows.append(f'$\\bar\\sigma\\times{v:g}$ & {cl} & {m["rms"]:.2f} & {m["min_x"]:.1f} & {m["maxabs"]:.1f} & {m["fuel_g"]:.1f} & {m["minE"]/1e3:.2f} \\\\')
rows.append('\\midrule')
for v in (0.0, 3600.0, 10800.0, 21600.0):
    m = get('latency', 'May2024', 400.0, 'swmpc', v)
    rows.append(f'latency {v/3600:g} h & SW-MPC & {m["rms"]:.2f} & {m["min_x"]:.1f} & {m["maxabs"]:.1f} & {m["fuel_g"]:.1f} & {m["minE"]/1e3:.2f} \\\\')
rows.append('\\midrule')
for c, cl in (('pid', 'PI'), ('mpc_thrust', 'MPC-T'), ('swmpc', 'SW-MPC'), ('swmpc_nosw', 'SW-MPC$^{-}$')):
    m = get('msis21', 'May2024', 400.0, c)
    rows.append(f'truth MSIS 2.1 & {cl} & {m["rms"]:.2f} & {m["min_x"]:.1f} & {m["maxabs"]:.1f} & {m["fuel_g"]:.1f} & {m["minE"]/1e3:.2f} \\\\')
out['tab_sens'] = '\n'.join(rows)
comb = []
for h in ('550', '400'):
    comb.append('\\multicolumn{10}{l}{\\textit{' + h + ' km}} \\\\')
    comb.append(out[f'tab_cl_{h}'])
    if h == '550': comb.append('\\midrule')
out['tab_cl_combined'] = '\n'.join(comb).replace('& +0.0 &', '& 0.0 &').replace('& -0.0 &', '& 0.0 &')
out['tab_valid'] = '\n'.join(l for l in out['tab_valid'].split('\n') if 'Original' not in l).rstrip()
if out['tab_valid'].endswith('\\midrule'): out['tab_valid'] = out['tab_valid'][:-len('\\midrule')].rstrip()
json.dump(out, open(D + 'tables_autogen.json', 'w'), indent=1)
for k, v in out.items(): print('%%', k); print(v)
