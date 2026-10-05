"""Figures for the revised manuscript (IEEE single-column width 3.5 in)."""
import numpy as np, json, pickle, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.linewidth': 0.6,
                     'lines.linewidth': 1.0, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.4,
                     'legend.fontsize': 6.5, 'legend.frameon': False, 'savefig.dpi': 300})
FIG = '../figures/'
CT = ['pid', 'pid_ff', 'rule', 'mpc_thrust', 'swmpc', 'swmpc_nosw']
COL = dict(zip(CT, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']))
LS = dict(zip(CT, ['-', '--', '-.', ':', '-', (0, (5, 1, 1, 1))]))
LAB = dict(pid='PI', pid_ff='PI+FF', rule='Rule-based', mpc_thrust='Thrust-only MPC', swmpc='SW-MPC', swmpc_nosw='SW-MPC (no SW)')
W = 3.5

def save(f, name):
    f.savefig(FIG + name + '.pdf', bbox_inches='tight'); f.savefig(FIG + name + '.png', bbox_inches='tight'); plt.close(f)

# ---- Fig 1: architecture ------------------------------------------------------
def fig1():
    f, ax = plt.subplots(figsize=(W, 2.5)); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06', fc=c, ec='#333333', lw=0.6))
        ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=5.4, linespacing=1.15)
    def arr(x1, y1, x2, y2, t=None, dx=0, dy=0.2):
        ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', lw=0.7, color='#333333'))
        if t: ax.text((x1+x2)/2+dx, (y1+y2)/2+dy, t, ha='center', fontsize=5.4)
    box(0.1, 5.0, 2.6, 1.5, 'Uplinked indices\n3-h $a_p$, $F_{10.7}$', '#d6e6f7')
    box(3.6, 5.0, 3.4, 1.5, 'TIM predictor\nthermal state, bulge\ngeometry, altitude', '#d9f0e6')
    box(7.9, 5.0, 4.0, 1.5, 'Constraint-tightened MPC\nconvex QP: thrust $u$, area $A$\nfloor, slew, battery', '#fbe9c7')
    box(7.9, 2.0, 4.0, 1.6, 'Spacecraft (truth model)\nNRLMSISE-00 density\nslew dynamics, battery', '#eadcf5')
    box(3.6, 2.0, 3.4, 1.6, 'Kalman filter\n$\\hat h$, orbit-periodic\ndensity bias $\\hat\\kappa$', '#f2f2f2')
    box(0.1, 2.0, 2.6, 1.6, 'Orbit geometry\n(onboard\nnavigation)', '#f2f2f2')
    arr(2.7, 5.75, 3.6, 5.75); arr(7.0, 5.75, 7.9, 5.75, r'$\hat\rho_k$')
    arr(9.9, 5.0, 9.9, 3.6, r'$u, A$', dx=0.5, dy=0)
    arr(7.9, 2.8, 7.0, 2.8, r'$h+v$', dy=-0.45)
    arr(5.3, 3.6, 5.3, 5.0, r'$\hat\kappa$', dx=0.3, dy=0)
    arr(7.0, 3.4, 8.2, 5.0); ax.text(7.2, 4.25, r'$\hat h$', fontsize=5.4)
    arr(2.7, 3.2, 3.6, 5.1)
    ax.text(6.0, 0.85, 'Archived margin: offline TIM/reference errors',
            ha='center', fontsize=5.3, style='italic')
    ax.text(6.0, 0.35, 'Operating margin: issued KF forecasts, separate calibration/test events',
            ha='center', fontsize=5.0, style='italic')
    save(f, 'fig1_architecture')

# ---- Fig 2: TIM validation ------------------------------------------------------
def fig2(with_orig=False, name='fig2_density_validation'):
    S = np.load('../data/tim_validation_series.npy', allow_pickle=True).item()
    names = [('Feb2022_G1G2', 'Feb 2022 (G1–G2)'), ('Mar2015_G4', 'Mar 2015 (G4)'), ('May2024_G5', 'May 2024 (G5)')]
    f, axs = plt.subplots(3, 1, figsize=(W, 4.4), sharex=True)
    for ax, (k, t) in zip(axs, names):
        d = S[k]; th = d['t_h']
        sm = lambda y: np.convolve(y, np.ones(96)/96, mode='same')    # 96-min orbit mean
        ax.semilogy(th, d['msis00'], color='#bbbbbb', lw=0.4)
        ax.semilogy(th, sm(d['msis00']), color='#2a78d6', lw=1.1, label='NRLMSISE-00 (orbit mean)')
        ax.semilogy(th, sm(d['msis21']), color='#1baf7a', lw=1.0, ls='--', label='NRLMSIS 2.1 (orbit mean)')
        ax.semilogy(th, sm(d['tim']), color='#eb6834', lw=1.1, label='Revised TIM (orbit mean)')
        if with_orig: ax.semilogy(th, sm(d['orig']), color='#4a3aa7', lw=1.0, ls=':', label='Original TIM (orbit mean)')
        ax.set_ylabel(r'$\rho$ [kg m$^{-3}$]'); ax.set_title(t + ', 550 km', fontsize=7, pad=2)
        a2 = ax.twinx(); a2.fill_between(th, d['ap'], step='post', color='#999999', alpha=0.18, lw=0)
        a2.set_ylim(0, 1300); a2.set_yticks([0, 200, 400]); a2.set_ylabel('$a_p$', fontsize=6); a2.grid(False)
        a2.tick_params(labelsize=5.5)
    axs[0].legend(loc='upper left', ncol=2, fontsize=5.5)
    axs[-1].set_xlabel('Time from window start [h]')
    save(f, name)

# ---- Fig 3: sigma profile --------------------------------------------------------
def fig3():
    s = json.load(open('../data/sigma_profile.json'))
    f, ax = plt.subplots(figsize=(W, 1.9))
    k = np.arange(1, 37)*5/60
    for h, c in (('550km', '#2a78d6'), ('400km', '#eb6834')):
        ax.plot(k, s[h]['sigma_step_p95'], color=c, ls=':', label=f'{h[:3]} km, per step')
        ax.plot(k, s[h]['sigma_cum_p95'], color=c, ls='--', label=f'{h[:3]} km, cumulative')
        ax.plot(k, s[h]['sigma_lam_p95'], color=c, ls='-', lw=1.4, label=f'{h[:3]} km, $\\lambda$-weighted ($\\sigma_k$)')
    ax.axhline(0.2, color='#555555', lw=0.7, ls=(0, (2, 2))); ax.text(2.95, 0.21, '20 % level', ha='right', fontsize=5.8)
    ax.set_xlabel('Lead time [h]'); ax.set_ylabel('95th pct. |rel. error|'); ax.set_ylim(0, 0.75)
    ax.legend(ncol=2, fontsize=5.5, loc='upper center')
    save(f, 'fig3_sigma_profile')

TS = None
# ---- Fig 4: tracking, G5 400 km and G4 550 km ---------------------------------
def fig4():
    f, axs = plt.subplots(2, 1, figsize=(W, 3.6), sharex=True)
    for ax, (ev, h, ttl) in zip(axs, [('Mar2015', 550e3, 'Mar 2015 (G4), 550 km'), ('May2024', 400e3, 'May 2024 (G5), 400 km')]):
        for c in ['pid', 'rule', 'mpc_thrust', 'swmpc', 'swmpc_nosw']:
            o = TS[(ev, h, c)]
            ax.plot(o['t_h'], o['x'], color=COL[c], ls=LS[c], lw=0.9 if c != 'swmpc' else 1.3, label=LAB[c])
        ax.axhline(-15, color='#e34948', lw=0.8); ax.text(71.5, -13.5, 'floor $x_{\\min}$', ha='right', fontsize=5.8)
        ax.set_ylabel('$h-h_{\\rm ref}$ [m]'); ax.set_title(ttl, fontsize=7, pad=2)
    axs[0].set_ylim(-18, 8); axs[1].set_ylim(-18, 45)
    axs[1].legend(ncol=2, loc='upper left', fontsize=5.6)
    axs[1].set_xlabel('Time [h]'); axs[1].set_xlim(0, 72)
    save(f, 'fig4_tracking')

# ---- Fig 5: control actions SW-MPC G5 400 --------------------------------------
def fig5():
    f, axs = plt.subplots(4, 1, figsize=(W, 4.3), sharex=True)
    for c in ['mpc_thrust', 'swmpc', 'rule']:
        o = TS[('May2024', 400e3, c)]
        axs[1].plot(o['t_h'], o['u']*1e3, color=COL[c], ls=LS[c], lw=0.8, label=LAB[c])
        axs[2].plot(o['t_h'], o['A'], color=COL[c], ls=LS[c], lw=0.9)
        axs[3].plot(o['t_h'], o['E']/1e3, color=COL[c], ls=LS[c], lw=0.9)
    o = TS[('May2024', 400e3, 'swmpc')]
    axs[0].fill_between(o['t_h'], o['ap'], step='post', color='#999999', alpha=0.4, lw=0); axs[0].set_ylabel('$a_p$')
    axs[1].axhline(12, color='#555555', lw=0.6, ls='--'); axs[1].set_ylabel('$u$ [mN]')
    axs[2].set_ylabel('$A$ [m$^2$]'); axs[3].set_ylabel('$E$ [kWh]')
    axs[3].axhline(0.3, color='#e34948', lw=0.7); axs[3].text(1, 0.36, '$E_{\\min}$', fontsize=5.8)
    axs[1].legend(ncol=3, fontsize=5.6, loc='upper left'); axs[-1].set_xlabel('Time [h]'); axs[-1].set_xlim(0, 72)
    save(f, 'fig5_control_actions')

# ---- Fig 6: fuel vs generation loss (R_A sweep) + baselines ------------------------
def fig6():
    sens = json.load(open('../data/sensitivity.json')); R = json.load(open('../data/results_metrics.json'))['results']
    f, axs = plt.subplots(1, 2, figsize=(W, 1.8))
    for ax, (ev, h, ttl) in zip(axs, [('Mar2015', 550.0, 'G4, 550 km'), ('May2024', 400.0, 'G5, 400 km')]):
        pts = sorted([(r['value'], r['metrics']) for r in sens if r['kind'] == 'RA' and r['event'] == ev and r['h_km'] == h])
        ax.plot([100*m['mean_gen_loss'] for _, m in pts], [m['fuel_g'] for _, m in pts], '-o', color=COL['swmpc'], ms=3, lw=1.0, label='SW-MPC ($R_A$ sweep)')
        sc = R[f'{ev}_{int(h)}km']
        for c, mk in (('pid', 's'), ('rule', '^'), ('mpc_thrust', 'D')):
            ax.plot(100*np.mean([m['mean_gen_loss'] for m in sc[c]]), np.mean([m['fuel_g'] for m in sc[c]]), mk, color=COL[c], ms=4, label=LAB[c])
        ax.set_title(ttl, fontsize=7, pad=2); ax.set_xlabel('Mean generation loss [%]')
    axs[0].set_ylabel('Propellant, 72 h [g]'); axs[0].legend(fontsize=5.2, loc='upper right')
    f.tight_layout(w_pad=0.6)
    save(f, 'fig6_tradeoff')

if __name__ == '__main__':
    import sys
    if '--architecture-only' in sys.argv:
        fig1()
    else:
        TS = pickle.load(open('../data/results_timeseries.pkl', 'rb'))
        for fn in (fig1, fig2, fig3, fig4, fig5, fig6): fn()
        fig2(True, 'figR1_original_tim')
    print('ok')
