"""Revised SW-MPC closed-loop simulator (revision R1).

Plant ("truth"):
  * circular 53-deg LEO, mean altitude h_ref (550 or 400 km)
  * density = NRLMSISE-00 along the geodetic track, driven by OBSERVED F10.7/ap,
    with local scale-height correction for altitude deviations
  * co-rotating atmosphere, exact Gauss equation for tangential forces
  * attitude: array feathering angle phi, A = A_min + (A_nom-A_min) cos(phi),
    orbit-average generation P_gen cos(phi); angle interpolated linearly over
    the minimum slew duration inferred from omega_max and alpha_max
  * battery energy with thruster power draw and shunt regulation
  * mean-altitude measurements with white noise sigma_meas
Controller model: TIM predictor (causal ap) x online bias kappa from a four-state KF.
"""
import numpy as np, time, json, os
import scipy.sparse as sp
from orbit_env import orbit_track, msis_density, MU, RE_MEAN
from tim import TIM, TIMPredictor, SW

G0 = 9.80665
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')

# ---------------------------------------------------------------------------
PAR = dict(
    m=260.0, Cd=2.2, A_nom=10.0, A_min=2.0, u_max=12e-3, Isp=1800.0, eta_T=0.5,
    P_gen=900.0, P_load=600.0, E_min=300.0, E_max=1500.0, E0=1350.0, E_phys_min=100.0,
    omega_max=np.deg2rad(0.5), alpha_max=np.deg2rad(0.02), T_slew=90.0,
    dt=300.0, sub=5, N=36, sigma_meas=2.0, x_min=-15.0,
    # cost weights (u in mN, A in m^2, x in m, E in Wh)
    Q=1.0, w_f=10.0, R_u=1e-2, R_A=1e-4, R_dA=0.1, w_E=2e-3, w_s=1e3, w_s2=10.0,
)
PAR['kp_power'] = G0*PAR['Isp']/(2*PAR['eta_T'])            # W per N

def slew_angle_limit(p):
    """Largest rest-to-rest angle achievable within T_slew with omega/alpha limits."""
    w, a, T = p['omega_max'], p['alpha_max'], p['T_slew']
    if w/a*2 >= T:          # triangular profile
        return a*(T/2)**2
    return w*(T - w/a)
PAR['dphi_max'] = slew_angle_limit(PAR)
PAR['dA_max'] = (PAR['A_nom']-PAR['A_min'])*(1-np.cos(PAR['dphi_max']))   # convex inner bound

def slew_time(dphi, p):
    w, a = p['omega_max'], p['alpha_max']; dphi = abs(dphi)
    if dphi <= w*w/a:
        return 2*np.sqrt(dphi/a)
    return dphi/w + w/a

def area_of_phi(phi, p): return p['A_min'] + (p['A_nom']-p['A_min'])*np.cos(phi)
def phi_of_area(A, p):   return np.arccos(np.clip((A-p['A_min'])/(p['A_nom']-p['A_min']), 0, 1))
def eta_of_area(A, p):   return (A-p['A_min'])/(p['A_nom']-p['A_min'])

# ---------------------------------------------------------------------------
class Scenario:
    def __init__(self, name, t_start, h_ref, dur_h=84.0, spinup_h=12.0, truth_version=0, orbit_phase_deg=0.0):
        self.name, self.h_ref = name, h_ref
        self.orbit_phase_deg = float(orbit_phase_deg)
        self.t0 = np.datetime64(t_start) - np.timedelta64(int(spinup_h*3600), 's')
        self.spinup_h, self.dur_h = spinup_h, dur_h
        self.tr = orbit_track(self.t0, dur_h*3600 + 6*3600, 60.0, h_mean=h_ref, u0_deg=self.orbit_phase_deg)   # +6 h horizon pad
        self.rho = msis_density(self.tr, truth_version)
        rho1 = msis_density(self.tr, truth_version, dalt_km=1.0)
        self.H = 1000.0/np.log(self.rho/rho1)                                   # local scale height [m]
        self.ap = SW.ap_at(self.tr['dt64'])
        self.tim = TIM.load(h_ref/1e3)
        self.a_ref = RE_MEAN + h_ref
        self.v_ref = np.sqrt(MU/self.a_ref)
        self.c_ref = 2*self.a_ref**2*self.v_ref/(MU*PAR['m'])                    # (m/s)/N

# ---------------------------------------------------------------------------
class Estimator:
    """Kalman filter on z = [x, k0, kc, ks]: orbit-periodic multiplicative density
    correction kappa(th) = k0 + kc cos th + ks sin th (th = orbital phase), i.e. a
    disturbance model for offset-free MPC [Pannocchia & Rawlings 2003].
        x_{k+1} = x_k + b u_k - g_k A_k (phi_k . kappa),   y_k = x_k + v_k
    phi_k = step average of [1, cos th, sin th]."""
    def __init__(self, p, q_x=0.05**2, q_k=0.03**2, q_h=0.02**2):
        self.p = p; self.z = np.array([0.0, 1.0, 0.0, 0.0])
        self.P = np.diag([4.0, 0.3**2, 0.2**2, 0.2**2])
        self.Qn = np.diag([q_x, q_k, q_h, q_h]); self.R = p['sigma_meas']**2
    def predict(self, b, gA, u_mN, phi):
        F = np.eye(4); F[0, 1:] = -gA*phi
        self.z = F @ self.z + np.array([b*u_mN, 0, 0, 0])
        self.P = F @ self.P @ F.T + self.Qn
    def update(self, y):
        H = np.array([1.0, 0, 0, 0]); S = H @ self.P @ H + self.R
        K = self.P @ H / S
        self.z = self.z + K*(y - self.z[0]); self.P = (np.eye(4) - np.outer(K, H)) @ self.P
    def mult(self, phi):
        """kappa multiplier and its 1-sigma for regressor rows phi (n x 3)."""
        m = phi @ self.z[1:]
        sd = np.sqrt(np.einsum('ij,jk,ik->i', phi, self.P[1:, 1:], phi))
        return np.maximum(m, 0.1), sd
    @property
    def x(self): return self.z[0]

# ---------------------------------------------------------------------------
class QPMPC:
    """Condensed, move-blocked convex QP solved by the Goldfarb-Idnani dual
    active-set method (quadprog).  Decision vector
        z = [U (nb), A (nb), P_sh (nb), s_x, s_E, s_T],  nb = N / L
    Per-step inputs are piecewise constant over blocks of L steps.
    Empirical constraint margins use the notional one-sided law
    u = u_bar + K max(0, x_bar - x), K = (1 - lam)/b. This auxiliary law
    is NOT applied in the closed-loop simulator; the margin is heuristic:
        delta_k = sigma_bar sum_{j<k} lam^(k-1-j) g_j A_j  (+ eps0),
    sigma_bar = max over lead time of the empirical 95th percentile of the
    lam-weighted relative density error (plus KF bias uncertainty),
    which tightens the altitude floor, the thrust bound and the battery floor.
    Units: u in mN, A in m^2, x = h - h_ref in m, E in Wh."""
    def __init__(self, p, coupled=True, L=3, lam=0.8):
        import quadprog
        self.qp = quadprog.solve_qp
        self.p, self.coupled, self.L, self.lam = p, coupled, L, lam
        N = p['N']; self.N = N; nb = N // L; self.nb = nb
        self.Bm = np.kron(np.eye(nb), np.ones((L, 1)))                  # N x nb
        self.Lt = np.tril(np.ones((N, N)))                               # x_{k+1} uses inputs 0..k
        self.Ll = np.array([[lam**(k-j) if j <= k else 0.0 for j in range(N)] for k in range(N)])
        self.nz = 3*nb + 3
        self.iU, self.iA, self.iP = np.arange(nb), np.arange(nb, 2*nb), np.arange(2*nb, 3*nb)
        self.isx, self.isE, self.isT = 3*nb, 3*nb+1, 3*nb+2

    def solve(self, x0, E0, A_prev, b, g, sig, eps0=0.0, power=None, load=None, area_floor=None, energy_goal=None):
        p, N, nb, Bm = self.p, self.N, self.nb, self.Bm
        lam = self.lam; K = (1 - lam)/b                                   # mN per m
        dth = p['dt']/3600.0; Rg = p['A_nom'] - p['A_min']
        generation = np.full(N,p['P_gen']) if power is None else np.asarray(power)
        bus = np.full(N,p['P_load']) if load is None else np.asarray(load)
        aP = dth*generation/Rg; bP = dth*p['kp_power']*1e-3
        nz = self.nz
        SU = np.zeros((N, nz)); SU[:, self.iU] = Bm                       # u_k
        SA = np.zeros((N, nz)); SA[:, self.iA] = Bm                       # A_k
        SP = np.zeros((N, nz)); SP[:, self.iP] = Bm
        # x_{k+1} = x0 + Lt (b u - g A)   (rows k=0..N-1 -> x_1..x_N)
        X = self.Lt @ (b*SU - g[:, None]*SA); x_c = np.full(N, x0)
        # tube: delta_{k+1} = sum_{j<=k} lam^{k-j} sig_j g_j A_j  (+eps0 carried with lam^k decay -> bound by eps0)
        Dl = sig[:, None]*(self.Ll @ (g[:, None]*SA)); d_c = np.full(N, eps0)
        Dprev = np.vstack([np.zeros((1, nz)), Dl[:-1]]); dprev_c = np.concatenate([[eps0], d_c[:-1]])  # delta at step k
        # energy with worst-case thrust u + K delta_k
        dE = aP[:,None]*SA - bP*(SU + K*Dprev) - dth*SP
        Ec = self.Lt @ dE; e_c = E0 + self.Lt @ (-aP*p['A_min'] - dth*bus - bP*K*dprev_c)
        # ------------------------------------------------------------ cost
        H = np.zeros((nz, nz)); f = np.zeros(nz)
        H += 2*p['Q']*X.T @ X; f += 2*p['Q']*X.T @ x_c
        H += 2*p['R_u']*SU.T @ SU; f += p['w_f']*p['dt']/(p['Isp']*G0)*SU.sum(0)
        H += 2*p['R_A']*SA.T @ SA; f += -2*p['R_A']*p['A_nom']*SA.sum(0)
        dA = SA - np.vstack([np.zeros((1, nz)), SA[:-1]]); dA_c = np.zeros(N); dA_c[0] = -A_prev
        H += 2*p['R_dA']*dA.T @ dA; f += 2*p['R_dA']*dA.T @ dA_c
        f += -p['w_E']*Ec.sum(0)
        H += 1e-6*SP.T @ SP
        for j in (self.isx, self.isE, self.isT):
            H[j, j] += 2*p['w_s2']; f[j] += p['w_s']
        H += 1e-9*np.eye(nz)
        # ------------------------------------------------------------ constraints  C z >= d
        C, d = [], []
        def ge(M, rhs): C.append(np.atleast_2d(M)); d.append(np.atleast_1d(rhs))
        ES = np.zeros(nz)
        sx = ES.copy(); sx[self.isx] = 1; sE = ES.copy(); sE[self.isE] = 1; sT = ES.copy(); sT[self.isT] = 1
        eq_rows = 0
        if not self.coupled:           # equality rows first (meq)
            ge(SA[::self.L], np.full(nb, p['A_nom'])); eq_rows = nb
        ge(X - Dl + sx, p['x_min'] - x_c + d_c)                          # robust floor
        ge(-(SU + K*Dprev), -(p['u_max']*1e3 - K*dprev_c))               # tightened thrust bound
        ge(SU[::self.L], np.zeros(nb))                                   # u >= 0
        if self.coupled:
            ge(SA[::self.L], np.full(nb,p['A_min'])) if area_floor is None else ge(SA, np.asarray(area_floor)); ge(-SA[::self.L], np.full(nb, -p['A_nom']))
        ge(SP[::self.L], np.zeros(nb))
        ge(Ec + sE, p['E_min'] - e_c)                                    # battery floor (worst case)
        Enom = self.Lt @ (aP[:,None]*SA - bP*SU - dth*SP); enom_c = E0 + self.Lt @ (-aP*p['A_min'] - dth*bus)
        if energy_goal is not None:
            goal_index,goal_Wh=energy_goal
            ge(Enom[goal_index],goal_Wh-enom_c[goal_index])
        ge(-Enom, -(p['E_max'] - enom_c))                                # battery ceiling (nominal)
        blk = dA[::self.L]; blk_c = dA_c[::self.L]
        if self.coupled:
            ge(blk, -p['dA_max'] - blk_c); ge(-blk, -p['dA_max'] + blk_c)   # slew
        ge(b*SU[-1] - (1 + sig[-1])*g[-1]*SA[-1] + b*sT, 0.0)           # terminal slack measured in mN (b: m/mN)
        ge(-(SU[-1] + K*Dl[-1]) + sT, -(p['u_max']*1e3 - K*d_c[-1]))     # terminal thrust margin with delta_N
        ge(aP[-1]*SA[-1] - bP*(SU[-1] + K*Dl[-1]) + bP*sT, aP[-1]*p['A_min'] + dth*bus[-1] + bP*K*d_c[-1])  # energy-neutral
        for j in (self.isx, self.isE, self.isT):
            e = ES.copy(); e[j] = 1; ge(e, 0.0)
        Cm = np.vstack(C); dv = np.concatenate(d)
        t = time.perf_counter()
        try:
            z, *_ , it, lag, act = self.qp(H, -f, Cm.T, dv, eq_rows)
            ok = True
        except ValueError as err:
            z = np.zeros(nz); ok = False; it = np.array([0, 0]); self.last_err = str(err)
        wall = time.perf_counter() - t
        if ok:
            residual = Cm @ z - dv
            violation = max(float(np.max(np.abs(residual[:eq_rows]))) if eq_rows else 0.0,
                            float(np.max(np.maximum(-residual[eq_rows:], 0))))
            stationarity = H @ z + f - Cm.T @ lag
            scale = max(1.0, float(np.linalg.norm(H @ z, np.inf)), float(np.linalg.norm(f, np.inf)),
                        float(np.linalg.norm(Cm.T @ lag, np.inf)))
            kkt_scaled = float(np.linalg.norm(stationarity, np.inf))/scale
            complementarity = float(np.max(np.abs(lag[eq_rows:]*residual[eq_rows:])))
        else:
            violation = kkt_scaled = complementarity = float('nan')
        u = SU @ z; A = SA @ z
        return dict(z=z, ok=ok, iters=int(np.sum(it)), solve_time=wall, u=u[0]*1e-3, A=A[0],
                    slack=float(z[self.isx]), slackE=float(z[self.isE]), slackT=float(z[self.isT]),
                    x_pred=x_c + X @ z, delta=d_c + Dl @ z, u_plan=u, A_plan=A, E_plan=e_c + Ec @ z,
                    n_var=nz, n_con=Cm.shape[0], qp_violation=violation,
                    kkt_scaled=kkt_scaled, complementarity=complementarity)

# ---------------------------------------------------------------------------
def run(scn, controller, seed=0, par=None, sigma_prof=None, latency_s=3600.0, pid_wn=None,
        use_sw=True, record=True, forecast_stride=None, strict_solar=False,
        generation_W=None, load_W=None, cd_truth=None, measurement_gap=None, index_outage=None, area_floor=None, solver_gap=None, terminal_energy=None, recovery_start_h=None, recovery_area_max=None):
    p = dict(PAR); p.update(par or {})
    rng = np.random.default_rng(seed)
    N, sub, dt = p['N'], p['sub'], p['dt']
    nsteps = int(scn.dur_h*3600/dt)
    tr = scn.tr
    pred = TIMPredictor(scn.tim, tr, scn.h_ref, lat_s=latency_s, strict_solar=strict_solar)
    if not use_sw:            # ablation: hold ap and the causal F10.7 features at t0
        pred.freeze_t = tr['dt64'][0]
        pred.X[:, 1:4] = pred.X[0, 1:4]
        pred.logrho_geo = pred.X @ scn.tim.w
    generation = np.full(len(tr['t']),p['P_gen']) if generation_W is None else np.asarray(generation_W)
    bus = np.full(len(tr['t']),p['P_load']) if load_W is None else np.asarray(load_W)
    if len(generation)!=len(tr['t']) or len(bus)!=len(tr['t']): raise ValueError('power/load track length')
    floor = np.full(nsteps,p['A_min']) if area_floor is None else np.asarray(area_floor)
    floor_padded = np.pad(floor,(0,N),mode='edge')
    forecast_rows=[]
    corot_mean = tr['corot'].mean()
    kg = scn.c_ref*dt*0.5*p['Cd']*scn.v_ref**2*corot_mean        # g_k = kg * rho * kappa  [m per m^2]
    b = scn.c_ref*dt*1e-3                                         # m per mN-step
    # constant, conservative level sigma_bar = max_k sigma_k (empirical margin; no feasibility proposition)
    sig = np.full(N, float(np.max(np.asarray(sigma_prof if sigma_prof is not None else [0.3])[:N])))
    est = Estimator(p)
    T_orb = 2*np.pi*np.sqrt(scn.a_ref**3/MU)
    mpc = QPMPC(p, coupled=(controller in ('swmpc', 'swmpc_nosw'))) if controller in ('swmpc', 'swmpc_nosw', 'mpc_thrust') else None
    # PID (PI on estimated altitude); natural frequency tuned on calibration storm
    wn = pid_wn if pid_wn is not None else 1/3600.0
    Kp = 2*0.9*wn/scn.c_ref; Ki = wn**2/scn.c_ref
    integ = None
    # state
    a = scn.a_ref; E = p['E0']; m = p['m']; phi = 0.0; A_cur = p['A_nom']
    out = {k: np.zeros(nsteps) for k in ['t_h', 'x', 'x_hat', 'u', 'A', 'E', 'kappa', 'kappa_sd', 'rho_true', 'rho_pred',
                                          'fuel', 'drag_imp', 'thr_imp', 'slack', 'slack_x', 'slack_E', 'slack_T', 'solve_ms', 'iters', 'ap', 'shunt', 'sigma_eff', 'qp_attempt', 'qp_ok', 'qp_violation', 'kkt_scaled', 'complementarity', 'unserved_Wh', 'measurement_used', 'area_floor', 'forced_fallback', 'generated_Wh', 'load_Wh', 'propulsion_Wh', 'x_after']}
    fuel = 0.0; dimp = 0.0; timp = 0.0
    solve_ms = []; loop_ms = []
    for k in range(nsteps):
        i = k*sub
        t_loop = time.perf_counter()
        # ---- measurement & estimation ---------------------------------------
        x_true = a - scn.a_ref
        y = x_true + rng.normal(0, p['sigma_meas'])
        if k > 0:
            est.predict(b, g0_prev*A_prev_cmd, u_prev*1e3, reg_prev)
        t_event_h=k*dt/3600-scn.spinup_h
        measured = measurement_gap is None or not (measurement_gap[0]<=t_event_h<measurement_gap[1])
        if measured: est.update(y)
        if index_outage is not None:
            if index_outage[0]<=t_event_h<index_outage[1]:
                if not hasattr(pred,'freeze_t'): pred.freeze_t=tr['dt64'][i]
            elif use_sw and hasattr(pred,'freeze_t'):
                del pred.freeze_t
        x_hat = est.x
        th = np.deg2rad(scn.orbit_phase_deg) + 2*np.pi*tr['t'][i:i+N*sub]/T_orb
        reg = np.column_stack([np.ones(N), np.cos(th).reshape(N, sub).mean(1), np.sin(th).reshape(N, sub).mean(1)])
        mult, msd = est.mult(reg); kap = mult[0]
        # ---- prediction ---------------------------------------------------
        rho_hat = pred.predict_steps(i, N, sub)
        g = kg*rho_hat*mult
        sig_eff = sig + 2*float(np.max(msd/mult))
        if forecast_stride and k%forecast_stride==0 and t_event_h>=0 and k+N<=nsteps:
            truth_future=scn.rho[i:i+N*sub].reshape(N,sub).mean(1)
            forecast_rows.append((t_event_h, rho_hat*mult, truth_future, sig_eff.copy()))
        goal=None
        if terminal_energy is not None:
            deadline_h,target_Wh=terminal_energy
            j_goal=int(round((deadline_h+scn.spinup_h)*3600/dt))-k-1
            if 0<=j_goal<N:goal=(j_goal,target_Wh)
        floor_plan=floor_padded[k:k+N]
        power_plan=generation[i:i+N*sub].reshape(N,sub).mean(1)
        load_plan=bus[i:i+N*sub].reshape(N,sub).mean(1)
        # ---- control ------------------------------------------------------
        # R7: a predefined common full-area PI+FF recovery after the event.
        # Keep the estimator state; reset only the PI integral at the switch.
        recovering = recovery_start_h is not None and t_event_h >= recovery_start_h
        if recovering and (k == 0 or (k-1)*dt/3600-scn.spinup_h < recovery_start_h):
            integ = 0.0
        active_controller = 'pid_ff' if recovering else controller
        info = None
        if active_controller in ('pid', 'pid_ff', 'rule', 'pid_lowdrag', 'pid_lowdrag_preview'):
            e = -x_hat
            if integ is None:   # PI: start at equilibrium drag; FF variants: drag is supplied by the feedforward
                integ = 0.0 if active_controller in ('pid_ff', 'rule', 'pid_lowdrag', 'pid_lowdrag_preview') else (kg*rho_hat[0]*p['A_nom']/b*1e-3)/Ki/dt
            ff = 0.0
            recovery_target = p['A_nom'] if recovery_area_max is None else float(recovery_area_max)
            if not p['A_min'] <= recovery_target <= p['A_nom']: raise ValueError('Invalid recovery area ceiling')
            A_cmd = (float(np.clip(recovery_target, A_cur-p['dA_max'], A_cur+p['dA_max']))
                     if recovering else p['A_nom'])
            if active_controller in ('pid_ff', 'rule', 'pid_lowdrag', 'pid_lowdrag_preview'):
                if active_controller == 'rule':
                    # operator-style rule: go to energy-neutral area when predicted
                    # 1-h mean drag exceeds 1.5x the pre-storm (first 12 h) mean
                    if k == 0: out['_ref_rho'] = np.mean(rho_hat[:12]*mult[:12])
                    ratio = np.mean(rho_hat[:12]*mult[:12])/out['_ref_rho']
                    A_E = p['A_min'] + (p['A_nom']-p['A_min'])*(p['P_load'] + p['kp_power']*p['u_max']*0.5)/p['P_gen']
                    A_target = A_E if (ratio > 1.5 and E > p['E_min'] + 100) else p['A_nom']
                    A_cmd = float(np.clip(A_target, A_cur - p['dA_max'], A_cur + p['dA_max']))
                if active_controller in ('pid_lowdrag','pid_lowdrag_preview'):
                    # Algebraic power-balanced area schedule; same causal density/KF,
                    # no optimizer, storm threshold, or future truth information.
                    # Recover E0 on a fixed 3-h time scale (Wh/h = W).
                    recover_W = (p['E0'] - E)/3.0
                    correction_N = max(0.0, Kp*e + Ki*integ*dt)
                    scheduled_power = power_plan.mean() if active_controller=='pid_lowdrag_preview' else power_plan[0]
                    scheduled_load = load_plan.mean() if active_controller=='pid_lowdrag_preview' else load_plan[0]
                    slope_W_m2 = scheduled_power/(p['A_nom']-p['A_min'])
                    drag_N_m2 = kg*rho_hat[0]*kap/b*1e-3
                    denom = slope_W_m2 - p['kp_power']*drag_N_m2
                    A_target = ((scheduled_load + recover_W + p['kp_power']*correction_N
                                 + slope_W_m2*p['A_min'])/denom
                                if denom > 0 else p['A_nom'])
                    A_target = np.clip(A_target, floor_plan[0], p['A_nom'])
                    A_cmd = float(np.clip(A_target, A_cur-p['dA_max'], A_cur+p['dA_max']))
                ff = kg*rho_hat[0]*kap*A_cmd/b*1e-3            # N, feedforward drag
            v_pi = Kp*e + Ki*integ*dt
            u_raw = v_pi + ff
            u_cmd = float(np.clip(u_raw, 0, p['u_max']))
            if u_cmd == u_raw or (u_raw > p['u_max'] and e < 0) or (u_raw < 0 and e > 0):
                integ += e                                   # conditional integration (anti-windup)
        else:
            if solver_gap is not None and solver_gap[0]<=t_event_h<solver_gap[1]:
                info = dict(ok=False, solve_time=0., iters=0, slack=0., slackE=0., slackT=0.,
                            qp_violation=np.nan,kkt_scaled=np.nan,complementarity=np.nan)
                out['forced_fallback'][k]=1
            else:
                info = mpc.solve(x_hat, E, A_cur, b, g, sig_eff, eps0=2*np.sqrt(est.P[0, 0]),
                                 power=power_plan, load=load_plan,
                                 area_floor=floor_plan if area_floor is not None else None, energy_goal=goal)
            if info['ok']:
                u_cmd = float(np.clip(info['u'], 0, p['u_max'])); A_cmd = float(np.clip(info['A'], p['A_min'], p['A_nom']))
            else:   # fallback: hold area, cancel predicted drag
                A_cmd = A_cur; u_cmd = float(np.clip(g[0]*A_cur/b*1e-3, 0, p['u_max']))
            solve_ms.append(info['solve_time']*1e3)
        loop_ms.append((time.perf_counter() - t_loop)*1e3)
        # ---- plant propagation over one control step (60-s substeps) -------
        phi_target = phi_of_area(A_cmd, p); dphi = phi_target - phi
        T_sl = slew_time(dphi, p) if abs(dphi) > 1e-9 else 0.0
        shunt = 0.0; A_used = 0.0; unserved = 0.0
        generated_Wh = load_Wh = propulsion_Wh = 0.0
        for j in range(sub):
            tj = (j + 0.5)*60.0
            frac = min(1.0, tj/T_sl) if T_sl > 0 else 1.0
            ph = phi + dphi*frac; A_eff = area_of_phi(ph, p)
            v = np.sqrt(MU/a)
            rho = scn.rho[i+j]*np.exp(-(a - scn.a_ref)/scn.H[i+j])
            D = 0.5*rho*(p['Cd'] if cd_truth is None else cd_truth)*A_eff*v**2*tr['corot'][i+j]
            u_app = u_cmd if E > p['E_phys_min'] else 0.0      # load shedding protects the battery
            dadt = 2*a**2*v/MU*(u_app - D)/m
            a += dadt*60.0
            dm = u_app*60.0/(p['Isp']*G0); m -= dm; fuel += dm*1e3
            dimp += D*60.0; timp += u_app*60.0
            generated_Wh += generation[i+j]*np.cos(ph)/60.0
            load_Wh += bus[i+j]/60.0
            propulsion_Wh += p['kp_power']*u_app/60.0
            Pnet = generation[i+j]*np.cos(ph) - bus[i+j] - p['kp_power']*u_app
            E_new = E + Pnet*60.0/3600.0
            if E_new > p['E_max']: shunt += (E_new - p['E_max']); E_new = p['E_max']
            # Do not represent a battery with negative stored energy. A zero
            # battery and unserved bus load mark operation outside this model's
            # valid powered-control regime; continuation is diagnostic only.
            if E_new < 0:
                unserved += -E_new
                E_new = 0.0
            E = E_new; A_used += A_eff/sub
        phi = phi_target; A_cur = A_cmd
        u_prev, A_prev_cmd, g0_prev, reg_prev = u_cmd, A_used, kg*rho_hat[0], reg[0]
        if record:
            out['x_after'][k] = a - scn.a_ref
            out['t_h'][k] = k*dt/3600 - scn.spinup_h; out['x'][k] = x_true; out['x_hat'][k] = x_hat
            out['u'][k] = u_cmd; out['A'][k] = A_used; out['E'][k] = E; out['kappa'][k] = kap
            out['kappa_sd'][k] = msd[0]; out['rho_true'][k] = scn.rho[i:i+sub].mean()
            out['rho_pred'][k] = rho_hat[0]*kap; out['fuel'][k] = fuel; out['drag_imp'][k] = dimp
            out['thr_imp'][k] = timp; out['ap'][k] = scn.ap[i]; out['shunt'][k] = shunt
            out['sigma_eff'][k] = sig_eff[-1]
            out['unserved_Wh'][k] = unserved
            out['generated_Wh'][k] = generated_Wh
            out['load_Wh'][k] = load_Wh
            out['propulsion_Wh'][k] = propulsion_Wh
            out['measurement_used'][k] = measured
            out['area_floor'][k] = floor_plan[0]
            if info is not None:
                out['slack_x'][k] = info['slack']; out['slack_E'][k] = info['slackE']; out['slack_T'][k] = info['slackT']
                out['slack'][k] = max(info['slack'], info['slackE'], info['slackT']); out['iters'][k] = info['iters']
                out['solve_ms'][k] = info['solve_time']*1e3
                out['qp_attempt'][k] = 1
                out['qp_ok'][k] = int(info['ok'])
                for diagnostic in ('qp_violation', 'kkt_scaled', 'complementarity'):
                    out[diagnostic][k] = info[diagnostic]
    if forecast_rows:
        out['forecast_trace'] = dict(t_h=np.array([r[0] for r in forecast_rows]),
            prediction=np.stack([r[1] for r in forecast_rows]), truth=np.stack([r[2] for r in forecast_rows]),
            margin=np.stack([r[3] for r in forecast_rows]))
    out['parameters'] = p
    out['solve_ms_all'] = np.array(solve_ms); out['loop_ms_all'] = np.array(loop_ms)
    out.pop('_ref_rho', None)
    return out

def metrics(o, t_from=0.0, storm=None, x_min=PAR['x_min']):
    p = o.get('parameters', PAR)
    msk = o['t_h'] >= t_from
    first = int(np.flatnonzero(msk)[0])
    x = o['x'][msk]
    # Accumulators at timestamp k already include the k-th 5-minute step.
    # Include that step whenever x[k] belongs to the measured window.
    f0 = float(o['fuel'][first-1]) if first else 0.0
    drag0 = float(o['drag_imp'][first-1]) if first else 0.0
    thr0 = float(o['thr_imp'][first-1]) if first else 0.0
    d = dict(rms=float(np.sqrt(np.mean(x**2))), maxabs=float(np.abs(x).max()), min_x=float(x.min()),
             fuel_g=float(o['fuel'][-1] - f0), minA=float(o['A'][msk].min()), minE=float(o['E'][msk].min()),
             n_floor_viol=int((x < x_min).sum()),
             drag_imp=float(o['drag_imp'][-1] - drag0),
             thr_imp=float(o['thr_imp'][-1] - thr0),
             E_start_Wh=float(o['E'][first-1]) if first else float(p['E0']),
             E_end_Wh=float(o['E'][-1]),
             x_end=float(o.get('x_after', o['x'])[-1]), mean_gen_loss=float(1 - np.mean(eta_of_area(o['A'][msk], p))),
             max_slack=float(o['slack'][msk].max()), max_slack_x_m=float(o['slack_x'][msk].max()),
             max_slack_E_Wh=float(o['slack_E'][msk].max()), max_slack_T_mN=float(o['slack_T'][msk].max()), u_sat_frac=float(np.mean(o['u'][msk] >= p['u_max']*0.999)))
    if 'generated_Wh' in o:
        for name in ('generated_Wh', 'load_Wh', 'propulsion_Wh', 'shunt', 'unserved_Wh'):
            d['total_'+name] = float(o[name][msk].sum())
        d['energy_closure_error_Wh'] = (d['E_end_Wh']-d['E_start_Wh'] -
            (d['total_generated_Wh']-d['total_load_Wh']-d['total_propulsion_Wh']-
             d['total_shunt']+d['total_unserved_Wh']))
        d['fuel_power_closure_error_Wh'] = d['total_propulsion_Wh'] - p['kp_power']*p['Isp']*G0*d['fuel_g']/3.6e6
    if 'area_floor' in o:
        d['n_area_floor_viol'] = int((o['A'][msk]<o['area_floor'][msk]-1e-6).sum())
        d['forced_fallback_steps'] = int(o['forced_fallback'][msk].sum())
        d['missing_measurement_steps'] = int((o['measurement_used'][msk]==0).sum())
    if 'unserved_Wh' in o:
        d['unserved_bus_Wh'] = float(o['unserved_Wh'][msk].sum())
        d['n_energy_floor_viol'] = int((o['E'][msk] < p['E_min']).sum())
        d['n_physical_battery_floor'] = int((o['E'][msk] <= p['E_phys_min']).sum())
        d['powered_regime_valid'] = bool(not (o['E'] <= p['E_phys_min']).any())
    if 'qp_attempt' in o:
        attempts = o['qp_attempt'].astype(bool)
        successful = attempts & o['qp_ok'].astype(bool)
        d['qp_attempts_all'] = int(attempts.sum())
        d['qp_failures_all'] = int((attempts & ~successful).sum())
        for diagnostic in ('qp_violation', 'kkt_scaled', 'complementarity'):
            d['max_'+diagnostic] = float(o[diagnostic][successful].max()) if successful.any() else None
    if storm is not None:
        s = (o['t_h'] >= storm[0]) & (o['t_h'] <= storm[1])
        d['rms_storm'] = float(np.sqrt(np.mean(o['x'][s]**2))); d['maxabs_storm'] = float(np.abs(o['x'][s]).max())
    if len(o['solve_ms_all']):
        sm = o['solve_ms_all']
        d.update(solve_med_ms=float(np.median(sm)), solve_p99_ms=float(np.percentile(sm, 99)), solve_max_ms=float(sm.max()))
    lm = o['loop_ms_all']
    d.update(loop_med_ms=float(np.median(lm)), loop_p99_ms=float(np.percentile(lm, 99)), loop_max_ms=float(lm.max()))
    return d
