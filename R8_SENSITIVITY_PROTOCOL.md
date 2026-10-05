# R8 one-at-a-time recovery and resource sensitivity — 5 October 2026

Fixed before the campaign; no gains, weights, margins, event selection or matching tolerances will be tuned after seeing outputs. This is numerical sensitivity, not measured hardware uncertainty or an operational envelope.

Design: February 2022, March 2015 and May 2024; 400 and 550 km; initial orbital phase 0 degrees only; constant bus load; paired noise seeds 0,1,2; PI+FF-LD-P and SW-MPC. Every run retains 12 h warm-up and the same R7 geometry-only endpoint at/after event hour 96. The event RMS covers hours 0–72. All fuel includes warm-up, the event and the full remaining duty. No endpoint extension based on outcomes.

Fourteen cases, varying only the named factor from the R7 baseline:
- baseline: recovery starts at hour 72, recovery maximum area 10 m2, sunlit full-area generation 1500 W, bus 600 W, battery capacity 1500 Wh, thruster efficiency 0.5.
- recovery delays: 6, 12, 24 h. Original event controller continues until the delayed switch; then the common PI+FF policy takes over. Endpoint stays fixed; this is a fixed-deadline sensitivity, not recovery-duration matching.
- recovery area ceilings: 8 or 6 m2, only after the common policy switch; existing area-step bound applies in either direction. Area power law and nominal event area remain unchanged. The ceiling is a commanded limit; settling follows existing slew physics.
- full-area generation: 1200 or 1800 W.
- bus load: 480 or 720 W.
- storage capacity: 1200 or 1800 Wh. Initial charge, control floor and physical floor remain 90%,20%,1/15 of capacity, respectively, so initial state of charge is equal. Matching threshold is capacity minus 5 Wh.
- thruster efficiency: 0.4 or 0.6, updating thrust-to-power conversion consistently in plant and controller; specific impulse remains 1800 s.

Generation/load/storage values are ±20% numerical perturbations; efficiency values are absolute ±0.1. They are not fitted, literature-derived hardware ranges, or supplier specifications. No claim of flight-qualified power values will be made.

Matching: both powered, zero sampled altitude/battery floor violations, zero unmet bus energy; both endpoints at least capacity−5 Wh; endpoint energy difference ≤5 Wh and altitude difference ≤1 m. Report every pair including failures. Qualified fuel benefit may be reported only for matched pairs; unmatched raw differences must be labelled as unequal-resource outcomes. Compare event RMS, minimum charge fraction, endpoint energy/altitude, recovery fuel, QP failures and independent energy/fuel ledger closure.

14 cases ×3 events ×2 heights ×1 phase ×3 seeds ×2 controllers =504 runs,252 pairs. This is one-factor sensitivity at one orbital phase, not joint worst-case analysis or independent storm sampling. R7 remains authoritative for its original factorial design. Baseline seed0 must reproduce retained R7 metrics within 1e−9 absolute tolerance (non-timing metrics).
