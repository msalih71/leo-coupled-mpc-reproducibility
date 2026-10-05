# R6 predefined comparison protocol — 2 October 2026

Written before the new campaign. No controller tuning or event selection from the new results.

- Three original events: February 2022, March 2015, May 2024; two altitudes: 400 and 550 km.
- Initial argument-of-latitude phases: 0 and 180 degrees; ascending-node longitude fixed at zero. This is phase sensitivity, not a full orbit-orientation survey.
- Three paired measurement-noise seeds: 0, 1, 2. These are simulation noise repetitions, not independent observed storms.
- Two known load schedules: constant 600 W and 600 W plus 200 W during event hours 30–36.
- Cylindrical eclipse; sunlit full-area power 1500 W; existing cosine area/power relation.
- Controllers: existing PI+FF low-drag preview schedule and existing constraint-tightened MPC. Both have the same known eclipse/load schedules and causal density inputs. No gains or weights are retuned.
- MPC terminal lower bound: 1450 Wh at event hour 72, prescribed for every case. The PI energy-recovery rule is retained unchanged. Thus the two controllers do not have identical terminal constraints; actual endpoint matching must be checked, not presumed.
- Primary accounting includes all 84 hours, including the 12-hour filter warm-up, from the same initial altitude, battery energy (1350 Wh), attitude and estimator state. Secondary 72-hour event-window metrics record their potentially unequal start energies.
- Resource-matched subset requires both complete runs powered, zero sampled altitude/battery-floor violations, both endpoint energies at least 1445 Wh and endpoint difference at most 5 Wh. This is a reporting criterion, not proof of feasibility between samples. All cases remain in output regardless of qualification.
- Report fuel, tracking RMS, start/end/minimum energy, unserved bus load, sampled violations, QP failures, and endpoint mismatch. Do not describe unmatched or unpowered cases as a matched-resource fuel benefit.
- Fuel saving = 100*(baseline fuel - MPC fuel)/baseline fuel. Report every event/altitude/load/phase group and paired-seed range; avoid pooled confidence statements based on correlated time samples.
- Planned total: 3 events × 2 altitudes × 2 loads × 2 phases × 2 controllers × 3 seeds = 144 runs.
