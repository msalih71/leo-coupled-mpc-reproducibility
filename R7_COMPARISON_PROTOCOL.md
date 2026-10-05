# R7 common-recovery protocol — 5 October 2026

Written before the R7 campaign. This is a new experiment motivated by the R6
terminal-target failure, not a replacement or reclassification of R6 results.

## Fixed design

- Three archived storms (February 2022, March 2015, May 2024); 400/550 km;
  initial orbital phases 0/180 degrees, fixed node; constant 600 W and
  scheduled +200 W at event hours 30–36.
- Five paired measurement-noise seeds 0–4. These are simulation repetitions,
  not independent storm observations. No gains, weights or margins are tuned.
- Three policies during warm-up/event: PI+FF-LD-P, SW-MPC, and the same MPC
  with solar/geomagnetic features frozen at the initial issue. All retain
  online density-bias estimation and the same known generation/load schedules.
- Full-area sunlit generation 1500 W, cylindrical eclipse, original area/power
  law, strict causal solar protocol and the R5 actual-filter empirical margin.
- Same initial altitude, area, estimator and 1350-Wh energy; 12-h warm-up and
  72-h event. No added terminal-energy constraint is imposed in R7.
- At event hour 72, all policies switch to the same full-area PI+FF recovery:
  reset the PI integrator to zero, keep the estimator, slew to full area with
  the existing area-step limit, and retain thrust, slew and battery physics.
  The no-weather policy continues its frozen features during recovery, so its
  ablation covers the complete mission. No truth density is given to control.
- Recovery lasts at least 24 h. Its endpoint is the FIRST 5-minute control
  boundary at/after event hour 96 for which the preceding step is entirely
  sunlit and the next step contains eclipse (the last complete sunlit step).
  The boundary is computed from orbit geometry alone, before running any
  controller. It is shared by all policies/seeds in that scenario. Do not
  extend recovery in response to results.
- Primary fuel accounting includes warm-up, event and the whole recovery.
  Report event-only fuel/RMS and recovery fuel separately; primary tracking
  comparison remains the 72-h event RMS so recovery cannot dilute errors.
- Primary resource qualification: every complete run remains powered with
  zero sampled altitude/battery-floor violations and zero unmet bus load;
  both end at least 1495 Wh, endpoint energy difference at most 5 Wh and
  endpoint mean-altitude difference at most 1 m. Report all failures and all
  unqualified pairs. Qualification is empirical, not a safety guarantee.
- Report paired fuel differences for every seed and group, event tracking,
  minimum battery energy, endpoint differences, QP attempts/failures, and
  controller computation on the server. Do not pool time samples into
  confidence intervals. Positive fuel differences do not establish dominance
  when tracking or battery reserves deteriorate.
- Weather ablation compares MPC against frozen-feature MPC over the same
  completion window. It does not claim that forecast indices are generally
  unnecessary, or that observed historical indices are real uplink records.
- Total: 3 events × 2 heights × 2 loads × 2 phases × 3 policies × 5 seeds =
  360 runs, 120 PI/MPC pairs and 120 MPC/frozen-feature pairs.

## Scope

This comparison evaluates event control followed by a common recovery duty.
It does not certify indefinite station keeping, recurrent storms, charging
hardware, formal recursive feasibility or execution on a flight processor.
R6 remains reported as the test of an unsuitable fixed terminal target.

## Engineering check before the campaign

Two preliminary May-2024/400-km/phase-0 seed-0 runs checked the recovery
switch and endpoint code. Selecting the first sunlit step can stop near
sunrise before capacity is reached (1421 Wh in both runs). Before the final
campaign, the geometric rule was therefore fixed to the last complete
sunlit step before eclipse. A prematurely launched campaign used the old
endpoint rule; it was stopped, and its outputs are archived under
engineering_first_sunlit, excluded from the final comparison. No controller
gains, event selection, noise seeds, matching tolerance or minimum recovery
length were changed. These engineering runs are not part of the final
360-run comparison.
