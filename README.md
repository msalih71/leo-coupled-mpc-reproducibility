# Coupled thrust, drag-area and energy MPC for LEO station keeping

Reproducible **R8** code and derived data for **Model Predictive Control of Coupled Thrust and Drag Area for LEO Station-Keeping Under Energy Constraints**, by Mahgoub A. Salih and Mohamed Y. Shirgawi, Department of Physics, College of Science, Qassim University.

This is a numerical study. Empirical margins do not establish robust stability, recursive feasibility, safety or onboard readiness. Negative outcomes are retained.

## Public repository scope

This repository contains code, derived numerical data, numerical protocols and verification only. Full manuscript text, manuscript drafts, document assembly scripts, referee reports and editorial correspondence are excluded. Do not add them to this repository.

## Install

Python **3.12** is the tested environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py verify
```

Numerical verification uses bundled derived inputs. External JB2008 analysis requires `gfortran` and the provider files described in `external_validation/README.md`.

## Reproduce

Commands below run from the repository root; the runner also works from other working directories.

| Command | Scope |
|---|---|
| `python reproduce.py verify` | Validate retained claims, campaign design, resource matching, energy/fuel accounting and selected regression runs; does not rerun every campaign. |
| `python reproduce.py r6` | Recompute 144 terminal-target runs and associated negative outcomes. |
| `python reproduce.py r7` | Recompute 360 common-recovery runs and weather ablation. |
| `python reproduce.py r8` | Recompute 504 one-factor recovery/resource sensitivity runs. |
| `python reproduce.py figures` | Refresh numerical tables and the architecture figure using archived metrics; retains older trace figures. |
| `python reproduce.py full` | Recompute calibration, benchmarks, all campaigns, figures and numerical tables. This is the expensive numerical pipeline. |
| `python reproduce.py external` | Recompute external density comparisons after ESA/SET input retrieval and JB2008 compilation. Requires network and local setup. |

Use `NPROC=2 python reproduce.py r8` to reduce workers. Local checkpoints are protocol-hash guarded. Delete a campaign's `data/*checkpoint.json` to rerun it afresh. Numerical results are deterministic in the tested environment; timing is not. Other numerical-library environments may introduce floating-point differences.

The notebooks use **manual ZIP upload**. `TAES_R8_resource_sensitivity_colab.ipynb` reproduces the latest campaign. Notebook code was syntax checked, but not executed on Google Colab. Upload a ZIP containing this repository; the archive root name is immaterial.

## Structure

| Path | Contents |
|---|---|
| `code/` | Simulator, predictor, estimator, QP/PI policies, campaigns and numerical verification |
| `data/` | Calibration inputs, fixed weather-index snapshot, campaign JSON/CSV and selected computed traces |
| `external_validation/` | Swarm/JB2008 scripts and derived comparison CSVs |
| `figures/` | Standalone scientific figures |
| `provenance/` | Frozen R6 simulator and numerical-repeat records |
| `engineering_first_sunlit/` | Disclosed preliminary R7 runs, excluded from final claims |
| `R6_COMPARISON_PROTOCOL.md`, `R7_COMPARISON_PROTOCOL.md`, `R8_SENSITIVITY_PROTOCOL.md` | Written experimental designs |

Full per-step traces for every run are regenerated rather than all archived. Raw ESA density records and SET software/indices are obtained from the providers and are not redistributed. Journal correspondence and referee reports are outside this reproduction repository.

## Findings and limits

- R6: only 14/72 PI/MPC pairs qualify; 1308 QP failures and the unsuitable terminal target remain reported.
- R7: 120/120 PI/MPC pairs qualify after shared recovery. May2024/400km saves 7.63–7.96%, with worse tracking and lower reserves; four other event/height cases favor PI in fuel.
- R8: 230/252 pairs qualify. All 36 qualified May2024/400km pairs save 6.62–8.80%. A 24h recovery delay can prevent matching; a 6m² recovery ceiling causes unmet load and battery-floor violations. Unqualified raw fuel differences are not matched-resource savings.

R8 resource values are numerical perturbations, not hardware specifications. Repeated seeds on three storms are not hundreds of independent observed events. Swarm comparison does not validate closed-loop control on the simulated orbit.

## Attribution and reuse

See `CITATION.cff` and `DATA_PROVENANCE.md`. No additional open-source license is granted here; the authors may select one separately. Third-party materials remain subject to their own terms.

Funding: The Researchers would like to thank the Deanship of Graduate Studies and Scientific Research at Qassim University (www.qu.edu.sa) for financial support (QU-APC-2026).

Repository: https://github.com/msalih71/leo-coupled-mpc-reproducibility
