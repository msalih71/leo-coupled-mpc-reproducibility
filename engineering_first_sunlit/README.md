# Preliminary engineering outputs — excluded from the final R7 comparison

These runs tested endpoint selection and revealed why a first-sunlit-step
endpoint can stop before battery restoration. A prematurely launched run
also used the old endpoint implementation/checkpoint. Their protocol-hash
file records the intended protocol text, not compliance of these preliminary
runs with the final geometric boundary. They are retained for transparency.
The final 360-run data are in ../data/common_recovery_*; each final row stores
its protocol hash, and verify_common_recovery.py independently recomputes
its actual completion boundary. These engineering outputs are not used in
the final manuscript's new numerical claims. Redundant checkpoints are omitted.
