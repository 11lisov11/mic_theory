# Equal torque-envelope exploration, frozen before running

The preceding 96200--96201 pilot suggests large joint-estimator tracking gains
but larger post-load peak currents. The confidence-coupled FOC is worse on
ordinary-speed tracking, so it is NOT promoted. This follow-up checks the
obvious control-envelope confound before any superiority claim.

NEW training seeds 96210--96211: 4 cases, fixed Q .0001/.001/.01/.1 and joint
estimator with unchanged FOC parameters; all receive the SAME torque-cap grid
.45/.6/.8/1.1 Nm. Ts=100 us, duration 2.5 s, same previous plant/noise generator.
These are exploratory tuning seeds, NOT independent confirmation. Scoped
process-local controller-factory injection changes ONLY the declared torque
limit; original plant/observer/stress harness remains byte-for-byte unchanged.

Choose one common torque envelope using training current/accuracy tradeoffs;
then extend training on fresh 96212--96219 if feasible. A separate pre-test
selection manifest and statistical contract must be frozen before 97200--97219.
Do not silently retune a holdout after seeing a bad result.

Large reductions must also satisfy <=5% mean peak increase vs strongest fair
fixed reference, no failed/current-violating trials, and report copper, ripple,
recovery censoring and CPU cost. Better speed estimates do not imply correct
parameter identification. Torque clipping/adaptive EKF are known methods, not
novel on their own. This tests usable physical control benefit before advancing
PWM excitation + uncertainty-aware adaptation as a genuinely new candidate.
