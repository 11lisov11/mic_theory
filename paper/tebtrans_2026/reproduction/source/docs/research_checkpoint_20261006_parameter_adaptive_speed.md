# Research checkpoint: large closed-loop gain, novelty still open

Read `docs/parameter_adaptive_speed_research.md`, the three frozen protocols,
and `experiments/parameter_adaptive_speed/confirmation_20261006/results_report.md`.
This replaces the older checkpoint's immediate next-action suggestion.

## Latest completed block

Current-only nine-state log-Rs/Rr/Lm EKF, analytic RK4 tangents, Joseph update,
positive parameter mapping preserving leakage; direct and contraction-based
controller-parameter coupling. Process-local scoped torque-factory injection
reuses the frozen plant/loop without modifying historical sources.

960 complete research trials (64 pilot,192 initial training,160 cap-grid,
224 common-cap training,320 confirmation). Frozen selection common torque
.6 Nm; four fixed-Q choices trained separately in each regime. Holdout
97200--97219,20 independent paired seed clusters,4 arms x4 cases. No hardware
commands/firmware changes, no true angle/state supplied to policies.

Primary contraction-coupled vs tuned fixed reference:
drift MAE3.720798->1.258487 (-66.18%), ratio CI97.5 .2254--.4732;
composite3.768604->1.223899 (-67.52%), CI .2409--.4281.
Mean peak -.15% /+.84%, upper ratio CI1.0197/1.0274 (<=1.05).
Copper-energy proxy +2.99%/+2.63%; torque SD -34.34%/-31.83%.
All holdout numerical/current-envelope screens pass; one earlier joint-only
training trial exceeded2.8 A, retained in the report, never hidden.

Direct coupling is slightly BETTER than contraction on speed MAE; confidence
heuristic adds no confirmed value. Joint-only has best accuracy under drift
but peaks+32.7%/+38.4%. Nominal contraction MAE83.6%worse; slow3rad/s MAE2.076
remains unacceptable for tight tracking. Verdict MIXED/NICHE, not universal.

Raw holdout SHA92f344e804b47553f8aa9cfe741c2b31501c4fabc36189144692a9c7f7041d5a;
source/input/training hashes pass. Runtime325.89s,12processes. Full focused
regression132passed. Analysis figures are black/white Times New Roman;
first rendering had title over plot, corrected in report-only renderer.
Do not alter frozen sources/protocols and claim old source hashes still match.

## Earlier sampling stage now documented

ADC train92200--19/test93200--99; plain timing ~1.04% gain, calibration hybrid
not confirmed. Mixture train94200--19/test95200--99:1.897%gain vs fixed timing,
but not better than simpler greedy estimator. Coverage92.375%<nominal95%.
No drive-level gain/novelty from those single-window studies. Mixture figure
visually checked and completion appended to `docs/svm_sample_timing_research.md`.

## Next meaningful research action

Use direct-coupled/joint EKF as strong ADAPTIVE references, not just fixed EKF.
Target control-relevant conditional parameter information under wrong leakage,
sensor/voltage bias and thermal drift. Derive a nuisance-projected sensitivity
and cost-constrained update/excitation rule; compare actual control error,
current/torque, cost, calibrated uncertainty and runtime on NEW frozen cohorts.
No more tuning on97200--97219. Add nonlinear energy-saturation plant and other
motor parameter sets before generalizing this linear F2 result.

Broad parameter adaptation is prior art (Ozsoy2010/Inan-Barut2014); online-first
EAIA-DEKF page is accessible with future publication metadata2026-12-15.
Do not claim a new law from combining known names or only beating fixed Q.
Absolute shaft angle stays unobservable in the symmetric plant.

Preserve all dirty unrelated tracked changes and older artifacts. No commit,
push, automation edit, or hardware action was performed in this block. All
research processes finished before the close-out.
