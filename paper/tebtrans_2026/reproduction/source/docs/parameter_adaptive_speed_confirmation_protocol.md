# Prespecified closed-loop confirmation, 2026-10-06

All results below are from simulation, not power-stage operation. Old sampling
studies are not control evidence. Joint parameter EKF is established prior art.

## Frozen selection before confirmation

Use common torque envelope .6 Nm for EVERY arm: .45 left inadequate margin
above declared load up to .45 Nm plus friction at 150 rad/s; .6 preserves a
margin without the old 1.1 Nm transient budget. Current limit 2.8 A, voltage
limit, speed bandwidth 80 rad/s, robustness and delay replay remain identical.
No cap or Q retuning on confirmation. This is the second stage, explicitly
informed by exploratory pilot/cap grid, not a retroactive prespecification.

Training: 96212--96219, all four Q values and three joint-adaptive alternatives,
four common scenarios. Choose strongest fixed Q by smallest training tracking
MAE independently in each named operating regime; resolve ties lexicographically.
This allows an aggressively regime-tuned reference, not an online oracle.
Each drive's operating regime is fixed externally; no actual hidden state or
fault label is supplied to the policy. Three candidate families are frozen:
joint observer/fixed FOC parameters; always-coupled FOC; contraction-coupled FOC.

Save exact training SHA, selected methods, source/input hashes and this protocol
in selection.json before launching holdout. Holdout seeds 97200--97219 have not
been inspected. Four arms x four cases x20 independent seeds =320 trials.

## Primary hypotheses, gates and analysis

Prespecified primary: contraction-coupled vs strongest fixed reference in
parameter_drift and composite, two separate endpoints. Paired seed-cluster
bootstrap 10000 draws seed 103314, two-sided CI97.5 for each MAE ratio (Bonferroni
family .05). Also report all other cases and adaptive references; do not average
ordinary and slow speed errors into an unnormalized single number.

Practical target: observed tracking improvement >=20%, ratio CI97.5 upper <1,
and mean-peak ratio <=1.05. Stronger peak non-inferiority requires its CI97.5
upper <=1.05; if only point estimate passes, state this explicitly. Zero failed,
incomplete or >2.8 A trials. Report copper energy, torque/current SD, full peak
range, censored recoveries and CPU time. Expected/mean peak is NOT a hard bound.
No numerical failure is not successful speed tracking. At 3 rad/s, the absolute
error and percentage of reference must be reported even if a large relative
improvement against a poor baseline occurs.

Other adaptive families provide strong causal alternatives but their comparisons
are secondary; no novelty is established if the mixture/heuristic cannot beat
ordinary always-coupled/joint EKF. Negative ordinary-speed results preclude a
universal-controller claim even when both primary degraded cases improve.

## Limits

F2 fitted electrical/mechanical prior, linear magnetic averaged-voltage plant,
unidentified hardware parameters, shared predictor/plant equation family.
Wrong leakage is unmodeled in the observer. Posterior contraction and constrained
parameter clipping do not prove correct identification or Bayesian coverage.
No angle oracle, absolute-angle estimate, inverter gate validation, MCU timing,
efficiency/THD proof, or firmware-release permission follows from this cohort.
