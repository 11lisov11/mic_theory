# Parameter-adaptive speed control

EXPLORATORY ONLY: training/pilot, no confirmation claim.
Seed clusters: 8. All arms complete with no numerical failures.

| Scenario | Candidate | Strong fixed Q | MAE reduction % | Peak increase % | Copper change % | >2.8 A trials |
|---|---|---|---:|---:|---:|---:|
| nominal | joint_estimator | fixed_q1 | 8.69 | 14.87 | 0.15 | 0 |
| nominal | joint_confidence | fixed_q1 | -162.22 | -13.94 | -0.30 | 0 |
| parameter_drift | joint_estimator | fixed_q0001 | 78.03 | 28.48 | -5.63 | 0 |
| parameter_drift | joint_confidence | fixed_q0001 | 61.61 | 0.35 | 2.99 | 0 |
| composite | joint_estimator | fixed_q0001 | 77.78 | 33.05 | -5.89 | 0 |
| composite | joint_confidence | fixed_q0001 | 63.47 | 1.23 | 2.80 | 0 |
| slow_composite | joint_estimator | fixed_q0001 | 67.89 | 23.15 | -1.99 | 0 |
| slow_composite | joint_confidence | fixed_q0001 | 55.97 | -6.06 | 6.15 | 0 |

Fixed selection and exploratory intervals are not a prespecified holdout superiority test.
No novelty, hardware readiness, switching-level safety or parameter-identifiability proof.
Mean peak and current-command limits are not a bound on every measured-current sample.
