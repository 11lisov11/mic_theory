# Parameter-adaptive speed control

EXPLORATORY ONLY: training/pilot, no confirmation claim.
Seed clusters: 2. All arms complete with no numerical failures.

| Scenario | Candidate | Strong fixed Q | MAE reduction % | Peak increase % | Copper change % | >2.8 A trials |
|---|---|---|---:|---:|---:|---:|
| nominal | rs_only | fixed_q1 | -7.86 | -8.52 | -0.02 | 0 |
| nominal | joint_estimator | fixed_q1 | 50.60 | 46.18 | 0.36 | 0 |
| nominal | joint_coupled | fixed_q1 | -201.34 | -17.82 | 0.10 | 0 |
| nominal | joint_confidence | fixed_q1 | -197.17 | -17.87 | 0.10 | 0 |
| parameter_drift | rs_only | fixed_q001 | -72.93 | 199.53 | 25.01 | 2 |
| parameter_drift | joint_estimator | fixed_q001 | 73.19 | 57.36 | -0.46 | 0 |
| parameter_drift | joint_coupled | fixed_q001 | -33.27 | -9.71 | 1.35 | 0 |
| parameter_drift | joint_confidence | fixed_q001 | -33.71 | -9.42 | 1.34 | 0 |
| composite | rs_only | fixed_q001 | -76.49 | 189.88 | 26.85 | 2 |
| composite | joint_estimator | fixed_q001 | 75.32 | 56.41 | -0.50 | 0 |
| composite | joint_coupled | fixed_q001 | -18.36 | -7.98 | 1.18 | 0 |
| composite | joint_confidence | fixed_q001 | -14.22 | -8.06 | 1.11 | 0 |
| slow_composite | rs_only | fixed_q01 | -27.43 | 289.83 | 57.16 | 2 |
| slow_composite | joint_estimator | fixed_q01 | 72.75 | 17.05 | -0.78 | 0 |
| slow_composite | joint_coupled | fixed_q01 | 46.75 | -14.26 | 2.79 | 0 |
| slow_composite | joint_confidence | fixed_q01 | 43.33 | -14.27 | 2.66 | 0 |

Fixed selection and exploratory intervals are not a prespecified holdout superiority test.
No novelty, hardware readiness, switching-level safety or parameter-identifiability proof.
Mean peak and current-command limits are not a bound on every measured-current sample.
