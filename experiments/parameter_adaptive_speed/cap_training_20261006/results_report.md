# Parameter-adaptive speed control

EXPLORATORY ONLY: training/pilot, no confirmation claim.
Seed clusters: 2. All arms complete with no numerical failures.

| Scenario | Candidate | Strong fixed Q | MAE reduction % | Peak increase % | Copper change % | >2.8 A trials |
|---|---|---|---:|---:|---:|---:|
| nominal | joint_estimator_t045 | fixed_q01_t045 | 3.85 | 10.69 | 0.21 | 0 |
| nominal | joint_estimator_t060 | fixed_q01_t045 | 7.85 | 18.77 | 0.29 | 0 |
| nominal | joint_estimator_t080 | fixed_q01_t045 | 7.85 | 18.77 | 0.29 | 0 |
| nominal | joint_estimator_t110 | fixed_q01_t045 | 7.85 | 18.77 | 0.29 | 0 |
| parameter_drift | joint_estimator_t045 | fixed_q0001_t045 | 80.80 | 24.55 | -6.79 | 0 |
| parameter_drift | joint_estimator_t060 | fixed_q0001_t045 | 80.08 | 35.60 | -6.76 | 0 |
| parameter_drift | joint_estimator_t080 | fixed_q0001_t045 | 80.08 | 35.60 | -6.76 | 0 |
| parameter_drift | joint_estimator_t110 | fixed_q0001_t045 | 80.08 | 35.60 | -6.76 | 0 |
| composite | joint_estimator_t045 | fixed_q0001_t045 | 85.02 | 31.36 | -7.13 | 0 |
| composite | joint_estimator_t060 | fixed_q0001_t045 | 83.39 | 44.41 | -7.08 | 0 |
| composite | joint_estimator_t080 | fixed_q0001_t045 | 83.39 | 44.41 | -7.08 | 0 |
| composite | joint_estimator_t110 | fixed_q0001_t045 | 83.39 | 44.41 | -7.08 | 0 |
| slow_composite | joint_estimator_t045 | fixed_q0001_t045 | 58.54 | 8.43 | -1.95 | 0 |
| slow_composite | joint_estimator_t060 | fixed_q0001_t045 | 59.60 | 17.92 | -1.91 | 0 |
| slow_composite | joint_estimator_t080 | fixed_q0001_t045 | 59.60 | 17.92 | -1.91 | 0 |
| slow_composite | joint_estimator_t110 | fixed_q0001_t045 | 59.60 | 17.92 | -1.91 | 0 |

Fixed selection and exploratory intervals are not a prespecified holdout superiority test.
No novelty, hardware readiness, switching-level safety or parameter-identifiability proof.
Mean peak and current-command limits are not a bound on every measured-current sample.
