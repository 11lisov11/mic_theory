# Parameter-adaptive speed control

EXPLORATORY ONLY: training/pilot, no confirmation claim.
Seed clusters: 8. All arms complete with no numerical failures.

| Scenario | Candidate | Strong fixed Q | MAE reduction % | Peak increase % | Copper change % | >2.8 A trials |
|---|---|---|---:|---:|---:|---:|
| nominal | joint_estimator_t060 | fixed_q1_t060 | -31.66 | 37.94 | 9.66 | 1 |
| nominal | joint_coupled_t060 | fixed_q1_t060 | -75.83 | -16.33 | -0.25 | 0 |
| nominal | joint_confidence_t060 | fixed_q1_t060 | -78.64 | -16.05 | -0.25 | 0 |
| parameter_drift | joint_estimator_t060 | fixed_q0001_t060 | 78.57 | 37.23 | -4.43 | 0 |
| parameter_drift | joint_coupled_t060 | fixed_q0001_t060 | 60.70 | -1.32 | 2.50 | 0 |
| parameter_drift | joint_confidence_t060 | fixed_q0001_t060 | 60.72 | -2.15 | 2.53 | 0 |
| composite | joint_estimator_t060 | fixed_q0001_t060 | 80.84 | 43.82 | -4.66 | 0 |
| composite | joint_coupled_t060 | fixed_q0001_t060 | 67.71 | -2.99 | 2.31 | 0 |
| composite | joint_confidence_t060 | fixed_q0001_t060 | 69.78 | -2.45 | 2.31 | 0 |
| slow_composite | joint_estimator_t060 | fixed_q001_t060 | 65.73 | 15.16 | -2.53 | 0 |
| slow_composite | joint_coupled_t060 | fixed_q001_t060 | 47.87 | -9.53 | 4.46 | 0 |
| slow_composite | joint_confidence_t060 | fixed_q001_t060 | 51.63 | -9.68 | 4.51 | 0 |

Fixed selection and exploratory intervals are not a prespecified holdout superiority test.
No novelty, hardware readiness, switching-level safety or parameter-identifiability proof.
Mean peak and current-command limits are not a bound on every measured-current sample.
