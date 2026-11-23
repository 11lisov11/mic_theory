# Causal parameter-adaptive induction-drive research

2026-10-06. This track moves from synthetic 350 us sampling windows to causal
closed-loop drive trajectories, in response to the request for substantial
speed-control improvement rather than a 1--2% sensing metric.

## Implemented law and observation contract

The nine-dimensional observer state is
`[i_alpha,i_beta,psi_r_alpha,psi_r_beta,omega_m,T_load,log(Rs/Rs0),log(Rr/Rr0),log(Lm/Lm0)]`.
Resistance/inductance estimates remain positive; Lm changes preserve known
leakage inductances. State and parameter columns of the continuous Jacobian
and differentiated RK4 transition are analytic and independently checked by
central finite differences. A zero-current/zero-flux unexcited trajectory
does not spuriously identify parameters. Correction uses Joseph covariance.

Only measured current, historical reconstructed voltage, sample indices,
declared noise covariance and nominal fitted parameter priors enter estimation.
FOC sees estimated state, requested speed, declared bus voltage and time. True
speed, flux, load, Rs/Rr/Lm are restricted to the independent plant and metrics.
No absolute mechanical angle is estimated. Chronological delay replay and NIS
outlier weighting are identical across compared families.

Three adaptive/controller combinations isolate the control effect:

- Joint EKF with unchanged nominal FOC parameters.
- Joint EKF with directly updated Rs/Rr/Lm in FOC feedforward/reference scaling.
- The same update blended by posterior contraction: for log parameter `z_j`,
  use `g_j*z_j`, `g_j=clip(1-P_jj/.04,0,1)` in FOC.

The last rule is a testable heuristic, NOT a proof that the parameter is
identifiable or calibrated. Log estimates are clipped to +/-.7; covariance is
not the exact posterior of a truncated distribution. Nonidentifiability and
wrong-leakage bias can remain even with small covariance and good speed control.

## Exploration and fairness

All source/protocol/input hashes are retained with each cohort. Older frozen
sources and historical evidence were not overwritten.

| Stage | Seeds | Complete trials | Purpose |
|---|---|---:|---|
| Initial pilot | 96200--96201 | 64 | Eight estimator/controller alternatives |
| Initial training | 96202--96209 | 192 | Four fixed-Q / two adaptive families |
| Equal torque-cap grid | 96210--96211 | 160 | Common .45/.6/.8/1.1 Nm across all arms |
| Common .6 Nm training | 96212--96219 | 224 | Four fixed-Q / three adaptive families |
| Untouched confirmation | 97200--97219 | 320 | Four frozen arms in four regimes |

Initial training revealed large drift-case MAE reductions but a transient
peak-current penalty for the joint-observer-only variant. Updating FOC from
the observer greatly changed this tradeoff; ordinary-speed nominal tracking
could worsen. One training trial of joint-only/.6 Nm exceeded the 2.8 A
reference envelope despite a bounded torque command. Do not hide this fact.

A common .6 Nm envelope was selected for the final stage: .45 lacked margin
over the declared maximum .45 Nm load plus friction. Reference load-process
Q is selected among .0001/.001/.01/.1 on eight training seeds for each fixed
operating regime. This is a strong separately configured reference, not a
controller supplied with online fault labels. Three adaptive families remain
frozen. Exact selection is in `experiments/parameter_adaptive_speed/selection_20261006.json`.

Confirmation has two prespecified degraded-case endpoints, paired cluster
bootstrap CI97.5, current/peak gates and all-family comparisons. The source
contract is `docs/parameter_adaptive_speed_confirmation_protocol.md`. Do not
use the final seeds for tuning. Sampling windows within a run are not counted
as independent samples: the statistical unit is one seed cluster.

## Model and release limitations

Each trial is 2.5 s, Ts=100 us, with an external RK4 current-flux plant. It is
implemented separately but shares the predictor's linear induction equation
family. Voltage is averaged; physical converter drop depends on plant currents,
while reconstruction uses estimated currents. Scenarios add Lm/leakage mismatch,
Rs thermal ramp, anisotropic current noise and, in composite runs, 0.3 ms delay,
impulses, quantization and dropout. Slow-composite reference is 3 rad/s; other
cases use a 150 rad/s ramp. These are simulation tasks, not a complete AIR56
operating envelope. F2 sample 128 is a fitted prior, not a unique passport
identification or measured model. Neither firmware nor power hardware changed.

Tracking MAE uses t>=.8 s, torque/current SD t>=1.7 s. Whole-run current peaks
and copper-energy proxy include startup. A 2.8 A current-reference setting is
not a guaranteed physical current bound. Copper loss is not total efficiency.
Recovery is explicitly censored if no continuous 100 ms error interval below
3 rad/s (.3 rad/s slow) occurs. Runtime is host parallel-run evidence, not MCU
deadline performance. No THD, switching losses, dead-time, BKIN or interlocks
are validated by this averaged simulation.

## Completed confirmation results

All 320 holdout trials completed, no numerical failures and no observed >2.8 A
trials. The entire research block comprises 960 trials including tuning, but
confirmation has only 20 independent seed clusters, not 320 independent seeds.
Training/algorithm/input provenance checks passed; raw confirmation SHA256 is
`92f344e804b47553f8aa9cfe741c2b31501c4fabc36189144692a9c7f7041d5a`.
Holdout runtime 325.89 s with twelve host processes. Focused regression: 132
tests passed. Results/analysis/CSV and black-white PNG/SVG are in
`experiments/parameter_adaptive_speed/confirmation_20261006/`.

Prespecified contraction-coupled vs strongest training-selected fixed reference:

| Regime | Fixed speed MAE rad/s | Candidate MAE rad/s | Reduction | MAE ratio CI97.5 | Mean peak change | Peak ratio CI97.5 |
|---|---:|---:|---:|---|---:|---|
| Parameter drift | 3.720798 | 1.258487 | 66.18% | .2254--.4732 | -.15% | .9774--1.0197 |
| Composite | 3.768604 | 1.223899 | 67.52% | .2409--.4281 | +.84% | .9896--1.0274 |

Both speed hypotheses and <=5% mean-peak noninferiority screens pass their
declared CI97.5 rules. These are empirical paired-bootstrap results, not global
stability/current-bound theorems. Copper-energy proxy increases 2.99% / 2.63%;
steady torque SD decreases 34.34% / 31.83%; current SD decreases 64.26% / 56.11%.
Reference recovery is censored in 7/20 trials in each degraded case; none of
the coupled adaptive alternatives are censored there. This metric does not
assert permanent recovery outside the simulated duration.

The indispensable adaptive comparison changes the interpretation: direct
parameter coupling achieves MAE 1.229650 / 1.181358, i.e. is slightly better
than the contraction variant. Contraction/direct MAE ratio is 1.02345
(CI97.5 .9465--1.1137) in drift and 1.03601 (1.0025--1.0751) in composite;
these are secondary, not multiplicity-controlled new-law endpoints. There is
NO confirmed incremental benefit from the confidence rule. Large improvement
over a fixed model comes from established parameter-adaptive sensorless control.

Joint-EKF-only reaches .713146 / .771738 rad/s but increases mean peak roughly
32.7% / 38.4%, failing the common practical peak criterion. It is not promoted
simply for winning one accuracy column.

Nominal contraction variant MAE 1.613769 vs .878949 (83.60% worse). Slow-composite
MAE improves 5.528670 ->2.076051, but at a 3 rad/s request this is still 69.2%
of reference and every coupled trial lacks the tight .3 rad/s recovery window.
The low-speed problem and universal-controller claim remain unresolved.

Verdict: **MIXED / NICHE**. A substantial degradation-control effect is confirmed
under the declared model, and now supplies a strong adaptive reference for
new PWM/identification-control co-design. Scientific novelty, all-position
sensorless operation and hardware readiness are NOT established. No additional
tuning may use the consumed 97200--97219 cohort as an untouched holdout.

## Prior art and novelty boundary

[Ozsoy, Gokasan and Estrada, 2010](https://journals.tubitak.gov.tr/elektrik/vol18/iss5/9/)
already estimates rotor/stator resistances and states with a joint EKF. The
initial exploratory protocol mistakenly labels this citation as Barut et al.;
the DOI/link is correct, authors are corrected here without editing its frozen hash.
[Inan and Barut, 2014](https://journals.tubitak.gov.tr/elektrik/vol22/iss3/7/)
already includes Rs/Rr/Lm/load and sensorless vector control. Our nine-state
single EKF is not a full reproduction of their bi-input EKF, and its gain over
a fixed model must not be described as surpassing their published algorithm.

An online-first [excitation-aware adaptive dual EKF page](https://ejournal-purwakarta.upi.edu/jmai/article/view/178)
is also accessible, but lists a future publication date 2026-12-15 relative
to this report. Treat its date/publication status cautiously; the established
2010/2014 references alone already block a broad novelty claim for adaptation.

The defensible next new-law candidate is NOT ordinary parameter adaptation.
It is constrained PWM/current-sampling co-design directed at *control-relevant,
data-identifiable parameter directions*: remove state/load nuisance directions,
rank parameter uncertainty by its effect on future constrained control, and
spend excitation only when passive transitions cannot separate those directions.
This requires a fresh algorithm/protocol and comparison against the strongest
adaptive family established here, not against the fixed model alone. Conditional
information, dual control and input design themselves are established ideas;
the exact bounded implementation and incremental value still require audit.

## Reproduction

```powershell
python -m pytest tests/test_parameter_adaptive_induction_ekf.py tests/test_study_parameter_adaptive_speed.py tests/test_analyze_parameter_adaptive_speed.py tests/test_torque_capped_adaptation.py tests/test_parameter_adaptive_confirmation.py -q
python tools/study_parameter_adaptive_speed.py --seeds 2 --seed-offset 96200 --out experiments/parameter_adaptive_speed/pilot_NEW
python -m tools.study_torque_capped_adaptation --seeds 8 --seed-offset 96212 --caps .6 --methods fixed_q0001 fixed_q001 fixed_q01 fixed_q1 joint_estimator joint_coupled joint_confidence --out experiments/parameter_adaptive_speed/training_NEW
python -m tools.confirm_parameter_adaptive_speed select --training experiments/parameter_adaptive_speed/training_NEW/results.json --selection experiments/parameter_adaptive_speed/selection_NEW.json
python -m tools.confirm_parameter_adaptive_speed run --selection experiments/parameter_adaptive_speed/selection_NEW.json --out experiments/parameter_adaptive_speed/confirmation_NEW
python -m tools.analyze_parameter_adaptive_confirmation experiments/parameter_adaptive_speed/confirmation_NEW
```

`NEW` directories prevent overwriting evidence. Re-running consumed seeds is
reproduction, not a new independent confirmation cohort.
