# Joint-parameter sensorless speed control: frozen exploratory screen

This stage answers the user's request for substantial closed-loop improvements,
not another small ADC timing result. No hardware is operated. The parameter
bundle is fitted F2 prior sample 128, not measured identification from a passport.

## Hypothesis and prior art

Current-only adaptation of Rs/Rr/Lm may resolve the prior fixed-model drift
failure; coupling controller parameters only after posterior contraction may
avoid acting on uncertain parameter estimates. Joint EKF adaptation is established
prior art, not a novelty claim: Barut et al. 2010 simultaneous Rs/Rr estimation,
https://journals.tubitak.gov.tr/elektrik/vol18/iss5/9/ ; Inan/Barut 2014 BI-EKF,
https://journals.tubitak.gov.tr/elektrik/vol22/iss3/7/ . Posterior contraction is
only a heuristic confidence proxy, not a proof of identifiability or calibration.

## Arms and physics

Strong fixed-Q references: load spectral density .0001/.001/.01/.1; all use
timestamp replay and identical NIS=25 robustness. Adaptive arms: Rs only;
joint Rs/Rr/Lm with fixed controller; joint adaptation coupled to FOC; and
joint adaptation coupled after confidence blending. Same load Q=.01, log
parameter prior SD=.2, Q=1e-5, bounds +/-.7. Lm changes preserve both leakage
inductances. Analytical nine-state RK4 tangents must match finite differences.
Clipping does not implement an exact constrained Bayesian posterior.

Nominal, parameter-drift, composite and slow-composite scenarios reuse the
declared previous stress generator but NEW seeds. Random initial Rs/Rr/J/B
variation, Lm/leakage mismatch, Rs thermal ramp, noise/dropouts/impulses as
declared in the old runner. The controller sees no scenario label or actual
parameters; the requested speed profile is an exogenous task specification.
All arms receive common random samples; currents differ causally with control.
No absolute mechanical angle is estimated.

## Cohorts and decision rules

Pilot/exploration 96200--96201. Training extension 96202--96209 if justified.
Reserve 97200--97219 for untouched confirmation; do not run until an analysis
and selection contract is separately frozen. No claims from two pilot seeds.
Each trial 2.5 s at Ts=100 us. Evaluation MAE t>=.8 s; steady SD t>=1.7 s;
current peak/copper integrate the whole run. Recovery: first post-load interval
of 100 ms with error <3 rad/s (slow: .3 rad/s); censor absent recovery explicitly.

A practical candidate needs >=20% tracking MAE improvement on an independently
confirmed declared regime, no numerical failures, and <=5% mean peak increase.
Also report RMS/SD torque and current, copper energy, individual limit violations,
recovery censoring, parameter clipping, runtime and computational cost. A good
mean cannot hide a failed or current-violating trial. Fixed 2.8 A is a reference
command envelope, not a physically guaranteed measured-current bound.
No tuning on confirmation and no pooling low/high-speed errors unnormalized.

## Limitations

Averaged voltage, linear magnetic plant, reconstructed converter voltage,
shared equation family and no hardware, THD or MCU deadline validation. Constant
parameter leakage is known; drift cannot be uniquely attributed in general.
This establishes a useful reference before testing genuinely new excitation /
parameter-control co-design. Large improvements over an inadequate fixed model
alone do not prove a new method beats established adaptive sensorless control.
