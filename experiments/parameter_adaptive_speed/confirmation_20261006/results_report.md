# Closed-loop parameter-adaptive confirmation

MIXED / NICHE; no new-law or hardware claim.
320 complete trials; 20 paired seed clusters; no numerical failures.

| Case | Candidate | MAE change % | Peak change % | Copper change % | Torque SD change % |
|---|---|---:|---:|---:|---:|
| nominal | joint_only | -22.58 | 16.97 | 1.83 | 23.44 |
| nominal | direct_feedback | 88.39 | -14.07 | -0.07 | -62.20 |
| nominal | confidence_feedback | 83.60 | -13.38 | -0.19 | -61.61 |
| parameter_drift | joint_only | -80.83 | 32.71 | -4.86 | -32.01 |
| parameter_drift | direct_feedback | -66.95 | -0.38 | 3.05 | -34.00 |
| parameter_drift | confidence_feedback | -66.18 | -0.15 | 2.99 | -34.34 |
| composite | joint_only | -79.52 | 38.42 | -5.07 | -33.96 |
| composite | direct_feedback | -68.65 | 1.51 | 2.71 | -33.35 |
| composite | confidence_feedback | -67.52 | 0.84 | 2.63 | -31.83 |
| slow_composite | joint_only | -70.21 | 18.91 | -2.71 | -53.06 |
| slow_composite | direct_feedback | -63.63 | -7.10 | 5.09 | -46.64 |
| slow_composite | confidence_feedback | -62.45 | -7.14 | 4.96 | -47.18 |

Primary degraded-case comparisons (Bonferroni CI97.5):
```json
[
  {
    "scenario": "parameter_drift",
    "candidate": "confidence_feedback",
    "baseline": "fixed_reference",
    "metrics": {
      "tracking_mae_rad_s": {
        "candidate_mean": 1.2584871859264584,
        "baseline_mean": 3.7207980492567705,
        "ratio": 0.33823044660481943,
        "ratio_ci975": [
          0.22537199389924692,
          0.473224761187891
        ]
      },
      "peak_current_a": {
        "candidate_mean": 1.2820313214886991,
        "baseline_mean": 1.2839131247694089,
        "ratio": 0.998534321953405,
        "ratio_ci975": [
          0.97742794119632,
          1.0196543270309857
        ]
      },
      "copper_energy_j": {
        "candidate_mean": 33.7876257566795,
        "baseline_mean": 32.80702266783787,
        "ratio": 1.0298900360075331,
        "ratio_ci975": [
          1.0249953994253869,
          1.0355169182973607
        ]
      },
      "steady_torque_std_nm": {
        "candidate_mean": 0.005475981763613924,
        "baseline_mean": 0.00834051101019628,
        "ratio": 0.6565523091953879,
        "ratio_ci975": [
          0.625487629952766,
          0.6879750078166108
        ]
      },
      "steady_current_std_a": {
        "candidate_mean": 0.010761830261140077,
        "baseline_mean": 0.03010975902259915,
        "ratio": 0.3574200063528469,
        "ratio_ci975": [
          0.33888488337043976,
          0.37643198834705066
        ]
      }
    },
    "statistical_mae_pass": true,
    "observed_practical_pass": true,
    "peak_noninferiority_pass": true
  },
  {
    "scenario": "composite",
    "candidate": "confidence_feedback",
    "baseline": "fixed_reference",
    "metrics": {
      "tracking_mae_rad_s": {
        "candidate_mean": 1.2238994026142906,
        "baseline_mean": 3.7686041702075395,
        "ratio": 0.3247619933899531,
        "ratio_ci975": [
          0.24092033361248585,
          0.4281204324294086
        ]
      },
      "peak_current_a": {
        "candidate_mean": 1.290675819121137,
        "baseline_mean": 1.2798831609822505,
        "ratio": 1.0084325338967688,
        "ratio_ci975": [
          0.9895591571477681,
          1.0273991392018977
        ]
      },
      "copper_energy_j": {
        "candidate_mean": 33.75875044160772,
        "baseline_mean": 32.893810711191875,
        "ratio": 1.0262949081214707,
        "ratio_ci975": [
          1.0226034647271112,
          1.0305269186262873
        ]
      },
      "steady_torque_std_nm": {
        "candidate_mean": 0.0063242602538005855,
        "baseline_mean": 0.009277665396216165,
        "ratio": 0.6816650508197778,
        "ratio_ci975": [
          0.6418147360994183,
          0.7224717681986218
        ]
      },
      "steady_current_std_a": {
        "candidate_mean": 0.012455956671585087,
        "baseline_mean": 0.02837948419257964,
        "ratio": 0.4389070846763993,
        "ratio_ci975": [
          0.410646218007764,
          0.4671496509045946
        ]
      }
    },
    "statistical_mae_pass": true,
    "observed_practical_pass": true,
    "peak_noninferiority_pass": true
  }
]
```

Full adaptive comparisons, peaks, recovery censoring and runtimes are in JSON/CSV.
Improvement over a fixed model does not establish novelty. Contraction is not proof of identification.
Averaged linear plant and F2 fitted prior; no absolute angle, switching-level or hardware validation.
