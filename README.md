# MIC AI · Induction-drive research

Simulation, estimation and supervisory control for induction motors. This repository contains the machine models, controller comparisons, experiment protocols and publication artifacts. The companion [MIC AI Drive Bench](https://github.com/11lisov11/mic_practice) documents the STM32G431 development stand and its firmware.

## Current results

| Research track | Evidence | Scope |
| --- | --- | --- |
| Parameter-adaptive speed control / TEBTRANS 2026 | [Latest article, figures and reproduction](paper/tebtrans_2026/), [complete experiment chain](experiments/parameter_adaptive_speed/) | In two prespecified parameter-disturbance cases at 150 rad/s, speed MAE fell by 66–68% and torque SD by 32–34%; mean trial peak current changed by less than 1%, with about 3% higher stator copper-loss energy. Averaged-voltage simulation. |
| Coded PWM and rotor observability | [Methods and results](docs/info_optimal_coded_pwm_research.md), [trial tables](experiments/info_optimal_coded_pwm/) | The symmetric model gives no absolute mechanical-angle information. The added harmonic is hypothetical and periodic. On blind seeds, adaptive probes missed the 30% P95 error-reduction target against fixed probes at equal budget. |
| Confidence under motion | [Motion holdout](experiments/info_optimal_coded_pwm/confidence_transfer_holdout/summary.json), [predictive surprise](experiments/info_optimal_coded_pwm/predictive_surprise_holdout/summary.json) | A standstill confidence gate admitted large errors under motion. Predictive surprise showed no demonstrated advantage over matched random selection on the independent holdout. |
| Supervisory flux optimization | [Three-motor benchmark package](paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/), [verification report](paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/VERIFY_SUBMISSION_CANDIDATE.json) | Software benchmark for AIR56, AL31 and AO2. The verification report records the publication-package checks for that run. |
| Safe neural-horizon PWM | [Method and baseline audits](docs/safe_neural_horizon_pwm_research.md) | Host-side controller research, including baseline strength and algorithm-identity checks. |

The coded-PWM results are simulations with prescribed motion and synthetic observation channels. Hardware and safety validation are separate milestones. The [Uno Q deployment prototype](docs/air56_unoq_bringup.md) describes a different architecture from the G431 bench.

## Reproduce a small experiment

Use Python 3.10 or newer. Run from the repository root.

```sh
python -m pip install -r requirements.txt
python -m pytest -q tests/test_info_optimal_coded_pwm.py tests/test_predictive_rotor_surprise.py
python tools/run_info_optimal_coded_pwm_study.py --suite smoke --seeds 1 --steps 80 --seed-offset 6000 --out outputs/coded-pwm-demo
```

The demo is a one-seed execution check. The [research report](docs/info_optimal_coded_pwm_research.md#reproduce) records the full seed ranges, commands, comparisons and limitations. Use [Step 28 tooling](tools/reproduce_ieee_step28.py) to regenerate a publication bundle. Some trained checkpoints require separate inputs listed in the [checkpoint registry](config/checkpoint_registry.json).

## Repository map

| Path | Contents |
| --- | --- |
| [`models/`](models/), [`config/`](config/) | Machine models, motor parameters, environments and study configurations |
| [`control/`](control/), [`mic_ai/`](mic_ai/) | Control, state estimation, training and evaluation |
| [`tools/`](tools/), [`tests/`](tests/) | Experiment runners, validation and regression tests |
| [`experiments/`](experiments/), [`paper/`](paper/) | Reproducible evidence and publication bundles |

Use [CITATION.cff](CITATION.cff) to cite the software. Cite the release or commit used for a result. Code is distributed under the [MIT License](LICENSE).
