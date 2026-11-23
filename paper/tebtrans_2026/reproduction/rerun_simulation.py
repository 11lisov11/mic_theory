"""Run the frozen simulator from portable paths; compare with archived cases."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
DATA = REPO / 'experiments/parameter_adaptive_speed'
SOURCE = PAPER / 'reproduction/source'
sys.path.insert(0, str(SOURCE))


def configure_paths():
    from tools import study_current_only_speed_loop as loop
    from tools import study_parameter_adaptive_speed as benchmark
    loop.PLANT_SOURCE = SOURCE / 'external_model/induction_motor_current_flux_rk4.py'
    loop.DEFAULT_BUNDLE = PAPER / 'reproduction/motor_bundle.json'
    benchmark.DEFAULT_BUNDLE = loop.DEFAULT_BUNDLE


def execute(job):
    configure_paths()
    from tools.confirm_parameter_adaptive_speed import execute as original_execute
    return original_execute(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--full', action='store_true', help='Run all 320 original confirmation cases')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.jobs < 1 or out.exists() or out.is_relative_to(DATA):
        parser.error('Use positive jobs and a new output directory outside the historical data')

    # Check all packaged evidence before executing any research code.
    from verify_results import verify
    verified = verify()
    configure_paths()
    from tools import study_parameter_adaptive_speed as benchmark
    raw_path = DATA / 'confirmation_20261006/results.json'
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    bundle = json.loads(benchmark.DEFAULT_BUNDLE.read_text(encoding='utf-8'))
    nominal = benchmark.nominal_motor(bundle)
    cases = raw['cases'] if args.full else ['composite']
    seeds = raw['seeds'] if args.full else raw['seeds'][:1]
    labels = raw['methods'] if args.full else ['fixed_reference', 'confidence_feedback']
    jobs = [dict(nominal=nominal, seed=seed, case=case,
                 method=raw['selection']['selection'][case][label], label=label,
                 torque_limit=raw['selection']['torque_cap_nm'], duration=2.5,
                 save_trace=seed == raw['seeds'][0])
            for case in cases for seed in seeds for label in labels]
    out.mkdir(parents=True)
    started = time.perf_counter()
    rows, traces = [], []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=configure_paths) as pool:
        for row, trace in pool.map(execute, jobs):
            rows.append(row)
            if trace:
                traces.append(dict(scenario=row['scenario'], method=row['method'], trace=trace))
            print(f'{len(rows)}/{len(jobs)} complete', flush=True)
    lookup = {(r['scenario'], r['seed'], r['method']): r for r in raw['rows']}
    metrics = ('tracking_mae_rad_s', 'estimation_mae_rad_s', 'peak_current_a',
               'copper_energy_j', 'steady_torque_std_nm', 'steady_current_std_a')
    comparisons = []
    for row in rows:
        reference = lookup[(row['scenario'], row['seed'], row['method'])]
        identical_conditions = all(row[key] == reference[key] for key in
            ('actual_initial_parameters', 'load_start_s', 'load_level_nm', 'torque_limit_nm'))
        numeric = {key: dict(actual=row[key], archived=reference[key],
                   matches=math.isclose(row[key], reference[key], rel_tol=1e-7, abs_tol=1e-9)
                   if row[key] is not None else False) for key in metrics}
        comparisons.append(dict(scenario=row['scenario'], seed=row['seed'], method=row['method'],
            pass_=identical_conditions and row['failure'] is None and row['completed_steps'] == 25000
                  and all(value['matches'] for value in numeric.values()),
            identical_exogenous_conditions=identical_conditions, metrics=numeric))
    passed = all(row['pass_'] for row in comparisons)
    summary = dict(status='PASS' if passed else 'DIFFERENCES', full_confirmation=args.full,
                   cases=len(rows), runtime_s=time.perf_counter()-started,
                   original_sha256=verified['raw_sha256'],
                   wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   portable_path_overrides_only=True, relative_tolerance=1e-7,
                   absolute_tolerance=1e-9, comparisons=comparisons)
    results = dict(rows=rows, traces=traces, settings=raw['settings'],
                   source='Portable rerun of the frozen original simulator', comparison=summary)
    for name, value in (('results.json', results), ('comparison.json', summary)):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=summary['status'], cases=len(rows), output=str(out)), ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
