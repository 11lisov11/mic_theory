"""Supplementary, post-review control at matched load process covariance.

Uses the unchanged original simulator and the original 20 disturbance seeds.
This is an additional analysis, not a newly independent confirmation sample.
"""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, r'C:\mic_theory')
from tools import study_parameter_adaptive_speed as benchmark
from tools.study_torque_capped_adaptation import configured_trial

HERE = Path(__file__).resolve().parent
OUT = HERE / 'matched_covariance_20261027'
RAW_PATH = Path(r'C:\mic_theory\experiments\parameter_adaptive_speed\confirmation_20261006\results.json')
CASES = ['parameter_drift', 'composite']
METRICS = ['tracking_mae_rad_s', 'peak_current_a', 'steady_torque_std_nm', 'copper_energy_j']


def run_job(job):
    row, trace = configured_trial(job)
    row['method'] = 'fixed_same_q'
    row['load_density'] = 0.01
    return row, trace


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    raw = json.loads(RAW_PATH.read_text(encoding='utf-8'))
    assert len(raw['seeds']) == 20
    assert benchmark.Q['fixed_q01'] == .01
    for path, expected in raw['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    OUT.mkdir(exist_ok=True)
    if (OUT / 'results.json').exists():
        raise FileExistsError('Do not overwrite research evidence')
    nominal = benchmark.nominal_motor(json.loads(benchmark.DEFAULT_BUNDLE.read_text(encoding='utf-8')))
    protocol = {
        'status': 'supplementary analysis specified after review of original results',
        'question': 'Does changing load process density alone account for the comparison with parameter-adaptive observers?',
        'original_results': str(RAW_PATH),
        'original_sha256': hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
        'seeds': raw['seeds'],
        'cases': CASES,
        'additional_method': 'fixed_same_q',
        'simulator_method': 'fixed_q01',
        'load_density': .01,
        'all_common_state_P0_Q_R': 'same as adaptive observer; no parameter states in the fixed observer',
        'torque_limit_nm': .6,
        'duration_s': 2.5,
        'dt_s': benchmark.DT,
        'metrics': METRICS,
        'comparators': raw['methods'],
        'reused_original_realizations': True,
        'independent_new_holdout': False,
        'bootstrap_draws': 10000,
        'bootstrap_seed': 20261027,
        'bootstrap_confidence': .95,
        'interval_interpretation': 'exploratory paired intervals without multiplicity adjustment',
        'source_sha256': raw['source_sha256'],
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'nominal_motor': asdict(nominal),
    }
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    jobs = [dict(nominal=nominal, seed=seed, case=case, method='fixed_q01',
                 duration=2.5, torque_limit=.6, save_trace=seed == raw['seeds'][0])
            for case in CASES for seed in raw['seeds']]
    started = time.perf_counter()
    rows, traces = [], []
    with ProcessPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            row, trace = future.result()
            assert row['failure'] is None and row['completed_steps'] == 25000
            original = next(r for r in raw['rows'] if r['seed'] == row['seed'] and r['scenario'] == row['scenario'] and r['method'] == 'confidence_feedback')
            for key in ['actual_initial_parameters', 'load_start_s', 'load_level_nm']:
                assert row[key] == original[key], key
            rows.append(row)
            if trace:
                traces.append({'scenario': row['scenario'], 'method': row['method'], 'trace': trace})
            print(f'{len(rows)}/{len(jobs)} completed, no failures; elapsed {time.perf_counter()-started:.1f} s', flush=True)
    rows.sort(key=lambda row: (row['scenario'], row['seed']))
    for path, expected in raw['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    result = dict(protocol=protocol, rows=rows, traces=traces, runtime_s=time.perf_counter()-started)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    indices = rng.integers(0, 20, size=(10000, 20))
    summary = {}
    for case in CASES:
        matched = sorted([r for r in rows if r['scenario'] == case], key=lambda r: r['seed'])
        summary[case] = {'fixed_same_q': {key: statistics.mean(r[key] for r in matched) for key in METRICS}}
        for method in raw['methods']:
            comparison = sorted([r for r in raw['rows'] if r['scenario'] == case and r['method'] == method], key=lambda r: r['seed'])
            summary[case][method] = {}
            for key in METRICS:
                baseline = np.array([r[key] for r in matched])
                candidate = np.array([r[key] for r in comparison])
                ratios = np.mean(candidate[indices], axis=1) / np.mean(baseline[indices], axis=1)
                ratio = float(np.mean(candidate) / np.mean(baseline))
                summary[case][method][key] = {
                    'mean': float(np.mean(candidate)),
                    'ratio_to_fixed_same_q': ratio,
                    'percent_change': 100 * (ratio - 1),
                    'ratio_ci95': np.quantile(ratios, [.025, .975]).tolist(),
                }
    (OUT / 'analysis.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
