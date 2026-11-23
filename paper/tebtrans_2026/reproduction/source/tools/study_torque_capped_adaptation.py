"""Equal-budget torque-envelope sweep using the frozen causal stress harness."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from tools import study_parameter_adaptive_speed as benchmark


def configured_trial(job):
    job = dict(job)
    torque_limit = job.pop("torque_limit")
    original = benchmark.CurrentOnlySpeedFOC
    def factory(motor, config):
        return original(motor, replace(config, torque_limit=torque_limit))
    # Process-local, scoped factory injection preserves the frozen plant/loop.
    with patch.object(benchmark, "CurrentOnlySpeedFOC", factory):
        row, trace = benchmark.run_trial(**job)
    row["base_method"] = row["method"]
    row["method"] += f"_t{round(torque_limit*100):03d}"
    row["torque_limit_nm"] = torque_limit
    return row, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-offset", type=int, default=96210)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--caps", type=float, nargs="+", default=[.45, .6, .8, 1.1])
    parser.add_argument("--methods", nargs="+", default=list(benchmark.METHODS[:4])+["joint_estimator"])
    parser.add_argument("--cases", nargs="+", default=list(benchmark.CASES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.jobs < 1 or any(c <= 0 for c in args.caps):
        parser.error("positive counts and caps required")
    if (args.out/"results.json").exists():
        parser.error("do not overwrite evidence")
    args.out.mkdir(parents=True, exist_ok=True)
    nominal = benchmark.nominal_motor(json.loads(benchmark.DEFAULT_BUNDLE.read_text(encoding="utf-8")))
    paths = [benchmark.ROOT/p for p in (
        "tools/study_torque_capped_adaptation.py", "tools/study_parameter_adaptive_speed.py",
        "control/parameter_adaptive_induction_ekf.py", "tools/study_current_observer_stress.py",
        "control/timestamped_current_observer.py", "docs/torque_capped_adaptation_protocol.md")]
    hashes = benchmark.source_hashes()
    hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    bundle_hash = hashlib.sha256(benchmark.DEFAULT_BUNDLE.read_bytes()).hexdigest()
    jobs = [dict(nominal=nominal, seed=args.seed_offset+i, case=c, method=m, torque_limit=cap,
                 duration=2.5, save_trace=i == 0)
            for c in args.cases for i in range(args.seeds) for m in args.methods for cap in args.caps]
    rows, traces = [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for row, trace in pool.map(configured_trial, jobs):
            rows.append(row)
            if trace:
                traces.append(dict(scenario=row["scenario"], method=row["method"], trace=trace))
            if len(rows) % 20 == 0:
                print(f"{len(rows)}/{len(jobs)}; failures {sum(r['failure'] is not None for r in rows)}", flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    assert hashlib.sha256(benchmark.DEFAULT_BUNDLE.read_bytes()).hexdigest() == bundle_hash
    result = dict(rows=rows, traces=traces, runtime_s=time.perf_counter()-started,
        source_sha256=hashes, bundle_sha256=bundle_hash, seeds=list(range(args.seed_offset, args.seed_offset+args.seeds)),
        cases=args.cases, methods=[m+f"_t{round(c*100):03d}" for m in args.methods for c in args.caps],
        hardware_claim=False, novelty_claim=False, settings=dict(dt_s=benchmark.DT, duration_s=2.5,
        torque_caps_nm=args.caps, parameter_density=1e-5, parameter_prior_sd=.2, log_bound=.7),
        contract="all base methods receive identical torque cap grid, physical exogenous seeds, current/command voltage only")
    (args.out/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(trials=len(rows), runtime_s=result["runtime_s"])))


if __name__ == "__main__":
    main()
