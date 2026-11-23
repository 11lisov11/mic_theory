"""Freeze training-selected references, then run untouched common-cap holdout."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from tools import study_torque_capped_adaptation as sweep
from tools.analyze_parameter_adaptive_speed import analyze

ROOT = sweep.benchmark.ROOT
PROTOCOL = ROOT/"docs/parameter_adaptive_speed_confirmation_protocol.md"
FAMILIES = dict(joint_only="joint_estimator", direct_feedback="joint_coupled", confidence_feedback="joint_confidence")


def selection_from_training(result):
    validated = analyze(result)
    if result["settings"]["torque_caps_nm"] != [.6] or len(result["seeds"]) < 8:
        raise ValueError("eight common-cap training seeds required")
    selected = {}
    for case in result["cases"]:
        rows = [r for r in validated["aggregates"] if r["scenario"] == case]
        fixed = [r for r in rows if r["method"].startswith("fixed_")]
        if len(fixed) != 4:
            raise ValueError("four strongest-reference tuning arms required")
        best = min(fixed, key=lambda r: (r["tracking_mae_rad_s"], r["method"]))["method"]
        selected[case] = dict(fixed_reference=best.removesuffix("_t060"), **FAMILIES)
    return selected


def execute(job):
    job = dict(job)
    label = job.pop("label")
    row, trace = sweep.configured_trial(job)
    row["configured_method"] = row["method"]
    row["method"] = label
    return row, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "run"])
    parser.add_argument("--training", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    if args.stage == "select":
        if args.training is None or args.selection.exists():
            parser.error("new selection file and training result required")
        raw = json.loads(args.training.read_text(encoding="utf-8"))
        selected = selection_from_training(raw)
        paths = (Path(__file__).resolve(), PROTOCOL)
        hashes = dict(raw["source_sha256"])
        hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
        manifest = dict(selection=selected, source_sha256=hashes, bundle_sha256=raw["bundle_sha256"],
            training_path=str(args.training.resolve()), training_sha256=hashlib.sha256(args.training.read_bytes()).hexdigest(),
            holdout_seeds=list(range(97200, 97220)), torque_cap_nm=.6,
            primary_candidate="confidence_feedback", primary_cases=["parameter_drift", "composite"],
            bootstrap_draws=10000, bootstrap_seed=103314, confidence=.975)
        args.selection.parent.mkdir(parents=True, exist_ok=True)
        args.selection.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(selected))
        return
    if args.out is None or (args.out/"results.json").exists() or args.jobs < 1:
        parser.error("new output directory and positive jobs required")
    manifest = json.loads(args.selection.read_text(encoding="utf-8"))
    for p, h in manifest["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h:
            raise ValueError(f"source changed: {p}")
    if hashlib.sha256(Path(manifest["training_path"]).read_bytes()).hexdigest() != manifest["training_sha256"]:
        raise ValueError("training changed")
    bundle = sweep.benchmark.DEFAULT_BUNDLE
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != manifest["bundle_sha256"]:
        raise ValueError("motor bundle changed")
    selection_hash = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    nominal = sweep.benchmark.nominal_motor(json.loads(bundle.read_text(encoding="utf-8")))
    jobs = [dict(nominal=nominal, seed=seed, case=case, method=base, label=label,
        torque_limit=manifest["torque_cap_nm"], duration=2.5, save_trace=seed == manifest["holdout_seeds"][0])
        for case, arms in manifest["selection"].items() for seed in manifest["holdout_seeds"] for label, base in arms.items()]
    rows, traces = [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for row, trace in pool.map(execute, jobs):
            rows.append(row)
            if trace:
                traces.append(dict(scenario=row["scenario"], method=row["method"], trace=trace))
            if len(rows) % 20 == 0:
                print(f"{len(rows)}/{len(jobs)}; failures {sum(r['failure'] is not None for r in rows)}", flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in manifest["source_sha256"].items())
    assert hashlib.sha256(args.selection.read_bytes()).hexdigest() == selection_hash
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == manifest["bundle_sha256"]
    result = dict(rows=rows, traces=traces, seeds=manifest["holdout_seeds"], cases=list(manifest["selection"]),
        methods=["fixed_reference", *FAMILIES], settings=dict(dt_s=1e-4, duration_s=2.5, torque_caps_nm=[.6]),
        selection=manifest, selection_sha256=selection_hash, source_sha256=manifest["source_sha256"],
        runtime_s=time.perf_counter()-started, hardware_claim=False, novelty_claim=False)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(trials=len(rows), runtime_s=result["runtime_s"])))


if __name__ == "__main__":
    main()
