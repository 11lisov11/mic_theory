"""Verify AIR56 run lineage, summarize paired PWM and hash the release artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_trials(path):
    with path.open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(rows):
    keys = ("angle_p95_deg", "full_angle_mae_deg", "speed_rmse_rad_s", "probe_fraction",
            "probe_copper_energy_j", "positive_electrical_input_j", "switch_loss_proxy_j",
            "switches_per_s", "torque_ripple_rms_nm", "current_peak_a", "marker_terminal_probe_energy_delta_j")
    summary = {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
    widths = [float(row["hpd95_width_deg"]) for row in rows if row["hpd95_width_deg"]]
    widths = [x for x in widths if np.isfinite(x)]
    summary["hpd95_width_deg"] = float(np.mean(widths)) if widths else None
    return summary


def paired_pwm(marker_path, control_path):
    marker_rows, control_rows = read_trials(marker_path), read_trials(control_path)
    assert len(marker_rows) == len(control_rows) and marker_rows
    summarize = lambda rows: {method: metrics([r for r in rows if r["method"] == method])
                              for method in sorted({r["method"] for r in rows})}
    matched = {}
    for marker, control in zip(marker_rows, control_rows):
        for key in ("seed", "scenario", "speed_fraction", "mismatch", "method", "stimulus_digest", "period_deg"):
            assert marker[key] == control[key], (key, marker[key], control[key])
        matched.setdefault(marker["method"], []).append(float(marker["angle_p95_deg"])-float(control["angle_p95_deg"]))
    records = {method: dict(mean_deg=float(np.mean(diff)), max_abs_deg=float(np.max(abs(np.array(diff)))))
               for method, diff in matched.items()}
    delta = []
    by_case = {}
    quantity = "angle_p95_deg"
    for r in marker_rows:
        by_case.setdefault((r["seed"], r["scenario"], r["speed_fraction"], r["mismatch"], r["stimulus_digest"]), {})[r["method"]] = r
    for key, arms in by_case.items():
        if not {"B2_fixed_probe", "B3_info_fisher"}.issubset(arms):
            continue
        fixed, adaptive = arms["B2_fixed_probe"], arms["B3_info_fisher"]
        if float(fixed["probe_fraction"]) <= 0 or abs(float(fixed["probe_fraction"])-float(adaptive["probe_fraction"])) > 1e-12:
            continue
        delta.append((int(key[0]), float(adaptive[quantity])-float(fixed[quantity])))
    rng = np.random.default_rng(994865)
    grouped = np.array([np.mean([v for group, v in delta if group == seed]) for seed in sorted({s for s, _ in delta})])
    budget = None
    if len(grouped):
        boot = grouped[rng.integers(0, len(grouped), (4000, len(grouped)))].mean(axis=1)
        budget = {"case_count": len(delta), "seed_groups": len(grouped),
                  "adaptive_minus_fixed_p95_deg": float(grouped.mean()),
                  "bootstrap_95_deg": np.quantile(boot, (0.025, 0.975)).tolist()}
    return {"trials_per_channel": len(marker_rows), "marker": summarize(marker_rows),
            "unmarked": summarize(control_rows), "paired_effect_of_marker": records,
            "equal_nonzero_probe_count_fixed_vs_fisher": budget}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    paths = {
        "baseline": "baseline_smoke", "cad": "cad", "fem": "fem_release",
        "observability": "observability_paired", "coded_pwm_marker": "coded_pwm_marker_paired",
        "coded_pwm_control": "coded_pwm_control_paired", "controller": "controller_release",
        "visuals": "visuals_release",
    }
    base = args.root
    fem = read_json(base/paths["fem"]/"fem_summary.json")
    assert len(read_json(base/paths["fem"]/"refinement_check.json")) == 3
    for path, digest in fem["files_sha256"].items():
        assert sha256(base/paths["fem"]/path) == digest
    for path, digest in fem["inputs_canonical_sha256"].items():
        source = ROOT/"research/air56_interlaminar_marker"/path
        assert hashlib.sha256(source.read_text(encoding="utf8").encode("utf8")).hexdigest() == digest
    source_hash = fem["files_sha256"]["channel.npz"]
    obs = read_json(base/paths["observability"]/"observability.json")
    assert source_hash == obs["data_sha256"]
    for key in ("coded_pwm_marker", "coded_pwm_control"):
        summary = read_json(base/paths[key]/"summary.json")
        assert source_hash == summary["plant_extension"]["data_sha256"]
    assert source_hash == read_json(base/paths["controller"]/"controller_summary.json")["channel_sha256"]
    assert read_json(base/paths["baseline"]/"summary.json")["trial_count"] == 126
    assert max(x["energy_balance_rel"] for x in fem["mesh_3d"]) < 5e-7
    assert max(x["reciprocity_rel"] for x in fem["mesh_3d"]) < 5e-5
    assert all(abs(x["thermal_energy_balance"]-1) < 1e-6 for x in (fem["control_fields"], fem["marker_fields"]))
    cad = read_json(base/paths["cad"]/"cad_summary.json")["results"]
    assert all(v["objects"] == v["step_solids"] for v in cad.values())
    pwm = paired_pwm(base/paths["coded_pwm_marker"]/"trials.csv", base/paths["coded_pwm_control"]/"trials.csv")
    output = base/"paired_pwm_release.json"
    output.write_text(json.dumps(pwm, indent=2, allow_nan=False)+"\n", encoding="utf8")
    artifacts = [x for path in paths.values() for x in (base/path).iterdir() if x.is_file()]
    artifacts += [output, base/"ex33.py", base/"ex33_solution.vtk"]
    artifacts += [base/"regression_release.log"] if (base/"regression_release.log").exists() else []
    assert all(x.stat().st_size for x in artifacts if x.suffix != ".log")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = dict(branch=branch, git_head=head, fem_versions=fem["versions"], results=paths,
                    files={x.relative_to(base).as_posix(): {"sha256": sha256(x), "bytes": x.stat().st_size}
                           for x in sorted(artifacts)})
    (base/"release_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    print(f"Verified {len(artifacts)} artifacts and {pwm['trials_per_channel']} paired PWM trial keys")


if __name__ == "__main__":
    main()
