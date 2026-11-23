"""Verify the portable publication bundle without modifying historical evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
DATA = REPO / 'experiments/parameter_adaptive_speed'
SOURCE = PAPER / 'reproduction/source'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), f'{label}: {actual} != {expected}')


def verify():
    inventory = read(PAPER / 'manifest.json')
    for item in inventory['files']:
        path = (REPO / item['path']).resolve()
        require(path.is_relative_to(REPO), 'Manifest path outside the repository')
        require(path.is_file(), f'Missing file: {item["path"]}')
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], f'Hash mismatch: {item["path"]}')

    raw = read(DATA / 'confirmation_20261006/results.json')
    raw_sha = sha(DATA / 'confirmation_20261006/results.json')
    require(raw_sha == '92f344e804b47553f8aa9cfe741c2b31501c4fabc36189144692a9c7f7041d5a', 'Original evidence changed')
    mapping = {item['original_path']: item for item in read(PAPER / 'reproduction/source_paths.json')}
    require(set(mapping) == set(raw['source_sha256']), 'Frozen source mapping incomplete')
    for original, expected in raw['source_sha256'].items():
        require(sha(REPO / mapping[original]['repository_path']) == expected == mapping[original]['sha256'], f'Frozen source changed: {original}')
    require(sha(DATA / 'selection_20261006.json') == raw['selection_sha256'], 'Selection changed')
    require(sha(DATA / 'common_cap_training_20261006/results.json') == raw['selection']['training_sha256'], 'Training changed')
    require(sha(PAPER / 'reproduction/motor_bundle.json') == raw['selection']['bundle_sha256'], 'Motor input changed')
    require(len(raw['rows']) == 320 and raw['seeds'] == list(range(97200, 97220)), 'Confirmation size or seed set changed')

    sys.path.insert(0, str(SOURCE))
    from tools.analyze_parameter_adaptive_confirmation import analyze
    recalculated = analyze(raw)
    saved = read(DATA / 'confirmation_20261006/analysis.json')
    effects = {}
    for actual, expected in zip(recalculated['primary'], saved['primary']):
        require(actual['scenario'] == expected['scenario'], 'Primary scenario changed')
        for metric, values in actual['metrics'].items():
            for key in ('candidate_mean', 'baseline_mean', 'ratio'):
                close(values[key], expected['metrics'][metric][key], f'{actual["scenario"]}/{metric}/{key}')
            for a, b in zip(values['ratio_ci975'], expected['metrics'][metric]['ratio_ci975']):
                close(a, b, f'{actual["scenario"]}/{metric}/CI')
        require(actual['statistical_mae_pass'] and actual['peak_noninferiority_pass'], 'Primary comparison failed')
        values = actual['metrics']
        effects[actual['scenario']] = {
            'speed_error_reduction_percent': 100 * (1 - values['tracking_mae_rad_s']['ratio']),
            'torque_std_reduction_percent': 100 * (1 - values['steady_torque_std_nm']['ratio']),
            'mean_trial_peak_current_change_percent': 100 * (values['peak_current_a']['ratio'] - 1),
            'stator_copper_energy_change_percent': 100 * (values['copper_energy_j']['ratio'] - 1),
        }

    for item in read(PAPER / 'evidence/original/comparison_figure_audit.json'):
        rows = [r for r in raw['rows'] if r['scenario'] == item['scenario'] and r['method'] == item['method']]
        close(statistics.mean(r[item['metric']] for r in rows), item['mean'], 'Figure 3')
    trace_audit = read(PAPER / 'evidence/original/trace_figure_audit.json')
    for item in trace_audit['series']:
        trace = next(t['trace'] for t in raw['traces'] if t['scenario'] == trace_audit['scenario'] and t['method'] == item['method'])
        values = [p[item['metric']] for p in trace if .8 <= p['time_s'] <= 2.5]
        require(len(values) == item['displayed_points'], 'Figure 2 point count')
        close(min(values), item['min'], 'Figure 2 minimum')
        close(max(values), item['max'], 'Figure 2 maximum')

    supplemental = read(DATA / 'matched_covariance_20261027/results.json')
    require(len(supplemental['rows']) == 40, 'Supplementary size changed')
    for row in supplemental['rows']:
        require(row['failure'] is None and row['completed_steps'] == 25000, 'Incomplete supplementary case')
        original = next(r for r in raw['rows'] if r['seed'] == row['seed'] and r['scenario'] == row['scenario'] and r['method'] == 'confidence_feedback')
        for key in ('actual_initial_parameters', 'load_start_s', 'load_level_nm'):
            require(row[key] == original[key], 'Supplementary exogenous conditions changed')

    audit = read(PAPER / 'evidence/original/mechanism_final_audit.json')
    docx = next((PAPER / 'article').glob('*.docx'))
    require(sha(docx) == audit['output_sha256'], 'Final reviewed Word changed')
    with ZipFile(docx) as document:
        root = ET.fromstring(document.read('word/document.xml'))
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
              'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
        require(len(root.findall('.//w:tbl', ns)) == 0, 'Unexpected numeric tables')
        require(len(root.findall('.//wp:inline', ns)) == 3, 'Expected three figures')
        embedded = {hashlib.sha256(document.read(name)).hexdigest() for name in document.namelist() if name.startswith('word/media/')}
        for figure in (PAPER / 'figures').glob('*.jpg'):
            require(sha(figure) in embedded, f'Embedded figure mismatch: {figure.name}')
    pdf = next((PAPER / 'article').glob('*.pdf'))
    require(pdf.read_bytes().startswith(b'%PDF-'), 'Invalid PDF')
    stages = {p.parent.name: len(read(p)['rows']) for p in sorted(DATA.glob('*/results.json'))}
    require(sum(stages.values()) == 1000, 'Expected 960 original and 40 supplementary cases')
    return dict(status='PASS', manifest_files_verified=len(inventory['files']),
                frozen_source_hashes_verified=len(mapping), confirmation_cases=320,
                total_archived_cases=sum(stages.values()), stages=stages,
                figure_series_verified=6, figure_means_verified=8,
                primary_statistics_recomputed=True, final_article_unchanged=True,
                raw_sha256=raw_sha, article_sha256=sha(docx), pdf_sha256=sha(pdf), effects=effects)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Optional new verification report outside the historical datasets')
    args = parser.parse_args()
    result = verify()
    output = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        destination = args.report.resolve()
        require(not destination.is_relative_to(DATA), 'Do not overwrite historical evidence')
        with destination.open('x', encoding='utf-8') as handle:
            handle.write(output)
    print(output)


if __name__ == '__main__':
    main()
