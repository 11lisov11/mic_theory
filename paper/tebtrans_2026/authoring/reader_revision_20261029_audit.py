import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from zipfile import ZipFile

from docx import Document

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
WORK = HERE / 'reader_revision_20261029'
OUT = HERE.parent.parent / 'Марикин_Повышение_точности_ТЭБТРАНС_2026.docx'
RAW_PATH = Path(r'C:\mic_theory\experiments\parameter_adaptive_speed\confirmation_20261006\results.json')
RAW_SHA = '92f344e804b47553f8aa9cfe741c2b31501c4fabc36189144692a9c7f7041d5a'
assert hashlib.sha256(RAW_PATH.read_bytes()).hexdigest() == RAW_SHA
raw = json.loads(RAW_PATH.read_text(encoding='utf-8'))
analysis = json.loads((RAW_PATH.parent / 'analysis.json').read_text(encoding='utf-8'))
assert raw['selection']['primary_candidate'] == 'confidence_feedback'
assert raw['selection']['primary_cases'] == ['parameter_drift', 'composite']
assert analysis['provenance']['raw_sha256'] == RAW_SHA
metrics = ['tracking_mae_rad_s', 'peak_current_a', 'steady_torque_std_nm', 'copper_energy_j']
computed = {}
for case in raw['selection']['primary_cases']:
    computed[case] = {}
    arrays = {}
    for method in ['fixed_reference', 'confidence_feedback']:
        rows = [r for r in raw['rows'] if r['scenario'] == case and r['method'] == method]
        assert len(rows) == 20 and sorted(r['seed'] for r in rows) == raw['seeds']
        assert all(r['failure'] is None and r['completed_steps'] == 25000 for r in rows)
        arrays[method] = rows
        computed[case][method] = {key: statistics.mean(r[key] for r in rows) for key in metrics}
    for seed in raw['seeds']:
        a = next(r for r in arrays['fixed_reference'] if r['seed'] == seed)
        b = next(r for r in arrays['confidence_feedback'] if r['seed'] == seed)
        for key in ['load_start_s', 'load_level_nm', 'actual_initial_parameters', 'torque_limit_nm']:
            assert a[key] == b[key]
    primary = next(r for r in analysis['primary'] if r['scenario'] == case)
    for key in metrics:
        a = computed[case]['confidence_feedback'][key]
        b = computed[case]['fixed_reference'][key]
        ratio = a / b
        assert math.isclose(ratio, primary['metrics'][key]['ratio'], abs_tol=1e-12)
    assert primary['statistical_mae_pass'] is True

speed = [100 * (1-v['confidence_feedback']['tracking_mae_rad_s']/v['fixed_reference']['tracking_mae_rad_s']) for v in computed.values()]
torque = [100 * (1-v['confidence_feedback']['steady_torque_std_nm']/v['fixed_reference']['steady_torque_std_nm']) for v in computed.values()]
current = [100 * (v['confidence_feedback']['peak_current_a']/v['fixed_reference']['peak_current_a']-1) for v in computed.values()]
copper = [100 * (v['confidence_feedback']['copper_energy_j']/v['fixed_reference']['copper_energy_j']-1) for v in computed.values()]
assert sorted(round(x) for x in speed) == [66, 68]
assert sorted(round(x) for x in torque) == [32, 34]
assert max(abs(x) for x in current) < 1
assert [round(x) for x in copper] == [3, 3]

bars = json.loads((WORK / 'comparison_figure_audit.json').read_text(encoding='utf-8'))
assert len(bars) == 8
for bar in bars:
    assert math.isclose(bar['mean'], computed[bar['scenario']][bar['method']][bar['metric']], abs_tol=1e-12)
traces = json.loads((WORK / 'trace_figure_audit.json').read_text(encoding='utf-8'))
assert traces['raw_sha256'] == RAW_SHA and traces['seed'] == raw['seeds'][0]
assert len(traces['series']) == 6
for series in traces['series']:
    saved = next(r['trace'] for r in raw['traces'] if r['scenario'] == 'composite' and r['method'] == series['method'])
    points = [r for r in saved if .8 <= r['time_s'] <= 2.5]
    assert series['displayed_points'] == len(points) == 340
    assert min(r[series['metric']] for r in points) == series['min']
    assert max(r[series['metric']] for r in points) == series['max']
supp = json.loads((HERE / 'matched_covariance_20261027' / 'analysis.json').read_text(encoding='utf-8'))
assert all(supp[case]['confidence_feedback']['tracking_mae_rad_s']['ratio_to_fixed_same_q'] < 1 for case in computed)
mechanism = {}
for case in computed:
    mechanism[case] = {}
    for method in ['fixed_reference', 'joint_only', 'confidence_feedback']:
        rows = [r for r in raw['rows'] if r['scenario'] == case and r['method'] == method]
        assert len(rows) == 20
        mechanism[case][method] = {key: statistics.mean(r[key] for r in rows) for key in ['estimation_mae_rad_s', 'tracking_mae_rad_s', 'peak_current_a']}
    m = mechanism[case]
    assert m['confidence_feedback']['estimation_mae_rad_s'] < m['fixed_reference']['estimation_mae_rad_s']
    assert m['joint_only']['estimation_mae_rad_s'] < m['fixed_reference']['estimation_mae_rad_s']
    assert m['confidence_feedback']['peak_current_a'] < m['joint_only']['peak_current_a']
    assert m['confidence_feedback']['tracking_mae_rad_s'] < m['fixed_reference']['tracking_mae_rad_s']
for path, expected in raw['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected

doc = Document(OUT)
old = Document(HERE / 'before_reader_revision_20261029.docx')
assert len(doc.tables) == 0 and len(doc.inline_shapes) == 3
paragraphs = [p.text.strip() for p in doc.paragraphs]
main = '\n'.join(paragraphs[:paragraphs.index('Библиографический список')])
assert len([p for p in paragraphs[:paragraphs.index('Библиографический список')] if re.match(r'^[123]\. ', p)]) == 3
assert not re.search(r'не установлено|не доказано|не удалось|прогон\w*|холд[ао]ут|фидбек', main, re.I)
for required in ['66–68%', '32–34%', 'менее чем на 1%', 'примерно на 3%']:
    assert required in main, required
assert re.search(r'расширенн\w+ фильтр\w+ Калмана', main)
for required in ['между измеренными и предсказанными токами', 'компенсации падения напряжения', 'постоянную времени роторной цепи', 'формировании намагничивающего тока', 'Коэффициенты ПИ-регулятора скорости', 'Дополнительное сравнение с адаптацией только наблюдателя']:
    assert required in main, required
assert not re.search(r'</?(?:sub|sup)>', main)
assert not re.search(r'впервые|уникальн\w+|новый закон|экспериментально на стенде', main, re.I)
cited = {int(x) for group in re.findall(r'\[([1-7](?:,\s*[1-7])*)\]', main) for x in re.findall(r'\d+', group)}
assert cited == set(range(1, 8))
for i in [1, 2, 3]:
    assert any(p.startswith(f'Рисунок {i} –') for p in paragraphs)
    assert any(f'рисунке {i}' in p.lower() or f'Рисунок {i} ' in p for p in paragraphs if not p.startswith(f'Рисунок {i} –'))

def bibliography(document, heading, after):
    text = [p.text.strip() for p in document.paragraphs]
    start = text.index(heading)+1
    end = next(i for i in range(start, len(text)) if text[i].startswith(after))
    result = [p for p in text[start:end] if re.match(r'^\d+\.', p)]
    assert len(result) == 7
    return result

for heading, after in [('Библиографический список', 'Контактная информация'), ('References', 'Author’s information')]:
    assert bibliography(doc, heading, after) == bibliography(old, heading, after)
reference = next(Path(r'C:\Study\Статьи и конференции\ВАК').glob('Марикин*.docx'))
assert doc.sections[0]._sectPr.xml == Document(reference).sections[0]._sectPr.xml
with ZipFile(OUT) as output, ZipFile(reference) as source:
    for part in ['word/styles.xml', 'word/footer1.xml', 'word/theme/theme1.xml', 'word/fontTable.xml']:
        assert output.read(part) == source.read(part)
    for i, picture in enumerate(doc.inline_shapes, 1):
        relid = picture._inline.xpath('.//a:blip/@r:embed')[0]
        part = doc.part.related_parts[relid]
        assert part.blob == (WORK / 'assets' / f'figure_{i}.jpg').read_bytes()

def body_text_with_tables(document):
    result = []
    for el in document._element.body:
        t = ''.join(el.xpath('.//w:t/text()'))
        if t == 'Библиографический список':
            break
        result.append(t)
    return '\n'.join(result)

old_numbers = len(re.findall(r'\d+(?:[.,]\d+)?', body_text_with_tables(old)))
new_numbers = len(re.findall(r'\d+(?:[.,]\d+)?', body_text_with_tables(doc)))
assert new_numbers < old_numbers
pages = list((WORK / 'render_mechanism_v2').glob('page-*.png'))
assert len(pages) == 8
report = dict(output=str(OUT), output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
              raw_sha256=RAW_SHA, prespecified_primary_comparison_preserved=True,
              mean_values_recomputed=16, graph_mean_values_verified=8,
              trace_series_verified=6, paired_seed_clusters=20,
              effects_percent=dict(speed_error_reduction=speed, torque_std_reduction=torque,
                                  mean_individual_current_maximum_change=current, stator_copper_energy_change=copper),
              same_covariance_supplementary_check_supports_statement=True,
              mechanism_evidence=mechanism, original_simulation_sources_match=True,
              numeric_tokens_before=old_numbers, numeric_tokens_after=new_numbers,
              tables_before=3, tables_after=0, figures=3, numbered_conclusions=3,
              bibliography_preserved=True, source_format_preserved=True,
              final_pages_visually_reviewed=list(range(1, 9)), new_simulations=False)
(WORK / 'mechanism_final_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
