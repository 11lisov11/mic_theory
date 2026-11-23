"""A focused research article explaining the prespecified drive-control result."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from PIL import Image
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Rect, Line, PolyLine, Polygon, String
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import rewrite_article as base

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
WORK = HERE / 'reader_revision_20261029'
ASSETS = WORK / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)
RAW_PATH = Path(r'C:\mic_theory\experiments\parameter_adaptive_speed\confirmation_20261006\results.json')
RAW_SHA = '92f344e804b47553f8aa9cfe741c2b31501c4fabc36189144692a9c7f7041d5a'
assert hashlib.sha256(RAW_PATH.read_bytes()).hexdigest() == RAW_SHA
RAW = json.loads(RAW_PATH.read_text(encoding='utf-8'))
PRIMARY = json.loads((RAW_PATH.parent / 'analysis.json').read_text(encoding='utf-8'))['primary']
OLD = HERE / 'before_reader_revision_20261029.docx'
PUBLIC_OLD = Path(r'C:\Study\Статьи и конференции\ТЭБТРАНС2026\Марикин_Параметрическая_адаптация_ТЭБТРАНС_2026.docx')
OUT = base.layout.ROOT / 'Марикин_Повышение_точности_ТЭБТРАНС_2026.docx'
TITLE = 'ПОВЫШЕНИЕ ТОЧНОСТИ РЕГУЛИРОВАНИЯ СКОРОСТИ АСИНХРОННОГО ЭЛЕКТРОПРИВОДА ПРИ ИЗМЕНЕНИИ ПАРАМЕТРОВ ДВИГАТЕЛЯ'
TITLE_EN = 'Improving Induction Motor Drive Speed Regulation under Motor Parameter Variations'
base.OUT = OUT
base.ASSETS = ASSETS
base.layout.TITLE = TITLE
base.layout.TITLE_EN = TITLE_EN
FONT = base.FONT
BOLD = 'Article Times Bold'
pdfmetrics.registerFont(TTFont(BOLD, r'C:\Windows\Fonts\timesbd.ttf'))

ABSTRACT = {
    'Цель:': 'Повысить точность регулирования скорости бездатчикового асинхронного электропривода при изменении параметров двигателя с учётом токовой нагрузки.',
    'Методы:': 'На математической модели сопоставлены система с постоянными расчётными параметрами и система с адаптацией наблюдателя и векторного регулятора. Исследованы изменение параметров двигателя и его совместное действие с помехами измерений.',
    'Результаты:': 'В исследованных режимах средняя абсолютная погрешность скорости снижена на 66–68%, стандартное отклонение электромагнитного момента — на 32–34%. Средние максимальные значения тока изменились менее чем на 1%. Полученное повышение качества регулирования сопровождается увеличением энергии потерь в меди статора примерно на 3%.',
    'Практическая значимость:': 'Результаты показывают возможность повышения точности поддержания скорости при параметрическом рассогласовании с сохранением среднего максимального тока вблизи исходного уровня и могут использоваться при разработке вспомогательных электроприводов транспорта.',
}
KEYWORDS = 'асинхронный электропривод; векторное управление; параметрическая адаптация; наблюдатель состояния; регулирование скорости; математическое моделирование.'

INTRO = [
    'Вспомогательные асинхронные электроприводы обеспечивают работу вентиляционного, насосного и компрессорного оборудования транспорта. Поддержание скорости определяет режим работы механизма, а максимальный ток характеризует электрическую нагрузку двигателя и преобразователя. Поэтому повышение точности регулирования должно оцениваться совместно с изменением токов.',
    'В процессе работы сопротивления обмоток изменяются при нагреве, а принятые в системе управления индуктивности могут отличаться от действительных. Векторный регулятор продолжает рассчитывать напряжение по исходной модели, что вызывает погрешности оценивания потокосцепления и формирования управляющих воздействий. При изменении нагрузки это рассогласование проявляется в отклонении скорости от задания.',
    'Работа продолжает исследования авторов векторного управления и нейроуправления асинхронным двигателем [1, 2], а также программно-аппаратной реализации и интеллектуализации электропривода [3, 4]. Оценивание сопротивлений и намагничивающей индуктивности расширенным фильтром Калмана рассмотрено в [5, 6]. Влияние настройки наблюдателя на качество оценивания исследовано в [7]. Для задачи регулирования необходимо определить, какой выигрыш даёт использование параметрических оценок в замкнутой системе и сопровождается ли он увеличением тока.',
    'Цель исследования — повысить точность поддержания скорости при изменении параметров двигателя посредством адаптации наблюдателя и векторного регулятора. Полученный научный результат — снижение погрешности скорости примерно втрое при сохранении средних максимальных токов вблизи базового уровня. Этот эффект исследован на математической модели при одинаковых заданиях и внешних воздействиях, включая ухудшение измерительного канала.',
]

BEFORE_SCHEME = [
    'На рисунке 1 показано, каким образом параметрическая адаптация включена в систему управления. По токам статора и восстановленному напряжению наблюдатель оценивает скорость, потокосцепление, момент нагрузки и электрические параметры двигателя. Оценка скорости замыкает контур регулирования, оценка нагрузки вводится в задание момента для компенсации возмущения, а оценки параметров поступают в блок корректировки расчётной модели регулятора.',
]

AFTER_SCHEME = [
    'В базовой системе наблюдатель рассчитывает токи и потокосцепление при постоянных параметрах двигателя. В адаптивной системе расширенный фильтр Калмана использует разность между измеренными и предсказанными токами для коррекции оценок скорости, потокосцепления и параметров R<sub>s</sub>, R<sub>r</sub>, L<sub>m</sub>. Регулятор скорости получает скорректированную оценку скорости, поэтому уменьшение ошибки наблюдения непосредственно влияет на точность поддержания действительной скорости.',
    'В регуляторе R<sub>s</sub> участвует в компенсации падения напряжения на сопротивлении статора; R<sub>r</sub> определяет постоянную времени роторной цепи и расчёт скольжения; L<sub>m</sub> используется при формировании намагничивающего тока и преобразовании задания момента в задание тока. Обновление этих параметров изменяет расчёт управляющего напряжения и распределение составляющих тока. Коэффициенты ПИ-регулятора скорости и ограничения тока и момента в сравниваемых системах сохранены.',
    'Для сохранения положительности параметров используются логарифмические переменные. Обновление расчётных параметров регулятора имеет вид',
]
EQUATIONS = [
    'a<sub>j,рег</sub> = a<sub>j0</sub> exp(w<sub>j</sub>θ<sub>j</sub>),     θ<sub>j</sub> = ln(â<sub>j</sub>/a<sub>j0</sub>);     (1)',
    'w<sub>j</sub> = min{1; max[0; 1 − P<sub>θjθj</sub>/P<sub>θjθj</sub><sup>0</sup>]}.     (2)',
]
METHODS = [
    'Здесь a<sub>j0</sub> — исходное значение одного из параметров R<sub>s</sub>, R<sub>r</sub>, L<sub>m</sub>; â<sub>j</sub> — его оценка; P<sub>θjθj</sub> и P<sub>θjθj</sub><sup>0</sup> — текущая и начальная дисперсии логарифмической оценки. Вес w<sub>j</sub> определяет степень использования оценки в регуляторе и увеличивается при уменьшении расчётной неопределённости. При обновлении L<sub>m</sub> индуктивности рассеяния расчётной модели сохраняются.',
    'Математическая модель составлена в неподвижных осях αβ и объединяет электрические процессы двигателя, уравнение движения и модель усреднённого напряжения преобразователя с учётом падения напряжения. Принята линейная магнитная цепь. Исходные параметры расчётного двигателя: R<sub>s</sub> = 7,87 Ом, R<sub>r</sub> = 20,83 Ом, L<sub>m</sub> = 0,4824 Гн; индуктивности рассеяния статора и ротора — 2,49 и 1,94 мГн соответственно. Момент инерции равен 7,65·10<sup>−5</sup> кг·м².',
    'Сопоставлены два режима. В первом намагничивающая индуктивность уменьшена на 15%, индуктивности рассеяния увеличены на 20%, а сопротивление статора в процессе расчёта возрастает на 30%. Во втором к тем же изменениям добавлены задержка, квантование, пропуски и импульсные помехи измерений тока. Это позволяет проверить, сохраняется ли эффект адаптации при ухудшении измерительного канала.',
    'После разгона задание скорости составляет 150 рад/с; затем прикладывается ступень момента нагрузки. Для каждого режима выполнено 20 реализаций с разбросом начальных параметров и момента нагрузки. В каждой реализации базовая и адаптивная системы получают одинаковые внешние воздействия. Продолжительность расчёта — 2,5 с, шаг интегрирования и управления — 100 мкс. Настройки выбраны до контрольной серии; для обеих систем сохранены одинаковые ограничения тока и момента.',
    'Точность оценивалась по среднему модулю отклонения действительной скорости от задания после 0,8 с. Для оценки токовой нагрузки максимум модуля тока определялся в каждой реализации, после чего эти максимумы усреднялись. Дополнительно рассчитаны стандартное отклонение момента после 1,7 с и энергия потерь в меди статора за всё время расчёта. Статистическая проверка выполнялась по парным реализациям; снижение погрешности подтверждено в обоих исследованных режимах.',
]

BEFORE_TRANSIENT = [
    'Рисунок 2 позволяет проследить действие адаптации после приложения нагрузки при совместных параметрических и измерительных возмущениях. Показана одна и та же реализация для двух систем; вертикальная линия отмечает момент изменения нагрузки. Сопоставление скорости, тока и момента необходимо, чтобы оценить улучшение регулирования вместе с электрической реакцией привода.',
]
AFTER_TRANSIENT = [
    'Видно, что при постоянных расчётных параметрах отклонение скорости после приложения нагрузки выражено сильнее, а после переходного процесса скорость остаётся дальше от задания. Адаптивная система уменьшает это отклонение и поддерживает скорость ближе к заданной. На графике момента также видны менее выраженные колебания на заключительном участке. Токовые кривые двух систем сохраняют сопоставимый уровень.',
    'Для оценки эффекта по всей серии на рисунке 3 сопоставлены средняя абсолютная погрешность скорости и средний максимум тока. В каждом режиме показаны результаты 20 реализаций. Диаграммы позволяют сопоставить выигрыш в точности с изменением токовой нагрузки.',
]
RESULTS = [
    'Как видно из рисунка 3, средняя абсолютная погрешность скорости уменьшается примерно втрое. Снижение составляет 66–68% относительно базовой системы и сохраняется при добавлении измерительных помех. В расчётах одновременно уменьшилась ошибка оценки скорости наблюдателем. Это объясняет полученный эффект: регулятор замыкается по оценённой скорости, и при ошибке наблюдения действительная скорость может отклоняться от задания даже при малой ошибке внутри самого контура.',
    'Средние максимальные значения тока изменяются менее чем на 1%. Дополнительное сравнение с адаптацией только наблюдателя показало, что передача параметрических оценок в регулятор уменьшает токовые максимумы, сохраняя выигрыш в точности относительно базовой системы. Стандартное отклонение электромагнитного момента при этом уменьшается на 32–34% относительно базы. Таким образом, адаптация воздействует на оценку скорости в обратной связи и на формирование тока и напряжения. Контроль при одинаковой настройке модели нагрузки также сохранил преимущество по погрешности скорости.',
    'Повышение точности сопровождается увеличением энергии потерь в меди статора примерно на 3%. Максимум тока и энергия потерь характеризуют разные стороны процесса: последняя зависит от интеграла квадрата тока за весь расчётный интервал. Поэтому небольшой прирост потерь следует учитывать при оценке полученного улучшения качества регулирования.',
    'Полученный результат относится к исследованным режимам стабилизации скорости при параметрическом рассогласовании. Он даёт основание рассматривать такую адаптацию при разработке вспомогательного электропривода с изменяющимися параметрами двигателя. Для переноса результата на конкретный транспортный механизм необходима проверка с его нагрузочной характеристикой и параметрами реальной машины.',
]
CONCLUSIONS = [
    '1. На математической модели реализована параметрическая адаптация наблюдателя и векторного регулятора асинхронного электропривода. Оценки сопротивлений и намагничивающей индуктивности используются для корректировки расчётных связей системы управления.',
    '2. В двух исследованных режимах с изменением параметров двигателя средняя абсолютная погрешность скорости снижена на 66–68% относительно базовой системы. Добавление измерительных помех сохраняет полученный выигрыш; средние максимальные значения тока изменяются менее чем на 1%.',
    '3. Стандартное отклонение электромагнитного момента уменьшено на 32–34%. Улучшение точности и сглаживание момента сопровождаются увеличением энергии потерь в меди статора примерно на 3%, что определяет соотношение достигнутого качества регулирования и электрических потерь.',
]
EN_ABSTRACT = [
    ('Objective: ', 'To improve sensorless induction motor drive speed regulation under motor parameter variations while accounting for current loading. '),
    ('Methods: ', 'A mathematical model was used to compare fixed-parameter control with coordinated observer and vector-controller adaptation. Parameter disturbances were examined both separately and together with current measurement disturbances. '),
    ('Results: ', 'In the investigated cases, mean absolute speed error decreased by 66–68% and electromagnetic torque standard deviation by 32–34%. Mean individual current maxima changed by less than 1%. The improved regulation was accompanied by an approximately 3% increase in stator copper-loss energy. '),
    ('Practical significance: ', 'The results demonstrate improved speed regulation under parameter mismatch with mean current maxima remaining close to baseline values. They can inform the development of auxiliary transport drives, subject to validation with the parameters and load characteristic of a specific machine.'),
]


def txt(d, x, y, value, size=12, anchor='middle', bold=False, color=colors.black):
    d.add(String(x, y, value, fontName=BOLD if bold else FONT, fontSize=size, textAnchor=anchor, fillColor=color))


def multiline(d, x, y, lines, size=13, gap=None):
    gap = gap or size*1.15
    for i, line in enumerate(lines):
        txt(d, x, y-i*gap, line, size)


def arrow(d, points, width=.8, color=colors.black):
    d.add(PolyLine(points, strokeColor=color, strokeWidth=width, fillColor=None))
    a, b = points[-2:]
    angle = math.atan2(b[1]-a[1], b[0]-a[0])
    length, wing = 5, 2.2
    pts = [b[0], b[1], b[0]-length*math.cos(angle)+wing*math.sin(angle), b[1]-length*math.sin(angle)-wing*math.cos(angle), b[0]-length*math.cos(angle)-wing*math.sin(angle), b[1]-length*math.sin(angle)+wing*math.cos(angle)]
    d.add(Polygon(pts, fillColor=color, strokeColor=color, strokeWidth=.3))


def box(d, x, y, w, h, lines, size=13):
    d.add(Rect(x, y, w, h, fillColor=colors.white, strokeColor=colors.black, strokeWidth=.8))
    top = y+h/2+(len(lines)-1)*size*.58-size*.32
    multiline(d, x+w/2, top, lines, size)


def save_svg(d, n):
    (ASSETS / f'figure_{n}.svg').write_text(renderSVG.drawToString(d), encoding='utf-8')


def draw_scheme():
    d = Drawing(600, 260)
    box(d, 75, 150, 145, 55, ['Векторный', 'регулятор'])
    box(d, 265, 150, 130, 55, ['Преобразователь', 'напряжения'])
    box(d, 440, 150, 140, 55, ['Асинхронный', 'двигатель'])
    box(d, 335, 25, 245, 65, ['Наблюдатель состояния', 'и параметров'])
    box(d, 75, 25, 165, 65, ['Корректировка', 'параметров'])
    arrow(d, [(10, 177), (75, 177)])
    txt(d, 40, 188, 'ω*', 15)
    arrow(d, [(220, 177), (265, 177)])
    txt(d, 242, 188, 'u*', 13)
    arrow(d, [(395, 177), (440, 177)])
    txt(d, 418, 188, 'u', 13)
    arrow(d, [(580, 177), (598, 177)])
    txt(d, 590, 190, 'ω', 14)
    arrow(d, [(510, 232), (510, 205)])
    txt(d, 510, 242, 'Момент нагрузки', 12)
    arrow(d, [(510, 150), (510, 90)])
    txt(d, 539, 115, 'iα, iβ', 13)
    arrow(d, [(330, 150), (330, 115), (420, 115), (420, 90)])
    txt(d, 375, 122, 'uα, uβ', 12)
    arrow(d, [(335, 48), (240, 48)])
    multiline(d, 286, 69, ['Оценки', 'параметров'], 11)
    arrow(d, [(157, 90), (157, 150)])
    multiline(d, 110, 121, ['Расчётные', 'параметры'], 11)
    arrow(d, [(335, 80), (295, 80), (295, 110), (205, 110), (205, 150)])
    multiline(d, 260, 137, ['Оценки скорости', 'и нагрузки'], 11)
    save_svg(d, 1)


def num(x, digits=None):
    if digits is None:
        value = f'{x:g}'
    else:
        value = f'{x:.{digits}f}'
    return value.replace('.', ',').replace('-', '−')


def axes(d, x, y, w, h, xmin, xmax, ymin, ymax, xticks, yticks, title, show_x_labels=True):
    tx = lambda t: x+(t-xmin)/(xmax-xmin)*w
    ty = lambda v: y+(v-ymin)/(ymax-ymin)*h
    for v in yticks:
        d.add(Line(x, ty(v), x+w, ty(v), strokeColor=colors.HexColor('#dddddd'), strokeWidth=.4))
        txt(d, x-8, ty(v)-4, num(v), 11, 'end')
    d.add(Line(x, y, x+w, y, strokeColor=colors.black, strokeWidth=.8))
    d.add(Line(x, y, x, y+h, strokeColor=colors.black, strokeWidth=.8))
    for t in xticks:
        d.add(Line(tx(t), y, tx(t), y-3, strokeColor=colors.black, strokeWidth=.6))
        if show_x_labels:
            txt(d, tx(t), y-15, num(t, 1), 11)
    txt(d, x, y+h+9, title, 12.5, 'start')
    return tx, ty


def draw_transient():
    d = Drawing(600, 425)
    traces = {r['method']: r['trace'] for r in RAW['traces'] if r['scenario'] == 'composite'}
    choices = [('fixed_reference', 'Базовая система', colors.HexColor('#777777'), None), ('confidence_feedback', 'Адаптивная система', colors.black, [5, 2.5])]
    for i, (_, label, color, dash) in enumerate(choices):
        x = 55+i*280
        d.add(PolyLine([(x, 411), (x+28, 411)], strokeColor=color, strokeWidth=1.2, strokeDashArray=dash))
        txt(d, x+35, 407, label, 12, 'start')
    load = next(r['load_start_s'] for r in RAW['rows'] if r['seed'] == 97200 and r['scenario'] == 'composite' and r['method'] == 'confidence_feedback')
    audit = []
    settings = [
        ('speed_true', 292, 120, 160, [120, 140, 160], 'а  Угловая скорость, рад/с'),
        ('current_a', 177, .6, 1.4, [.6, 1., 1.4], 'б  Модуль вектора тока, А'),
        ('torque_nm', 62, -.03, .45, [0., .2, .4], 'в  Электромагнитный момент, Н·м'),
    ]
    for key, y, lo, hi, ticks, title in settings:
        tx, ty = axes(d, 65, y, 505, 85, .8, 2.5, lo, hi, [.8, 1.2, 1.6, 2., 2.4], ticks, title, show_x_labels=key=='torque_nm')
        d.add(Line(tx(load), y, tx(load), y+85, strokeColor=colors.HexColor('#999999'), strokeDashArray=[2, 3], strokeWidth=.6))
        for method, _, color, dash in choices:
            points = [(r['time_s'], r[key]) for r in traces[method] if .8 <= r['time_s'] <= 2.5]
            assert all(lo <= value <= hi for _, value in points), (key, method)
            d.add(PolyLine([(tx(t), ty(v)) for t, v in points], strokeColor=color, strokeWidth=1.1, strokeDashArray=dash, fillColor=None))
            audit.append(dict(metric=key, method=method, displayed_points=len(points), min=min(v for _, v in points), max=max(v for _, v in points)))
        if key == 'speed_true':
            d.add(Line(tx(.8), ty(150), tx(2.5), ty(150), strokeColor=colors.HexColor('#aaaaaa'), strokeDashArray=[1, 3], strokeWidth=.7))
            txt(d, tx(load)+5, y+73, 'Нагрузка', 10.5, 'start')
            txt(d, 570, y+94, 'ω* = 150 рад/с', 11, 'end')
    txt(d, 570, 20, 'Время, с', 12, 'end')
    save_svg(d, 2)
    (WORK / 'trace_figure_audit.json').write_text(json.dumps(dict(raw_sha256=RAW_SHA, seed=97200, scenario='composite', display_step_s=.005, metric_step_s=.0001, series=audit), ensure_ascii=False, indent=2), encoding='utf-8')


def means(case, method, key):
    rows = [r for r in RAW['rows'] if r['scenario'] == case and r['method'] == method]
    assert len(rows) == 20
    return statistics.mean(r[key] for r in rows)


def draw_comparison():
    d = Drawing(600, 255)
    for x, label, fill in [(105, 'Базовая система', colors.HexColor('#bdbdbd')), (335, 'Адаптивная система', colors.black)]:
        d.add(Rect(x, 238, 17, 8, fillColor=fill, strokeColor=colors.black, strokeWidth=.6))
        txt(d, x+24, 236, label, 12, 'start')
    audit = []
    for panel, (key, maximum, ticks, title) in enumerate([
        ('tracking_mae_rad_s', 5., [0, 1, 2, 3, 4, 5], 'а  Погрешность скорости, рад/с'),
        ('peak_current_a', 1.6, [0, .4, .8, 1.2, 1.6], 'б  Средний максимум тока, А'),
    ]):
        x, y, w, h = 52+panel*300, 65, 220, 138
        ty = lambda value: y+value/maximum*h
        for v in ticks:
            d.add(Line(x, ty(v), x+w, ty(v), strokeColor=colors.HexColor('#dddddd'), strokeWidth=.4))
            txt(d, x-7, ty(v)-4, num(v), 11, 'end')
        d.add(Line(x, y, x+w, y, strokeColor=colors.black, strokeWidth=.8))
        d.add(Line(x, y, x, y+h, strokeColor=colors.black, strokeWidth=.8))
        txt(d, x, 214, title, 12.5, 'start')
        for group, (case, lines) in enumerate([('parameter_drift', ['Изменение', 'параметров']), ('composite', ['С помехами', 'измерений'])]):
            cx = x+55+group*110
            for j, method in enumerate(['fixed_reference', 'confidence_feedback']):
                value = means(case, method, key)
                bx = cx-35+j*42
                d.add(Rect(bx, y, 28, value/maximum*h, fillColor=colors.HexColor('#bdbdbd') if j == 0 else colors.black, strokeColor=colors.black, strokeWidth=.6))
                txt(d, bx+14, ty(value)+7, num(value, 2), 11)
                audit.append(dict(metric=key, scenario=case, method=method, mean=value))
            multiline(d, cx, 45, lines, 11.5, 13)
    save_svg(d, 3)
    (WORK / 'comparison_figure_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')


def equation(doc, text):
    p = base.rich(doc, text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)


def endmatter(doc):
    n, a, proto = base.layout.new_p, base.layout.add, base.layout.prototype
    p = n(doc, 45, 'Библиографический список', keep=True)
    p.paragraph_format.space_before = Pt(12)
    for i, text in enumerate(base.layout.RU_REFS):
        p = n(doc, 46 if i == 0 else 47, text)
        p.paragraph_format.keep_together = True
    n(doc, 58, 'Контактная информация', keep=True)
    n(doc, 59, 'МАРИКИН Александр Николаевич — доктор техн. наук, профессор', keep=True)
    n(doc, 61, 'ЖЕМЧУГОВ Валерий Григорьевич — канд. техн. наук, доцент, zhem@mail.ru', keep=True)
    p = n(doc, 62, 'ОНОФРИЙЧУК Андрей Дмитриевич — аспирант, 11lisov11@gmail.com')
    for r in p.runs:
        r.font.name = FONT
        r.font.size = Pt(12)
    n(doc, 63, TITLE_EN, keep=True)
    n(doc, 64, 'A. N. Marikin, V. G. Zhemchugov, A. D. Onofriychuk', keep=True)
    n(doc, 65, base.layout.SOURCE.paragraphs[65].text.strip(), keep=True)
    base.layout.short_citation(doc, english=True)
    p = n(doc, 67)
    a(p, 'Summary\n', proto=proto(67, 0), bold=True, size=12)
    for label, text in EN_ABSTRACT:
        a(p, label, proto=proto(67, 2), bold=True, size=12)
        a(p, text, proto=proto(67, 3), bold=False, size=12)
    p = n(doc, 68)
    a(p, 'Keywords: ', proto=proto(68, 0), bold=True, size=12)
    a(p, 'induction motor drive; vector control; parameter adaptation; state observer; speed regulation; mathematical modelling.', proto=proto(68, 1), size=12)
    n(doc, 69, 'References', keep=True)
    old = Document(OLD)
    old_text = [p.text for p in old.paragraphs]
    start = old_text.index('References')+1
    stop = next(i for i in range(start, len(old_text)) if old_text[i].startswith('Author’s information'))
    refs = [text for text in old_text[start:stop] if text.strip()]
    assert len(refs) == 7
    for i, text in enumerate(refs):
        p = n(doc, 70 if i == 0 else 71, text)
        p.paragraph_format.keep_together = True
    p = n(doc, 81)
    a(p, 'Author’s information\n', proto=proto(81, 0), bold=True, size=12)
    a(p, 'MARIKIN Alexander Nikolaevich — Doctor of Technical Sciences, Professor', proto=proto(81, 2), bold=False, size=12)
    n(doc, 83, 'ZHEMCHUGOV Valery Grigoryevich — Candidate of Technical Sciences, Associate Professor')
    n(doc, 84, 'ONOFRIYCHUK Andrey Dmitrievich — Postgraduate student, 11lisov11@gmail.com')


def build():
    if not OLD.exists():
        shutil.copy2(PUBLIC_OLD, OLD)
    for n in [1, 2, 3]:
        p = ASSETS / f'figure_{n}.jpg'
        if b'JFIF' not in p.read_bytes()[:32]:
            im = Image.open(p).convert('RGB')
            im.save(p, 'JPEG', quality=98, subsampling=0, dpi=(400, 400))
    doc = Document(base.layout.REF)
    for c in list(doc._element.body):
        if c.tag != qn('w:sectPr'):
            doc._element.body.remove(c)
    n, a, proto = base.layout.new_p, base.layout.add, base.layout.prototype
    n(doc, 0, 'УДК 681.518:621.313.333')
    n(doc, 1, TITLE, keep=True)
    p = n(doc, 2, keep=True)
    for i, name in enumerate(['А. Н. Марикин', 'В. Г. Жемчугов', 'А. Д. Онофрийчук'], 1):
        if i > 1:
            a(p, ', ', proto=proto(2, 0))
        a(p, name, proto=proto(2, 0))
        a(p, str(i), proto=proto(2, 1))
    n(doc, 3, base.layout.SOURCE.paragraphs[3].text.strip(), keep=True)
    base.layout.short_citation(doc)
    n(doc, 5, 'Аннотация.', keep=True)
    for key, value in ABSTRACT.items():
        base.layout.labelled(doc, key, value)
    base.layout.labelled(doc, 'Ключевые слова:', KEYWORDS, index=9)
    for i, text in enumerate(INTRO):
        base.rich(doc, text, 10 if i == 0 else 11)
    for text in BEFORE_SCHEME:
        p = base.rich(doc, text, keep=True)
        p.paragraph_format.keep_together = True
    base.picture(doc, 1, 'Включение параметрической адаптации в систему векторного управления')
    for text in AFTER_SCHEME:
        base.rich(doc, text)
    for text in EQUATIONS:
        equation(doc, text)
    for text in METHODS:
        base.rich(doc, text)
    for text in BEFORE_TRANSIENT:
        p = base.rich(doc, text, keep=True)
        p.paragraph_format.keep_together = True
    base.picture(doc, 2, 'Реакция на изменение нагрузки при совместных возмущениях; горизонтальная линия на графике а — задание скорости')
    for i, text in enumerate(AFTER_TRANSIENT):
        p = base.rich(doc, text, keep=i==len(AFTER_TRANSIENT)-1)
        if i == len(AFTER_TRANSIENT)-1:
            p.paragraph_format.keep_together = True
    base.picture(doc, 3, 'Сравнение точности регулирования и токовой нагрузки по 20 реализациям каждого режима')
    for text in RESULTS:
        base.rich(doc, text)
    n(doc, 37, 'По результатам исследования сформулированы следующие выводы.', keep=True)
    for text in CONCLUSIONS:
        p = base.rich(doc, text, 38)
        p.paragraph_format.keep_together = True
    endmatter(doc)
    doc.core_properties.title = TITLE.capitalize()
    doc.core_properties.author = 'А. Н. Марикин; В. Г. Жемчугов; А. Д. Онофрийчук'
    doc.core_properties.subject = 'Повышение точности регулирования скорости при параметрическом рассогласовании'
    doc.core_properties.comments = ''
    base.package_save(doc)
    assert hashlib.sha256(RAW_PATH.read_bytes()).hexdigest() == RAW_SHA
    plain = '\n'.join(p.text for p in doc.paragraphs)
    main = plain.split('Библиографический список')[0]
    (WORK / 'manuscript.txt').write_text(plain, encoding='utf-8')
    audit = dict(output=str(OUT), title=TITLE, raw_sha256=RAW_SHA,
                 selected_comparison='prespecified confidence_feedback vs fixed_reference',
                 selected_cases=['parameter_drift', 'composite'], independent_seed_clusters=20,
                 numeric_tables=0, figures=3, conclusions=3,
                 main_text_words=len(main.split()), main_text_numeric_tokens=len(re.findall(r'\d+(?:[.,]\d+)?', main)),
                 no_new_simulations=True, numerical_claims='rounded original confirmation effects',
                 no_claim_of_unique_superiority_over_other_adaptive_designs=True)
    (WORK / 'build_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'figures':
        draw_scheme()
        draw_transient()
        draw_comparison()
        print(ASSETS)
    else:
        build()
