from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from lxml import etree
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image

import build_vak as layout

sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent
ASSETS=HERE/'narrative_assets'
ASSETS.mkdir(exist_ok=True)
OUT=layout.ROOT/'Марикин_Повышение_точности_ТЭБТРАНС_2026.docx'
TITLE='ПОВЫШЕНИЕ ТОЧНОСТИ РЕГУЛИРОВАНИЯ АСИНХРОННОГО ЭЛЕКТРОПРИВОДА ПРИ ИЗМЕНЕНИИ ПАРАМЕТРОВ ДВИГАТЕЛЯ'
TITLE_EN='Improving the Speed Regulation Accuracy of an Induction Motor Drive under Motor Parameter Variations'
layout.TITLE=TITLE
layout.TITLE_EN=TITLE_EN
METRICS=json.loads((layout.WORK/'verified_metrics.json').read_text(encoding='utf-8'))
FONT='Times New Roman'
pdfmetrics.registerFont(TTFont(FONT,r'C:\Windows\Fonts\times.ttf'))
BLACK=colors.black
WHITE=colors.white

ABSTRACT={
    'Цель:':'Исследовать возможность повышения точности регулирования скорости бездатчикового асинхронного электропривода при изменении параметров двигателя посредством согласованной адаптации наблюдателя и векторного регулятора.',
    'Методы:':'Реализована система оценивания состояний и параметров двигателя на основе расширенного фильтра Калмана. Проведено сравнительное моделирование при постоянных параметрах системы управления, адаптации только наблюдателя и использовании параметрических оценок в регуляторе. Сопоставлены погрешность скорости, максимальные значения тока, колебания момента и потери в обмотке статора.',
    'Результаты:':'При изменении параметров двигателя и совместном действии возмущений согласованная адаптация уменьшила среднюю абсолютную погрешность скорости на 66,18 и 67,52% соответственно. Средние максимальные значения тока изменились на −0,15 и +0,84%, стандартное отклонение момента уменьшилось на 34,34 и 31,83%. Энергия потерь в меди обмотки статора увеличилась на 2,99 и 2,63%.',
    'Практическая значимость:':'Результаты определяют условия применения параметрической адаптации во вспомогательных электроприводах транспорта, для которых требуется поддержание скорости при изменяющихся параметрах двигателя с учётом токовой нагрузки. Практическое применение требует стендовой проверки.',
}
KEYWORDS='асинхронный электропривод; векторное управление; наблюдатель состояния; параметрическая адаптация; регулирование скорости; токовая нагрузка; вспомогательное оборудование транспорта.'

INTRO=[
    'Вспомогательные электроприводы транспортного оборудования обеспечивают работу механизмов вентиляции, перекачивания жидкостей и подачи сжатого воздуха. Для таких механизмов требуется поддержание заданной частоты вращения при изменении нагрузки. Отклонение скорости изменяет рабочий режим механизма, а увеличение тока повышает нагрузку на электрическую машину и преобразователь. Поэтому при разработке системы управления необходимо совместно оценивать точность регулирования и токовую нагрузку. В данной работе рассматривается один из факторов ухудшения регулирования — изменение электрических параметров асинхронного двигателя.',
    'В предыдущих работах авторов исследованы управление с оптронным датчиком [1], нейроуправление с обучением в цифровом двойнике [2], программно-аппаратная реализация управления с учётом потокосцепления ротора [3] и применение искусственного интеллекта в асинхронном приводе [4]. Дальнейшее развитие этих исследований связано с сохранением качества управления при изменении характеристик двигателя. Для решения этой задачи в настоящей работе используется оценивание текущих параметров машины с последующей корректировкой расчётных параметров системы управления.',
    'Векторный регулятор рассчитывает управляющие воздействия по принятой модели двигателя. При изменении сопротивлений и индуктивностей её параметры перестают соответствовать объекту. Например, сопротивление ротора R<sub>r</sub> и индуктивность L<sub>r</sub> определяют постоянную времени роторной цепи, а отношение индуктивностей входит в расчёт электромагнитного момента:',
]
AFTER_EQUATION=[
    'где T<sub>r</sub> — постоянная времени роторной цепи; k<sub>r</sub> — коэффициент, входящий в выражение электромагнитного момента; L<sub>m</sub> — индуктивность намагничивания; L<sub>σr</sub> — индуктивность рассеяния ротора. Изменение этих величин влияет на оценивание потокосцепления, формирование задания тока и компенсацию взаимосвязей в токовом контуре. Следовательно, уточнение параметров должно учитываться при анализе всей замкнутой системы управления.',
    'Совместное оценивание состояний и сопротивлений асинхронной машины расширенным фильтром Калмана рассмотрено в [5]. В работе [6] исследовано также оценивание индуктивности намагничивания, а в [7] показано влияние условий работы и настройки наблюдателя на качество оценивания. Для замкнутой системы необходимо определить влияние использования этих оценок в регуляторе на погрешность скорости и токовую нагрузку.',
    'Цель работы — определить влияние согласованной адаптации наблюдателя и регулятора на точность поддержания скорости асинхронного двигателя при изменении его параметров и действии возмущений. Для этого сопоставлены погрешность скорости, максимальные значения тока, колебания электромагнитного момента и энергия потерь в обмотке статора.',
    'Структура исследуемой системы представлена на рисунке 1. Векторный регулятор формирует задание напряжения для преобразователя. Наблюдатель по токам статора и восстановленным значениям напряжения определяет скорость, потокосцепление ротора и электрические параметры двигателя. Оценки скорости и потокосцепления используются для обратной связи, а оценки параметров — для корректировки расчётных коэффициентов регулятора. Под согласованной адаптацией далее понимается использование результатов параметрического оценивания в обоих звеньях системы управления.',
]
METHODS=[
    'Наблюдатель реализован на основе расширенного фильтра Калмана. Его вектор состояния содержит составляющие тока статора и потокосцепления ротора в осях αβ, скорость, момент нагрузки и логарифмы значений R<sub>s</sub>, R<sub>r</sub>, L<sub>m</sub>, нормированных на номинальные величины. Логарифмическое представление сохраняет положительность оцениваемых параметров. Оценки используются при формировании токовых заданий и расчёте напряжения.',
    'В основном исследуемом варианте логарифмическую оценку каждого параметра вводили в регулятор с весовым коэффициентом от нуля до единицы. Вес определяли по уменьшению соответствующей расчётной дисперсии относительно её начального значения 0,04. Этот приём ограничивает влияние оценок с большой неопределённостью, однако сама величина ковариации не доказывает идентифицируемость параметра. Влияние взвешивания отдельно сопоставлялось с непосредственным использованием параметрических оценок.',
    'Для численного исследования использована модель асинхронного двигателя в неподвижных осях αβ с линейной магнитной цепью и усреднённым напряжением преобразователя. Приняты R<sub>s</sub> = 7,868 Ом, R<sub>r</sub> = 20,832 Ом, L<sub>m</sub> = 0,4824 Гн; индуктивности рассеяния статора и ротора — 2,491 и 1,935 мГн; момент инерции — 7,646·10<sup>−5</sup> кг·м². Это расчётные значения, не полученные идентификацией конкретной машины. Дифференциальные уравнения интегрировали методом Рунге — Кутты четвёртого порядка с шагом 100 мкс; период управления также составлял 100 мкс.',
    'Чтобы установить роль параметрической адаптации, выделены три способа построения системы: базовая система с постоянными электрическими параметрами; адаптация только наблюдателя при неизменных параметрах регулятора; согласованная адаптация наблюдателя и регулятора. Дополнительно исследовано прямое обновление параметров регулятора без взвешивания. Регуляторы тока и процедуры учёта задержек и подавления выбросов измерений во всех случаях одинаковы. Оценки получают только по доступным токам и напряжению; действительные значения скорости и параметров используют при моделировании объекта и расчёте показателей качества.',
    'Основное сопоставление выполнено для двух режимов. В первом L<sub>m</sub> уменьшали на 15%, индуктивности рассеяния увеличивали на 20%, а R<sub>s</sub> дополнительно увеличивали на 30% за 1,2 с. Во втором к этим изменениям добавляли задержку измерений 0,3 мс, квантование тока с шагом 2 мА, потерю 0,5% отсчётов и импульсные помехи. Эти режимы далее обозначены как изменение параметров и совместное действие возмущений. Для оценки границ применимости рассмотрены также номинальный режим и совместное действие возмущений при заданной скорости 3 рад/с.',
    'В первых трёх режимах заданная скорость достигает 150 рад/с. Момент нагрузки 0,25–0,45 Н·м прикладывали в случайный момент времени 0,95–1,25 с. Начальные значения R<sub>s</sub>, R<sub>r</sub>, момента инерции и коэффициента трения варьировали в пределах ±5%. Для каждого способа управления и режима выполнено по 20 независимых реализаций случайных воздействий, всего 320 численных экспериментов продолжительностью 2,5 с. При сравнении алгоритмов сохраняли одинаковые начальные условия и возмущения. Напряжение звена постоянного тока составляло 325 В, ограничение задания момента — 0,6 Н·м, уставка ограничения тока — 2,8 А.',
    'Настройки определены до начала контрольных расчётов. Для базовой системы ковариационный параметр модели нагрузки выбирали отдельно для каждого режима по предварительному моделированию. При расчётах информация о наличии и типе возмущений в алгоритм управления не передавалась.',
    'Точность регулирования оценивали средней абсолютной погрешностью скорости (MAE) после 0,8 с. Токовую нагрузку характеризовали максимумом модуля вектора тока αβ за каждый эксперимент с последующим усреднением по 20 реализациям. Колебания момента оценивали его стандартным отклонением после 1,7 с. Энергию потерь в меди обмотки статора определяли за 2,5 с интегрированием мгновенных потерь (3/2)R<sub>s</sub>|i<sub>s</sub>|<sup>2</sup> при амплитудно-инвариантном преобразовании Кларка. Эти показатели позволяют установить, какой токовой и энергетической нагрузкой сопровождается повышение точности регулирования.',
    'Основным заранее выбранным сравнением являлось сопоставление согласованной адаптации с базовой системой в двух режимах изменения параметров. Доверительные интервалы уровня 97,5% определены методом парных повторных выборок с возвращением (10 000 выборок) с поправкой Бонферрони для двух гипотез о погрешности скорости. Сопоставления с адаптацией только наблюдателя и с прямым обновлением параметров имели дополнительный характер.',
]
RESULTS_BEFORE_FIGURE=[
    'На рисунке 2 показано, как способ использования параметрических оценок влияет на погрешность скорости и максимальный ток. В базовой системе при изменении параметров двигателя средняя абсолютная погрешность достигает 3,721 рад/с, а при совместном действии возмущений — 3,769 рад/с. Согласованная адаптация уменьшает эти значения до 1,258 и 1,224 рад/с соответственно. Следовательно, в рассмотренных режимах погрешность уменьшается примерно втрое — на 66,18 и 67,52%.',
]
RESULTS_BEFORE_TABLE=[
    'Адаптация только наблюдателя даёт ещё меньшую погрешность скорости — 0,713 и 0,772 рад/с. Однако средние максимальные значения тока при этом возрастают с 1,284 до 1,704 А и с 1,280 до 1,772 А, то есть на 32,71 и 38,42%. При согласованной адаптации соответствующие значения составляют 1,282 и 1,291 А и остаются близкими к базовым. Таким образом, использование оценок в регуляторе даёт иной компромисс между точностью и токовой нагрузкой; наименьшая погрешность скорости сама по себе не определяет предпочтительный способ управления.',
    'В таблице 1 приведено сопоставление базовой системы и системы с согласованной адаптацией. Относительные изменения рассчитаны по неокруглённым данным. Для отношения погрешностей скорости получены значения 0,338 и 0,325; верхние границы доверительных интервалов равны 0,473 и 0,428 и меньше единицы. Средние максимальные значения тока изменяются на −0,15 и +0,84%; верхние границы доверительных интервалов их отношений равны 1,020 и 1,027, что ниже заранее принятого в исследовании предельного отношения 1,05. Во всех 320 экспериментах наибольшее значение модуля тока составило 2,324 А при уставке 2,8 А.',
]
DISCUSSION=[
    'Одновременно с повышением точности уменьшаются колебания электромагнитного момента. Его стандартное отклонение в режиме изменения параметров снижается на 34,34%, а при совместном действии возмущений — на 31,83%. Снижение этого показателя характеризует более равномерное формирование момента на исследованном интервале.',
    'Энергия потерь в меди обмотки статора увеличивается на 2,99 и 2,63%. Полученное повышение точности и уменьшение колебаний момента сопровождаются небольшим ростом указанных потерь, который следует учитывать при выборе режима работы. Температурные процессы в данном исследовании не рассчитывались.',
    'При прямом обновлении параметров регулятора без взвешивания получены близкие, несколько меньшие погрешности скорости — 1,230 и 1,181 рад/с. Это указывает на определяющую роль использования параметрических оценок в регуляторе; самостоятельное преимущество ковариационного взвешивания в выполненном исследовании не установлено.',
    'В номинальном режиме MAE при согласованной адаптации равна 1,614 рад/с против 0,879 рад/с у базовой системы; при заданной скорости 3 рад/с она составляет 2,076 рад/с. Следовательно, установленный положительный эффект относится к рассмотренным режимам изменения параметров. Использованная модель имеет линейную магнитную цепь и усреднённое напряжение; модель двигателя и прогноз наблюдателя основаны на одной системе уравнений. Для практического применения требуются идентификация машины, выбор условий включения адаптации и стендовая проверка.',
]
CONCLUSIONS=[
    '1. Реализована система бездатчикового векторного управления асинхронным электроприводом с оцениванием R<sub>s</sub>, R<sub>r</sub>, L<sub>m</sub> и использованием этих оценок в регуляторе. Такая структура позволяет учитывать изменение параметров при восстановлении состояния двигателя и формировании управляющего воздействия.',
    '2. В режимах изменения параметров двигателя и совместного действия возмущений средняя абсолютная погрешность скорости уменьшена с 3,721 до 1,258 рад/с и с 3,769 до 1,224 рад/с соответственно. Снижение погрешности составляет 66,18 и 67,52%.',
    '3. Повышение точности достигнуто при изменении средних максимальных значений тока на −0,15 и +0,84% относительно базовой системы. Для сравнения, адаптация только наблюдателя сопровождается увеличением этих значений на 32,71 и 38,42%.',
    '4. Стандартное отклонение электромагнитного момента уменьшено на 34,34 и 31,83%. Энергия потерь в меди обмотки статора увеличилась на 2,99 и 2,63%.',
]
EN_ABSTRACT=[
    ('Objective: ','To investigate whether coordinated adaptation of the observer and vector controller can improve sensorless induction motor speed regulation under motor parameter variations. '),
    ('Methods: ','An extended Kalman filter was used to estimate motor states and parameters. Systems with fixed parameters, observer-only adaptation and coordinated observer/controller adaptation were compared through numerical modelling. Speed error, maximum current, torque fluctuations and stator winding copper-loss energy were assessed. '),
    ('Results: ','Under parameter variations and combined disturbances, coordinated adaptation reduced the mean absolute speed error by 66.18% and 67.52%, respectively. The mean maximum current changed by −0.15% and +0.84%, while the torque standard deviation decreased by 34.34% and 31.83%. Stator winding copper-loss energy increased by 2.99% and 2.63%. These results apply to the specified disturbed regimes; limitations were identified under nominal conditions and at low speed. '),
    ('Practical significance: ','The findings identify conditions for applying parameter adaptation to auxiliary transport drives that require speed regulation under motor parameter variations with due consideration of current loading. Practical implementation requires experimental validation on a test bench.'),
]


def rich(doc,text,index=11,keep=False):
    p=layout.new_p(doc,index,keep=keep)
    for token in re.split(r'(<sub>.*?</sub>|<sup>.*?</sup>)',text):
        if not token:continue
        sub=token.startswith('<sub>');sup=token.startswith('<sup>')
        r=layout.add(p,re.sub(r'</?(?:sub|sup)>','',token),proto=layout.prototype(index))
        if sub:r.font.subscript=True
        if sup:r.font.superscript=True
    return p


def label(d,x,y,value,size=12,anchor='middle'):
    d.add(String(x,y,value,fontName=FONT,fontSize=size,fillColor=BLACK,textAnchor=anchor))


def block(d,x,y,w,h,lines,size=12):
    d.add(Rect(x,y,w,h,strokeColor=BLACK,fillColor=WHITE,strokeWidth=.8))
    gap=size*1.16
    yy=y+h/2+(len(lines)-1)*gap/2-size*.34
    for line in lines:label(d,x+w/2,yy,line,size);yy-=gap


def arrow(d,points):
    for a,b in zip(points,points[1:]):d.add(Line(*a,*b,strokeColor=BLACK,strokeWidth=.85))
    a,b=points[-2:];angle=math.atan2(b[1]-a[1],b[0]-a[0]);r=4.5
    p1=(b[0]-r*math.cos(angle-.45),b[1]-r*math.sin(angle-.45))
    p2=(b[0]-r*math.cos(angle+.45),b[1]-r*math.sin(angle+.45))
    d.add(Polygon([*b,*p1,*p2],fillColor=BLACK,strokeColor=BLACK,strokeWidth=.5))


def draw_assets():
    d=Drawing(575,285)
    block(d,65,170,120,55,['Векторный','регулятор'])
    block(d,245,170,115,55,['Преобразователь','напряжения'])
    block(d,425,170,120,55,['Асинхронный','двигатель'])
    block(d,325,30,200,70,['Адаптивный наблюдатель','Оценивание скорости,','потокосцепления и параметров'])
    block(d,65,30,155,55,['Корректировка','расчётных параметров'])
    arrow(d,[(5,197.5),(65,197.5)]);label(d,31,209,'ω*',13)
    arrow(d,[(185,197.5),(245,197.5)])
    label(d,215,219,'Задание',10.5);label(d,215,207,'напряжения',10.5)
    arrow(d,[(360,197.5),(425,197.5)]);label(d,392,207,'Напряжение',10.5)
    arrow(d,[(545,197.5),(568,197.5)]);label(d,557,209,'ω',13)
    arrow(d,[(485,250),(485,225)])
    label(d,485,269,'Момент нагрузки',12)
    arrow(d,[(485,170),(485,100)]);label(d,500,137,'iα, iβ',12,anchor='start')
    arrow(d,[(302.5,170),(302.5,115),(365,115),(365,100)])
    label(d,333,128,'uα, uβ',12)
    arrow(d,[(325,65),(220,65)]);label(d,273,77,'Оценки Rs, Rr, Lm',10.5)
    arrow(d,[(140,85),(140,170)])
    label(d,132,121,'Параметры',11,anchor='end');label(d,132,108,'регулятора',11,anchor='end')
    arrow(d,[(325,90),(275,90),(275,140),(175,140),(175,170)])
    label(d,220,156,'Оценки скорости',10.5);label(d,220,144,'и потокосцепления',10.5)
    (ASSETS/'figure_1.svg').write_text(renderSVG.drawToString(d),encoding='utf-8')

    d=Drawing(560,285)
    methods=['fixed_reference','joint_only','confidence_feedback']
    fills=[WHITE,colors.HexColor('#bbbbbb'),colors.HexColor('#202020')]
    legends=[(12,271,'Базовая система'),(294,271,'Адаптация наблюдателя'),(12,252,'Согласованная адаптация')]
    for i,(x,y,name) in enumerate(legends):
        d.add(Rect(x,y-2,10,8,strokeColor=BLACK,fillColor=fills[i],strokeWidth=.6));label(d,x+16,y-1,name,11,anchor='start')
    cases=['parameter_drift','composite']
    for panel,(metric,title,ymax,step) in enumerate([
        ('tracking_mae_rad_s','а  Погрешность скорости, рад/с',4.,1.),
        ('peak_current_a','б  Средний максимум тока, А',2.,.5),
    ]):
        x=40+panel*285
        label(d,x+105,219,title,12)
        bc=VerticalBarChart();bc.x=x;bc.y=56;bc.width=210;bc.height=145
        bc.data=[[METRICS[c+'/'+m][metric] for c in cases] for m in methods]
        bc.categoryAxis.categoryNames=['Изменение\nпараметров','Совместные\nвозмущения']
        bc.categoryAxis.labels.fontName=FONT;bc.categoryAxis.labels.fontSize=11;bc.categoryAxis.labels.dy=-4;bc.categoryAxis.labels.boxAnchor='n'
        bc.categoryAxis.strokeWidth=.6
        bc.valueAxis.valueMin=0.;bc.valueAxis.valueMax=ymax;bc.valueAxis.valueStep=step
        bc.valueAxis.labels.fontName=FONT;bc.valueAxis.labels.fontSize=11
        bc.valueAxis.labelTextFormat=lambda v:f'{v:g}'.replace('.',',')
        bc.valueAxis.visibleGrid=True;bc.valueAxis.gridStrokeColor=colors.HexColor('#dddddd');bc.valueAxis.gridStrokeWidth=.35
        bc.valueAxis.strokeWidth=.6;bc.groupSpacing=20;bc.barSpacing=2;bc.bars.strokeWidth=.6
        for i,fill in enumerate(fills):bc.bars[i].fillColor=fill;bc.bars[i].strokeColor=BLACK
        d.add(bc)
    label(d,280,8,'Средние значения по 20 независимым реализациям случайных воздействий',11)
    (ASSETS/'figure_2.svg').write_text(renderSVG.drawToString(d),encoding='utf-8')
    print('Narrative scheme and comparison charts prepared from verified numerical data.')


def picture(doc,n,caption):
    p=layout.new_p(doc,17,keep=True)
    layout.add(p,'').add_picture(str(ASSETS/f'figure_{n}.jpg'),width=Cm(15.9))
    p.paragraph_format.space_before=Pt(10)
    cp=layout.new_p(doc,35,f'Рисунок {n} – {caption}')
    cp.paragraph_format.first_line_indent=Cm(0);cp.paragraph_format.left_indent=Cm(0);cp.paragraph_format.right_indent=Cm(0)
    cp.paragraph_format.space_before=Pt(3);cp.paragraph_format.space_after=Pt(8)
    return cp


def percent(v):return f'{v:+.2f}'.replace('.',',').replace('-','−')
def number(v,places=3):return f'{v:.{places}f}'.replace('.',',')


def result_table(doc):
    p=layout.new_p(doc,35,'Таблица 1 – Сопоставление базовой системы и согласованной адаптации',keep=True)
    p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.left_indent=Cm(0);p.paragraph_format.right_indent=Cm(0)
    p.paragraph_format.space_before=Pt(8);p.paragraph_format.space_after=Pt(4)
    t=doc.add_table(rows=1,cols=4);t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    widths=[6.2,3.0,3.6,3.1]
    for col,width in zip(t.columns,widths):col.width=Cm(width)
    for cell,width in zip(t.rows[0].cells,widths):cell.width=Cm(width)
    for c,value in zip(t.rows[0].cells,['Показатель','Базовая\nсистема','Согласованная\nадаптация','Изменение,\n%']):c.text=value
    rows=[]
    specifications=[
        ('Погрешность скорости (MAE),\nрад/с','tracking_mae_rad_s',1.,3),
        ('Средний максимум тока, А','peak_current_a',1.,3),
        ('Стандартное отклонение\nмомента, мН·м','steady_torque_std_nm',1000.,3),
        ('Энергия потерь в меди\nобмотки статора, Дж','copper_energy_j',1.,2),
    ]
    group_indices=[]
    for case,caption in [('parameter_drift','Изменение параметров двигателя'),('composite','Совместное действие возмущений')]:
        group_indices.append(len(t.rows));row=t.add_row();row.cells[0].merge(row.cells[-1]).text=caption
        f=METRICS[case+'/fixed_reference'];c=METRICS[case+'/confidence_feedback']
        for label_,key,mult,places in specifications:
            row=t.add_row()
            vals=[label_,number(f[key]*mult,places),number(c[key]*mult,places),percent(100*(c[key]/f[key]-1))]
            for cell,width,value in zip(row.cells,widths,vals):cell.width=Cm(width);cell.text=value
            rows.append({'scenario':case,'metric':key,'reference':f[key],'adaptive':c[key],'change_percent':100*(c[key]/f[key]-1)})
    for ri,row in enumerate(t.rows):
        trpr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trpr.append(cant)
        if ri==0:
            repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
        seen=set()
        for ci,cell in enumerate(row.cells):
            if id(cell._tc) in seen:continue
            seen.add(id(cell._tc));cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                el=OxmlElement('w:'+edge)
                for key,value in [('val','single'),('sz','4'),('color','000000')]:el.set(qn('w:'+key),value)
                borders.append(el)
            pr.append(borders)
            margins=OxmlElement('w:tcMar')
            for edge in ['top','left','bottom','right']:
                el=OxmlElement('w:'+edge);el.set(qn('w:w'),'70');el.set(qn('w:type'),'dxa');margins.append(el)
            pr.append(margins)
            for pp in cell.paragraphs:
                pp.alignment=WD_ALIGN_PARAGRAPH.CENTER if ci or ri==0 or ri in group_indices else WD_ALIGN_PARAGRAPH.LEFT
                pp.paragraph_format.line_spacing=1.;pp.paragraph_format.space_before=Pt(1);pp.paragraph_format.space_after=Pt(1)
                pp.paragraph_format.keep_with_next=ri<len(t.rows)-1
                for rr in pp.runs:rr.font.name=FONT;rr.font.size=Pt(12);rr.bold=ri==0 or ri in group_indices
    (HERE/'narrative_table_values.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')


def package_save(doc):
    used=set(doc._element.xpath('//@r:embed'))
    for rid,rel in list(doc.part.rels.items()):
        if rel.reltype.endswith('/image') and rid not in used:doc.part.drop_rel(rid)
    doc.save(OUT)
    allowed={'[Content_Types].xml','word/document.xml','word/_rels/document.xml.rels','docProps/core.xml','word/settings.xml'}
    with ZipFile(OUT) as generated,ZipFile(layout.REF) as reference:
        parts={n:(generated.getinfo(n),generated.read(n)) for n in generated.namelist()}
        for n in set(parts)&set(reference.namelist()):
            if n not in allowed:parts[n]=(reference.getinfo(n),reference.read(n))
    info,settings=parts['word/settings.xml'];st=etree.fromstring(settings)
    update=st.find(qn('w:updateFields'))
    if update is None:update=OxmlElement('w:updateFields');st.append(update)
    update.set(qn('w:val'),'true')
    parts['word/settings.xml']=(info,etree.tostring(st,encoding='UTF-8',xml_declaration=True,standalone=True))
    temp=OUT.with_suffix('.tmp.docx')
    with ZipFile(temp,'w') as z:
        for n,(i,b) in parts.items():z.writestr(i,b)
    temp.replace(OUT)
    with ZipFile(OUT) as z,ZipFile(layout.REF) as r:
        changes=[n for n in set(z.namelist())&set(r.namelist()) if z.read(n)!=r.read(n)]
        assert set(changes)<=allowed,changes
        for n in ['word/styles.xml','word/footer1.xml','word/theme/theme1.xml','word/fontTable.xml']:assert z.read(n)==r.read(n)
    assert Document(OUT).sections[0]._sectPr.xml==layout.SOURCE.sections[0]._sectPr.xml
    return changes


def build():
    reference_hash=layout.sha(layout.REF)
    doc=Document(layout.REF)
    for child in list(doc._element.body):
        if child.tag!=qn('w:sectPr'):doc._element.body.remove(child)
    layout.new_p(doc,0,'УДК 681.518:621.313.333')
    layout.new_p(doc,1,TITLE,keep=True)
    p=layout.new_p(doc,2,keep=True)
    for i,name in enumerate(['А. Н. Марикин','В. Г. Жемчугов','А. Д. Онофрийчук'],1):
        if i>1:layout.add(p,', ',proto=layout.prototype(2,0))
        layout.add(p,name,proto=layout.prototype(2,0));layout.add(p,str(i),proto=layout.prototype(2,1))
    layout.new_p(doc,3,layout.SOURCE.paragraphs[3].text.strip(),keep=True)
    layout.short_citation(doc)
    layout.new_p(doc,5,'Аннотация.',keep=True)
    for key,value in ABSTRACT.items():layout.labelled(doc,key,value)
    layout.labelled(doc,'Ключевые слова:',KEYWORDS,index=9)
    for i,value in enumerate(INTRO):rich(doc,value,10 if i==0 else 11)
    p=rich(doc,'T<sub>r</sub> = L<sub>r</sub>/R<sub>r</sub>;     k<sub>r</sub> = L<sub>m</sub>/L<sub>r</sub>;     L<sub>r</sub> = L<sub>m</sub> + L<sub>σr</sub>.     (1)')
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Cm(0)
    p.paragraph_format.space_before=Pt(5);p.paragraph_format.space_after=Pt(5)
    for value in AFTER_EQUATION:rich(doc,value)
    picture(doc,1,'Структура векторного управления с согласованной адаптацией наблюдателя и регулятора')
    for value in METHODS:rich(doc,value)
    for value in RESULTS_BEFORE_FIGURE:rich(doc,value)
    picture(doc,2,'Влияние способа адаптации на погрешность скорости и максимальный ток')
    for value in RESULTS_BEFORE_TABLE:rich(doc,value)
    result_table(doc)
    for value in DISCUSSION:rich(doc,value)
    layout.new_p(doc,37,'По результатам исследования сформулированы следующие выводы.',keep=True)
    for value in CONCLUSIONS:rich(doc,value,38)
    p=layout.new_p(doc,45,'Библиографический список',keep=True);p.paragraph_format.space_before=Pt(12)
    for i,value in enumerate(layout.RU_REFS):
        p=layout.new_p(doc,46 if i==0 else 47,value);p.paragraph_format.keep_together=True
    layout.new_p(doc,58,'Контактная информация',keep=True)
    layout.new_p(doc,59,'МАРИКИН Александр Николаевич — доктор техн. наук, профессор',keep=True)
    layout.new_p(doc,61,'ЖЕМЧУГОВ Валерий Григорьевич — канд. техн. наук, доцент, zhem@mail.ru',keep=True)
    p=layout.new_p(doc,62,'ОНОФРИЙЧУК Андрей Дмитриевич — аспирант, 11lisov11@gmail.com')
    for r in p.runs:r.font.size=Pt(12);r.font.name=FONT
    layout.new_p(doc,63,TITLE_EN,keep=True)
    layout.new_p(doc,64,'A. N. Marikin, V. G. Zhemchugov, A. D. Onofriychuk',keep=True)
    layout.new_p(doc,65,layout.SOURCE.paragraphs[65].text.strip(),keep=True)
    layout.short_citation(doc,english=True)
    p=layout.new_p(doc,67);layout.add(p,'Summary\n',proto=layout.prototype(67,0),bold=True,size=12)
    for key,value in EN_ABSTRACT:
        layout.add(p,key,proto=layout.prototype(67,2),bold=True,size=12);layout.add(p,value,proto=layout.prototype(67,3),bold=False,size=12)
    p=layout.new_p(doc,68);layout.add(p,'Keywords: ',proto=layout.prototype(68,0),bold=True,size=12)
    layout.add(p,'induction motor drive; vector control; state observer; parameter adaptation; speed regulation; current loading; auxiliary transport equipment.',proto=layout.prototype(68,1),size=12)
    layout.new_p(doc,69,'References',keep=True)
    old=Document(HERE/'before_narrative_rewrite.docx');texts=[p.text for p in old.paragraphs]
    start=texts.index('References')+1;end=next(i for i in range(start,len(texts)) if texts[i].startswith('Author’s information'))
    for i,value in enumerate(texts[start:end]):
        p=layout.new_p(doc,70 if i==0 else 71,value);p.paragraph_format.keep_together=True
    p=layout.new_p(doc,81);layout.add(p,'Author’s information\n',proto=layout.prototype(81,0),bold=True,size=12)
    layout.add(p,'MARIKIN Alexander Nikolaevich — Doctor of Technical Sciences, Professor',proto=layout.prototype(81,2),bold=False,size=12)
    layout.new_p(doc,83,'ZHEMCHUGOV Valery Grigoryevich — Candidate of Technical Sciences, Associate Professor')
    layout.new_p(doc,84,'ONOFRIYCHUK Andrey Dmitrievich — Postgraduate student, 11lisov11@gmail.com')
    doc.core_properties.title=TITLE.capitalize();doc.core_properties.author='А. Н. Марикин; В. Г. Жемчугов; А. Д. Онофрийчук'
    doc.core_properties.subject='Параметрическая адаптация наблюдателя и векторного регулятора асинхронного электропривода';doc.core_properties.comments=''
    changes=package_save(doc)
    assert layout.sha(layout.REF)==reference_hash
    audit={'output':str(OUT),'title':TITLE,'central_result':'approximately threefold reduction in speed MAE with near-baseline mean maximum current in the two specified disturbed regimes','source_reference_unchanged':True,'source_styles_section_footer_preserved':True,'changed_parts':changes,'figures':2,'tables':1,'references':7,'conclusions':4,'narrative_order':['application','parameter mismatch','purpose and prior work','control structure','numerical study','accuracy and current tradeoff','torque and copper loss','applicability limits','quantitative conclusions']}
    (HERE/'narrative_build_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(OUT));print('Source layout preserved; narrative article built with 4 quantitative conclusions.')


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='figures':draw_assets()
    else:build()
