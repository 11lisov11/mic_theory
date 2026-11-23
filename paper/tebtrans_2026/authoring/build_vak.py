from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent
WORK=HERE.parent
ROOT=WORK.parent
REF=next(Path(r'C:\Study\Статьи и конференции\ВАК').glob('Марикин*.docx'))
CONTENT=ROOT/'Марикин_Оценка_токовых_ТЭБТРАНС_2026.docx'
OUT=ROOT/'Марикин_Оценка_токовых_ТЭБТРАНС_2026_ВАК.docx'
OLD=Document(CONTENT)
SOURCE=Document(REF)
TEMPLATES=[copy.deepcopy(p._p) for p in SOURCE.paragraphs]
TITLE=OLD.paragraphs[1].text
TITLE_EN='Assessment of Current Loading and Speed Control Quality of an Induction Motor Drive with Parameter Adaptation'
RU_CONFERENCE_REFS=[
    '3. Марикин А. Н., Жемчугов В. Г., Самонин А. Л., Онофрийчук А. Д., Хвостова О. В. Программно-аппаратная реализация управления асинхронным электродвигателем с учетом вектора потокосцепления ротора // Eltrans–2025: сборник трудов XII Международного симпозиума, посвященного 155-летию Г. О. Графтио. Санкт-Петербург, 2025. С. 185–189.',
    '4. Марикин А. Н., Жемчугов В. Г., Онофрийчук А. Д., Хвостова О. В. Особенности применения искусственного интеллекта для управления асинхронным приводом // Развитие инфраструктуры и логистических технологий в транспортных системах (РИЛТТРАНС-2025): сборник трудов Шестой Международной научно-практической конференции. Санкт-Петербург, 2025. С. 348–352.',
]
EN_CONFERENCE_REFS=[
    '3. Marikin A. N., Zhemchugov V. G., Samonin A. L., Onofriychuk A. D., Khvostova O. V. Software and hardware implementation of induction motor control taking into account the rotor flux linkage vector. Eltrans–2025: Proceedings of the 12th International Symposium dedicated to the 155th anniversary of G. O. Graftio. Saint Petersburg, 2025. P. 185–189. (In Russian).',
    '4. Marikin A. N., Zhemchugov V. G., Onofriychuk A. D., Khvostova O. V. Features of artificial intelligence application to induction drive control. Development of Infrastructure and Logistics Technologies in Transport Systems (RILTTRANS–2025): Proceedings of the 6th International Scientific and Practical Conference. Saint Petersburg, 2025. P. 348–352. (In Russian).',
]
INTERNATIONAL_REFS=[re.sub(r'^\d+\.',f'{j}.',OLD.paragraphs[i].text) for j,i in enumerate(range(32,35),5)]
RU_REFS=[OLD.paragraphs[i].text for i in [30,31]]+RU_CONFERENCE_REFS+INTERNATIONAL_REFS
SCIENTIFIC_BODY={
    10:'Математическая модель асинхронного двигателя составлена в неподвижной системе координат αβ при допущении линейности магнитной цепи. Она включает уравнения токов статора, потокосцепления ротора и механического движения. Приняты следующие параметры: R<sub>s</sub> = 7,868 Ом, R<sub>r</sub> = 20,832 Ом, L<sub>m</sub> = 0,4824 Гн; индуктивности рассеяния статора и ротора — 2,491 и 1,935 мГн; момент инерции — 7,646·10<sup>−5</sup> кг·м². Указанные значения характеризуют расчётную модель и не получены идентификацией конкретной машины. Напряжение преобразователя представлено усреднёнными значениями; насыщение магнитопровода не учитывается. Дифференциальные уравнения интегрируются методом Рунге — Кутты четвёртого порядка.',
    11:'Входными сигналами наблюдателя служат измеренные токи статора и восстановленные значения напряжения с учётом предшествующих отсчётов. Действительные значения скорости, потокосцепления, момента нагрузки и параметров двигателя используются при моделировании объекта и оценке показателей качества; наблюдателю они недоступны. Вектор состояния адаптивного РФК содержит девять компонентов: две составляющие тока, две составляющие потокосцепления ротора, угловую скорость, момент нагрузки и логарифмы значений R<sub>s</sub>, R<sub>r</sub>, L<sub>m</sub>, нормированных на номинальные величины. Логарифмическое представление обеспечивает положительность оцениваемых параметров. При изменении L<sub>m</sub> индуктивности рассеяния сохраняются постоянными.',
    12:'Исследованы четыре алгоритма: F — РФК с постоянными электрическими параметрами; A — совместное оценивание состояний и параметров при неизменных параметрах векторного регулятора; D — непосредственное использование параметрических оценок РФК в регуляторе; C — использование тех же оценок с ковариационным взвешиванием. В алгоритме C логарифмическую оценку каждого параметра умножают на весовой коэффициент от нуля до единицы. Коэффициенты определяются уменьшением соответствующих диагональных элементов ковариационной матрицы относительно начальной дисперсии 0,04. Взвешивание носит эвристический характер: уменьшение расчётной дисперсии не гарантирует идентифицируемости параметров. Во всех алгоритмах одинаковы регуляторы тока и процедуры учёта задержек и подавления выбросов измерений.',
    14:'Для каждого из четырёх алгоритмов исследованы четыре режима работы при 20 независимых реализациях случайных воздействий; общее число численных экспериментов составляет 320. Продолжительность каждого эксперимента — 2,5 с. Сравниваемым алгоритмам задавали одинаковые начальные условия, нагрузку и возмущающие воздействия. Шаг интегрирования и период управления составляют 100 мкс, напряжение звена постоянного тока — 325 В, ограничение задания электромагнитного момента — 0,6 Н·м, уставка ограничения тока — 2,8 А. Параметры алгоритмов определены до контрольного моделирования. Для F ковариационный параметр модели нагрузки определяли отдельно для каждого режима по предварительным расчётам. Информация о наличии и типе возмущения в алгоритм управления не передавалась.',
    15:'Во всех режимах начальные значения R<sub>s</sub>, R<sub>r</sub>, момента инерции и коэффициента трения варьировали в пределах ±5%. В режиме изменения параметров L<sub>m</sub> уменьшали на 15%, индуктивности рассеяния увеличивали на 20%, а R<sub>s</sub> дополнительно увеличивали на 30% за 1,2 с. Режим совместного действия возмущений дополнительно включает задержку измерений 0,3 мс, квантование тока с шагом 2 мА, потерю 0,5% отсчётов и импульсные помехи. В первых трёх режимах заданная угловая скорость достигает 150 рад/с; в четвёртом режиме при совместном действии возмущений — 3 рад/с. Момент нагрузки 0,25–0,45 Н·м прикладывали в случайный момент времени в интервале 0,95–1,25 с.',
    16:'Среднюю абсолютную погрешность регулирования скорости (MAE) определяли по отклонению действительной скорости от заданной на интервале после 0,8 с. Максимум модуля вектора тока в осях αβ определяли за всю продолжительность каждого эксперимента; затем вычисляли среднее значение этих максимумов по 20 реализациям. Энергию потерь в меди обмотки статора рассчитывали за 2,5 с интегрированием мгновенных потерь (3/2)R<sub>s</sub>|i<sub>s</sub>|<sup>2</sup> при амплитудно-инвариантном преобразовании Кларка. Независимым наблюдением при статистическом анализе служила одна реализация случайных воздействий. Для двух заранее определённых сравнений C/F доверительные интервалы уровня 97,5% получены методом повторных выборок с возвращением при сохранении парности данных (10 000 выборок). Применена поправка Бонферрони для двух гипотез о погрешности скорости. Сопоставления алгоритмов A и D имеют дополнительный характер.',
    18:'В исследованной совокупности из 320 численных экспериментов превышений уставки ограничения тока 2,8 А не установлено. Наибольшее значение модуля вектора тока составляет 2,324 А. На рисунке 1 представлены показатели четырёх режимов работы; в таблице 1 приведены результаты для изменения параметров двигателя и совместного действия возмущений.',
    22:'При изменении параметров двигателя алгоритм C уменьшает среднюю абсолютную погрешность регулирования скорости с 3,721 до 1,258 рад/с, то есть на 66,18%. При совместном действии возмущений погрешность уменьшается с 3,769 до 1,224 рад/с, то есть на 67,52%. Отношения MAE алгоритмов C/F составляют 0,338 и 0,325; верхние границы доверительных интервалов меньше единицы (рисунок 2). Средние максимальные значения тока изменяются соответственно на −0,15 и +0,84%. Верхние границы доверительных интервалов отношений этих значений равны 1,020 и 1,027, что ниже принятого в исследовании предельного отношения 1,05.',
    25:'Снижение погрешности регулирования сопровождается увеличением энергии потерь в меди обмотки статора на 2,99 и 2,63% соответственно. Стандартное отклонение электромагнитного момента на интервале после 1,7 с уменьшается на 34,34 и 31,83%. Для алгоритма A средняя абсолютная погрешность скорости составляет 0,713 и 0,772 рад/с, однако средние максимальные значения тока возрастают на 32,71 и 38,42%, что превышает принятый сравнительный критерий. Следовательно, ранжирование алгоритмов только по погрешности скорости не учитывает увеличения токовой нагрузки.',
    26:'При непосредственном использовании параметрических оценок в регуляторе (алгоритм D) средняя абсолютная погрешность скорости составляет 1,230 и 1,181 рад/с и несколько ниже соответствующих значений алгоритма C. Преимущество ковариационного взвешивания по исследованным показателям не выявлено. В номинальном режиме MAE алгоритма C составляет 1,614 рад/с, а алгоритма F — 0,879 рад/с; относительное увеличение достигает 83,60%. При заданной скорости 3 рад/с MAE алгоритма C равна 2,076 рад/с, или 69,2% заданного значения. Ни в одном из 20 экспериментов после приложения нагрузки не получен непрерывный интервал длительностью 100 мс с абсолютной погрешностью менее 0,3 рад/с. Указанные результаты определяют ограничения применимости исследованного способа адаптации.',
}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def clear_content(element):
    for child in list(element):
        if child.tag!=qn('w:pPr'):element.remove(child)

def prototype(index,run=0):
    pp=SOURCE.paragraphs[index]
    if pp.runs:
        return copy.deepcopy(pp.runs[min(run,len(pp.runs)-1)]._r.rPr)
    return None

def add(p,text,*,proto=None,bold=None,italic=None,size=None):
    text=text.replace('10 000','10\u00a0000')
    text=re.sub(r'(\d+(?:[.,]\d+)?) (?=(?:рад/с|Ом|мГн|Гн|мкс|мс|Н·м|Дж|[сАВ])(?:[\s,.;]|$))',lambda m:m.group(1)+'\u00a0',text)
    r=p.add_run(text)
    if proto is not None:r._r.insert(0,copy.deepcopy(proto))
    if bold is not None:r.bold=bold
    if italic is not None:r.italic=italic
    if size is not None:r.font.size=Pt(size)
    return r

def new_p(doc,index,text=None,*,run=0,keep=False):
    element=copy.deepcopy(TEMPLATES[index]);clear_content(element)
    doc._element.body.insert(len(doc._element.body)-1,element)
    p=Paragraph(element,doc._body)
    if text is not None:add(p,text,proto=prototype(index,run))
    if keep:p.paragraph_format.keep_with_next=True
    return p

def source_text(doc,old_index,template=11):
    if old_index in SCIENTIFIC_BODY:
        p=new_p(doc,template)
        for token in re.split(r'(<sub>.*?</sub>|<sup>.*?</sup>)',SCIENTIFIC_BODY[old_index]):
            if not token:continue
            sub=token.startswith('<sub>');sup=token.startswith('<sup>')
            r=add(p,re.sub(r'</?(?:sub|sup)>','',token),proto=prototype(template))
            if sub:r.font.subscript=True
            if sup:r.font.superscript=True
        return p
    p=new_p(doc,template)
    for oldrun in OLD.paragraphs[old_index].runs:
        if not oldrun.text:continue
        r=add(p,oldrun.text,proto=prototype(template))
        if oldrun.font.subscript:r.font.subscript=True
        if oldrun.font.superscript:r.font.superscript=True
    return p

def labelled(doc,label,value,index=6,*,before=None):
    p=new_p(doc,index)
    if before is not None:p.paragraph_format.space_before=Pt(before)
    add(p,label,proto=prototype(6,0),bold=True)
    add(p,' '+value,proto=prototype(6,1),bold=False)
    return p

def short_citation(doc,english=False):
    p=new_p(doc,66 if english else 4)
    add(p,'For citation: ' if english else 'Для цитирования: ',proto=prototype(4,0),bold=True)
    names='Marikin A. N., Zhemchugov V. G., Onofriychuk A. D.' if english else 'Марикин А. Н., Жемчугов В. Г., Онофрийчук А. Д.'
    add(p,names+' ',proto=prototype(4,1),italic=True)
    add(p,(TITLE_EN if english else TITLE.capitalize())+'.',proto=prototype(6,1),italic=False)
    p.paragraph_format.keep_with_next=True
    return p

def picture(doc,n,caption):
    p=new_p(doc,17,keep=True)
    add(p,'').add_picture(str(WORK/f'figure_{n}.jpg'),width=Cm(15.9))
    p.paragraph_format.space_before=Pt(12)
    cp=new_p(doc,35,f'Рисунок {n} – {caption}')
    cp.paragraph_format.first_line_indent=Cm(0)
    cp.paragraph_format.left_indent=Cm(0)
    cp.paragraph_format.right_indent=Cm(0)
    cp.paragraph_format.space_before=Pt(4)
    cp.paragraph_format.space_after=Pt(10)
    return cp

def table(doc):
    p=new_p(doc,35,'Таблица 1 – Показатели качества регулирования и токовой нагрузки при изменении параметров двигателя и совместном действии возмущений',keep=True)
    p.paragraph_format.first_line_indent=Cm(0)
    p.paragraph_format.left_indent=Cm(0)
    p.paragraph_format.right_indent=Cm(0)
    p.paragraph_format.space_before=Pt(9)
    p.paragraph_format.space_after=Pt(4)
    el=copy.deepcopy(OLD.tables[0]._tbl)
    doc._element.body.insert(len(doc._element.body)-1,el)
    t=doc.tables[-1]
    headers=['Режим','Метод','MAE,\nрад/с','Средний максимум\nтока, А','Энергия потерь\nв обмотке статора, Дж']
    for cell,value in zip(t.rows[0].cells,headers):cell.text=value
    for a,b,label in [(1,4,'Изменение\nпараметров'),(5,8,'Совместные\nвозмущения')]:
        cell=t.cell(a,0).merge(t.cell(b,0));cell.text=label
    for ri,row in enumerate(t.rows):
        for cell in row.cells:
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr=cell._tc.get_or_add_tcPr()
            for node in list(tcpr):
                if node.tag in [qn('w:shd'),qn('w:tcBorders')]:tcpr.remove(node)
            borders=OxmlElement('w:tcBorders')
            for side in ['top','left','bottom','right']:
                node=OxmlElement('w:'+side)
                for key,value in [('val','single'),('sz','4'),('color','000000')]:node.set(qn('w:'+key),value)
                borders.append(node)
            tcpr.append(borders)
            for pp in cell.paragraphs:
                pp.alignment=WD_ALIGN_PARAGRAPH.CENTER
                pp.paragraph_format.line_spacing=1.
                pp.paragraph_format.space_before=Pt(2)
                pp.paragraph_format.space_after=Pt(2)
                pp.paragraph_format.keep_with_next=ri<len(t.rows)-1
                for rr in pp.runs:
                    rr.font.name='Times New Roman';rr.font.size=Pt(12);rr.bold=ri==0

def do_build():
    original_hash=sha(REF)
    audit={'reference':str(REF),'sha256':original_hash,'source_render_pages':10,'section_xml':SOURCE.sections[0]._sectPr.xml,'roles':{}}
    for name,i in [('udc',0),('title',1),('authors',2),('affiliation',3),('abstract',6),('body',11),('caption',35),('references',46),('contact',59),('english_title',63)]:
        audit['roles'][name]={'paragraph':i,'pPr':copy.deepcopy(TEMPLATES[i]).find(qn('w:pPr')).xml if TEMPLATES[i].find(qn('w:pPr')) is not None else ''}
    with ZipFile(REF) as z:
        audit['parts']={n:{'size':len(z.read(n)),'sha256':hashlib.sha256(z.read(n)).hexdigest()} for n in z.namelist()}
    (HERE/'reference_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')

    shutil.copy2(REF,HERE/'reference_working_copy.docx')
    doc=Document(HERE/'reference_working_copy.docx')
    for child in list(doc._element.body):
        if child.tag!=qn('w:sectPr'):doc._element.body.remove(child)
    new_p(doc,0,'УДК 681.518:621.313.333')
    new_p(doc,1,TITLE,keep=True)
    p=new_p(doc,2,keep=True)
    for i,name in enumerate(['А. Н. Марикин','В. Г. Жемчугов','А. Д. Онофрийчук'],1):
        if i>1:add(p,', ',proto=prototype(2,0))
        add(p,name,proto=prototype(2,0))
        add(p,str(i),proto=prototype(2,1))
    new_p(doc,3,SOURCE.paragraphs[3].text.strip(),keep=True)
    short_citation(doc)
    new_p(doc,5,'Аннотация.',keep=True)
    labelled(doc,'Цель:','Оценить влияние параметрической адаптации на погрешность регулирования скорости, максимальные значения тока и энергию потерь в меди обмотки статора бездатчикового асинхронного электропривода.')
    labelled(doc,'Методы:','Проведено 320 численных экспериментов для четырёх алгоритмов управления в четырёх режимах работы при 20 независимых реализациях случайных воздействий. Алгоритмы сопоставлены при одинаковых начальных условиях и возмущениях. Применено непараметрическое оценивание доверительных интервалов.')
    labelled(doc,'Результаты:','При изменении параметров двигателя и совместном действии возмущений использование адаптивных оценок в регуляторе уменьшило среднюю абсолютную погрешность скорости на 66–69%. При ковариационном взвешивании средние максимальные значения тока изменились менее чем на 1%, а энергия потерь в меди обмотки статора увеличилась на 2,6–3,0%. Преимущество взвешивания перед непосредственным использованием оценок не установлено.')
    labelled(doc,'Практическая значимость:','Совместная оценка погрешности регулирования, токовой нагрузки и электрических потерь позволяет определить условия применения адаптации во вспомогательных приводах транспорта. Практическая реализация требует идентификации параметров двигателя и экспериментальной проверки на стенде.')
    labelled(doc,'Ключевые слова:','асинхронный электропривод; бездатчиковое управление; параметрическая адаптация; расширенный фильтр Калмана; токовые нагрузки; техносферная безопасность.',index=9)

    new_p(doc,10,'Для вспомогательных приводов транспортной инфраструктуры существенны точность регулирования, токовые нагрузки и потери, определяющие условия работы электрической машины и преобразователя. Снижение погрешности регулирования скорости само по себе не характеризует допустимость режима. В предыдущих работах авторов рассмотрены управление с оптронным датчиком [1] и нейроуправление с обучением в цифровом двойнике [2].')
    new_p(doc,11,'Программно-аппаратной реализации управления с учётом вектора потокосцепления ротора посвящена работа [3], особенностям применения искусственного интеллекта для управления асинхронным приводом — работа [4]. Настоящее исследование развивает эту тематику в части количественной оценки параметрической адаптации без механического датчика скорости по совокупности показателей точности, тока и потерь.')
    new_p(doc,11,'Совместная оценка состояний и сопротивлений двигателя с помощью расширенного фильтра Калмана (РФК) рассмотрена в [5], а оценка индуктивности намагничивания и бездатчиковое управление в зоне ослабления поля — в [6]. В экспериментальном сравнении нелинейных фильтров Калмана [7] показана зависимость их настройки от условий работы и рассмотрено взаимодействующее многомодельное оценивание. Цель данной работы — количественно сопоставить способы использования адаптивных оценок в регуляторе по точности скорости, токовым нагрузкам и энергии потерь.')
    for i in [10,11,12,14,15,16,18]:source_text(doc,i)
    picture(doc,1,'Средняя абсолютная погрешность регулирования скорости и средние максимальные значения тока в четырёх режимах работы')
    table(doc)
    source_text(doc,22)
    picture(doc,2,'Отношения показателей алгоритмов C/F и доверительные интервалы уровня 97,5%')
    for i in [25,26]:source_text(doc,i)
    new_p(doc,37,'По результатам моделирования сформулированы следующие выводы.')
    conclusions=[
        '1. Использование адаптивных параметрических оценок в регуляторе уменьшает погрешность регулирования скорости при изменении параметров двигателя и совместном действии возмущений. Выбор алгоритма требует одновременного учёта точности регулирования, максимальных значений тока и электрических потерь.',
        '2. Преимущество ковариационного взвешивания перед непосредственным использованием параметрических оценок не установлено. В номинальном режиме и при малой скорости выявлены ограничения адаптации. В режимах изменения параметров и совместного действия возмущений алгоритмы C и D не обеспечивают уменьшения энергии потерь в обмотке статора; снижение погрешности регулирования само по себе не подтверждает безопасность электропривода.',
        '3. Результаты получены для модели с линейной магнитной цепью и усреднённым напряжением при 20 независимых реализациях случайных воздействий. Модель двигателя и модель прогноза наблюдателя основаны на одной системе дифференциальных уравнений. Коммутационные потери, температурные процессы и вероятность отказа не оценивались. Применение во вспомогательных приводах транспорта требует идентификации параметров машины, проверки защит и стендовых испытаний.',
    ]
    for value in conclusions:new_p(doc,38,value)
    p=new_p(doc,45,'Библиографический список',keep=True)
    p.paragraph_format.space_before=Pt(14)
    for i,value in enumerate(RU_REFS):
        p=new_p(doc,46 if i==0 else 47,value)
        p.paragraph_format.keep_together=True
    new_p(doc,58,'Контактная информация',keep=True)
    new_p(doc,59,'МАРИКИН Александр Николаевич — доктор техн. наук, профессор',keep=True)
    new_p(doc,61,'ЖЕМЧУГОВ Валерий Григорьевич — канд. техн. наук, доцент, zhem@mail.ru',keep=True)
    p=new_p(doc,62,'ОНОФРИЙЧУК Андрей Дмитриевич — аспирант, 11lisov11@gmail.com')
    for r in p.runs:r.font.size=Pt(12);r.font.name='Times New Roman'

    new_p(doc,63,TITLE_EN,keep=True)
    new_p(doc,64,'A. N. Marikin, V. G. Zhemchugov, A. D. Onofriychuk',keep=True)
    new_p(doc,65,SOURCE.paragraphs[65].text.strip(),keep=True)
    short_citation(doc,english=True)
    p=new_p(doc,67)
    add(p,'Summary\n',proto=prototype(67,0),bold=True,size=12)
    eng=[
        ('Objective: ','To assess the effects of parameter adaptation on speed regulation error, maximum current and stator winding copper-loss energy in a sensorless induction motor drive. '),
        ('Methods: ','A total of 320 numerical experiments were conducted for four control algorithms and four operating regimes with 20 independent realizations of random disturbances. The algorithms were compared under identical initial conditions and disturbances. Confidence intervals were estimated using a nonparametric resampling method. '),
        ('Results: ','Using adaptive parameter estimates in the controller reduced the mean absolute speed regulation error by 66–69% under parameter variation and combined disturbances. With covariance weighting, the mean maximum current changed by less than 1%, whereas stator winding copper-loss energy increased by 2.6–3.0%. No advantage of covariance weighting over direct use of the estimates was established. Limitations were identified under nominal conditions and at low speed. '),
        ('Practical significance: ','Joint assessment of regulation error, current loading and electrical losses determines the conditions for applying adaptation to auxiliary transport drives. Practical implementation requires motor parameter identification and experimental validation on a test bench.'),
    ]
    for label,value in eng:
        add(p,label,proto=prototype(67,2),bold=True,size=12)
        add(p,value,proto=prototype(67,3),bold=False,size=12)
    p=new_p(doc,68)
    add(p,'Keywords: ',proto=prototype(68,0),bold=True,size=12)
    add(p,'induction motor drive; sensorless control; parameter adaptation; extended Kalman filter; current loading; technosphere safety.',proto=prototype(68,1),size=12)
    new_p(doc,69,'References',keep=True)
    english_refs=[
        '1. Marikin A. N., Evstafyev A. M., Zhemchugov V. G., Onofriychuk A. D., Khvostova O. V. Vector control of an induction motor with an optocoupler algorithm-forming sensor. Proceedings of Petersburg Transport University. 2025. Vol. 22, no. 3. P. 667–675. DOI: 10.20295/1815-588X-2025-3-667-675. (In Russian).',
        '2. Marikin A. N., Evstafyev A. M., Zhemchugov V. G., Onofriychuk A. D. Study of the features of neural control of an induction motor trained in a digital twin. Proceedings of Petersburg Transport University. 2026. Vol. 23, no. 3. P. 596–603. DOI: 10.20295/1815-588X-2026-3-596-603. (In Russian).',
    ]+EN_CONFERENCE_REFS+INTERNATIONAL_REFS
    for i,value in enumerate(english_refs):
        p=new_p(doc,70 if i==0 else 71,value)
        p.paragraph_format.keep_together=True
    p=new_p(doc,81)
    add(p,'Author’s information\n',proto=prototype(81,0),bold=True,size=12)
    add(p,'MARIKIN Alexander Nikolaevich — Doctor of Technical Sciences, Professor',proto=prototype(81,2),bold=False,size=12)
    new_p(doc,83,'ZHEMCHUGOV Valery Grigoryevich — Candidate of Technical Sciences, Associate Professor')
    new_p(doc,84,'ONOFRIYCHUK Andrey Dmitrievich — Postgraduate student, 11lisov11@gmail.com')

    doc.core_properties.title=TITLE.capitalize()
    doc.core_properties.author='А. Н. Марикин; В. Г. Жемчугов; А. Д. Онофрийчук'
    doc.core_properties.subject='Оценка токовых нагрузок и качества регулирования асинхронного электропривода'
    doc.core_properties.comments=''
    used=set(doc._element.xpath('//@r:embed'))
    for rid,rel in list(doc.part.rels.items()):
        if rel.reltype.endswith('/image') and rid not in used:doc.part.drop_rel(rid)
    doc.save(OUT)

    allowed={'[Content_Types].xml','word/document.xml','word/_rels/document.xml.rels','docProps/core.xml','word/settings.xml'}
    with ZipFile(OUT) as generated,ZipFile(REF) as reference:
        parts={n:(generated.getinfo(n),generated.read(n)) for n in generated.namelist()}
        for n in set(parts)&set(reference.namelist()):
            if n not in allowed:parts[n]=(reference.getinfo(n),reference.read(n))
    info,settings=parts['word/settings.xml']
    from lxml import etree
    st=etree.fromstring(settings)
    update=st.find(qn('w:updateFields'))
    if update is None:update=OxmlElement('w:updateFields');st.append(update)
    update.set(qn('w:val'),'true')
    parts['word/settings.xml']=(info,etree.tostring(st,encoding='UTF-8',xml_declaration=True,standalone=True))
    temp=OUT.with_suffix('.tmp.docx')
    with ZipFile(temp,'w') as z:
        for n,(i,b) in parts.items():z.writestr(i,b)
    temp.replace(OUT)
    assert sha(REF)==original_hash
    new=Document(OUT)
    assert new.sections[0]._sectPr.xml==SOURCE.sections[0]._sectPr.xml
    with ZipFile(OUT) as z,ZipFile(REF) as r:
        differences=[n for n in set(z.namelist())&set(r.namelist()) if z.read(n)!=r.read(n)]
        assert set(differences)<=allowed,differences
        assert z.read('word/footer1.xml')==r.read('word/footer1.xml')
        assert z.read('word/styles.xml')==r.read('word/styles.xml')
    (HERE/'build_audit.json').write_text(json.dumps({'output':str(OUT),'reference_unchanged':True,'section_exact':True,'styles_exact':True,'footer_exact':True,'changed_parts':differences,'figures':len(new.inline_shapes),'tables':len(new.tables),'russian_and_english_blocks':True},ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(OUT))
    print('Exact source section, styles and footer preserved; paragraphs',len(new.paragraphs))

if __name__=='__main__':do_build()
