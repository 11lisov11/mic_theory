# Цифровой эксперимент АИР56

[Научный отчёт](../../docs/air56_rotor_fingerprinting_ru.md) связывает семейства R0–R5 с постановками и результатами. В [reference](reference/) сохранены небольшие итоговые таблицы и контрольные суммы. Полные поля, сетки, CAD и временные ряды находятся в [архиве расчёта](C:/mic_air56_marker_results/). [Исходные параметры и материалы](../air56_interlaminar_marker/) общие с контрольным исследованием.

### Проверка сохранённого расчёта

Из корня рабочего дерева в PowerShell:

```powershell
$python = 'C:/mic_theory/.venv/Scripts/python.exe'
$out = 'C:/mic_air56_marker_results/family_screen'
& $python research/air56_rotor_fingerprinting/reproduce.py --out $out --verify
```

Для нового расчёта передайте новый пустой каталог через `--out` и опустите `--verify`. Команда запускает расчёт поля двух сеток, температурное сравнение, CAD, GetDP, объёмную теплопроводность и сравнение контроллеров. Параметры `--cad-python`, `--getdp` и `--policy-root` задают интерпретатор FreeCAD, исполняемый файл GetDP и корень с обученными политиками. Версии библиотек и хэши исходных данных приведены в [манифесте](reference/study_manifest.json). Хэши относятся к сохранённым файлам. При новом экспорте STEP или FCStd возможны изменения служебных меток времени.

### Проверка наклонов токов и фазовой карты

Используются сохранённая нелинейная рабочая точка, независимые сетки калибровки и контроля, а также четыре непрерывные траектории. Следующий цикл рассчитан на новый каталог. Шум добавляется только при последующей оценке.

```powershell
$motion = 'C:/mic_air56_marker_results/pwm_slopes_reproduction'
$loaded = 'C:/mic_air56_marker_results/loaded_contrast_18_phase_fem'
foreach ($skew in 0,1) {
    foreach ($mesh in 0,1) {
        & $python tools/run_air56_pwm_slopes.py --loaded-dir $loaded --skew $skew --mesh $mesh --out "$motion/skew_$skew/mesh_$mesh"
        if ($LASTEXITCODE) { throw "Ошибка расчёта карты" }
    }
    & $python tools/run_air56_pwm_slopes.py --loaded-dir $loaded --skew $skew --direct --out "$motion/direct/skew_$skew"
    if ($LASTEXITCODE) { throw "Ошибка расчёта реверса" }
}
& $python tools/score_air56_pwm_slopes.py --maps $motion --direct "$motion/direct" --out "$motion/final"
```

[Построение исходной рабочей точки](../../docs/air56_moving_pwm.md) · [Сценарии межлистовой метки и контроля](../../docs/air56_interlaminar_marker_research.md) · [Сравнение методов при движении](C:/mic_air56_marker_results/pwm_slopes_18/final/) · [План будущих измерений](../air56_interlaminar_marker/measurement_protocol_ru.md).
