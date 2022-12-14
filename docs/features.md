Feature analysis
---------------

Анализ сделан по датасету с шести станций (тренировочному), в нем 356.5 тысяч записей.

### Признаки, с которыми все хорошо

- таких нет

### Признаки, которые легко выпрямить

Признаки ниже проще всего выпрямить линейно, потому что пропусков немного и они небольшие.

- `T` - температура у станции. 420 пропусков (0.1%), длина самого большого - 7 (21 час).
- `P` - давление на уровне моря. 561 пропуск (0.15%), длина самого большого - 13.
- `VV` - горизонтальная дальность видимости. 291 пропуск (0.08%), длина самого большого - 7.
- `U` - относительная влажность на высоте 2 метра. 841 пропуск (0.25%), длина самого большого - 7.
- `DD` - направление ветра. 3480 пропусков (1%), но длина самого длинного пропуска - 10. Могут
возникнуть сложности с тем, что прзнак текстовый, но вполне можно интерполировать предыдущими значениями
направления ветра / округлять линейную аппроксимацию до осмысленных направлений.
- `Ff` - скорость ветра на высоте 10 метров над землей. 3457 пропусков (1%), длина самого большого тоже 10.
Придется немного повозиться с регулярками, но в целом легко аппроксимируется.

### Признаки, которые можно наивно поправить, но достаточно много пропусков подряд

- `Po` - атмосферное давление на уровне станции. 2696 пропусков (0.7%), но самый длинный уже 441 ячейка (почти 2 месяца).
Из плюсов - признак невероятно сильно коррелирует с P, который уже легко восстанавливается линейной интерполяций, а этот
можно получить как умножение того на константу / прибавление константы в местах пропусков, качество получается хорошее.
- `N` - общая облачность в процентах. 9276 пропусков, самый большой - порядка 895 (не точно, так как сейчас таблица 
объединена по станциям). Можно пробовать восстанавливать по признакам W1, W2 с текстовым описанием, если они присутствуют.
Если нет, то можно пытаться предсказать по другим признакам в месте больших пропусков (есть и маленькие).
- `Td` - температура точки росы на высоте 2 метра над уровнем Земли. 9371 пропуск, 895 - самый длинный (число не похоже на случайность),
видимо где-то часть данных потерялась. Может быть легко предсказана по температуре и относительной влажности (привет, термодинамика!).

### Признаки, по которым много пропусков, но это нестрашно и можно почти ничего не делать

- `Pa` - изменение давления за 3 часа, а по давлению пропусков почти нет. Этого признака напропущено примерно полдатасета. 
- `ff10` - максимальная скорость ветра на высоте 10 метров за последние 10 минут перед измерением. Пропущено 345 тыс. записей (97%).
Признак может иметь большое значение для предсказания сильного ветра, так как его распределение в основном зажато между 10 м/c 
(сильный ветер) и 20 м/c (очень сильный ветер).
- `ff3` - максимальная скорость ветра на высоте 10 метров за последние 3 часа. Пропущено 327 тыс. записей (92%). Аналогично ff10 -
восстанавливать вряд ли имеет смысл, так как ветер в парадигме одной станции скорее таргет, чем признак.
- `WW`/`WW1`/`WW2` - для WW кажется, что пропусков ноль, но на самом деле их 222-224 тыс --- там строка, состоящая из одного пробела. 
Эти три признака, как правило, есть либо все 3, либо пропущены все. Хранят текстовое описание текущей/между сроками наблюдения 
(первая и вторая часть) погоды. Восстановить вряд ли получится, достаточно завести доп. категорию вида "ничего не было замечено".
- `RRR` - Количество выпавших осадков в mm. 261 тыс. пропусков (73%). Когда этого признака нет, осадков тоже (скорее всего) нет. Это можно попробовать проверить,
но скорее всего, это отсутствие осадков --- ставьте 0 или заведите категорию под отсутствие осадков.
- `tR` - Период времени, за который указано выпавшее число осадков в RRR. Пропусков столько же - 261 тыс. Править нужно
согласованно с RR. Бывает 1, 2, 3, 6, 9, 12, 15, 18 и 24 часа.
- `E` - Состояние поверхности почвы без снега или измеримого ледяного покрова. 320 тыс. пропусков (90%). Содержит текст в виде: `Поверхность почвы сырая` или `Несвязанная сухая пыль или песок не покрывают поверхность почвы полностью`. Дополнять текст не надо, просто завести под пропуски отдельную категорию.
- `E'` - состояние почвы со снегом или измеримым ледяным покровом. 340 тыс. пропусков (96%). Содержит текст аналогично `E`. Обрабатывать
так же. Возможно, за пределами зимы это должна быть еще одна категория - надо данные смотреть.

### Признаки, по которым много пропусков, и нужен доп. анализ и обработка

- `Tn` - минимальная температура воздуха за прошедший период (не более 12 часов). 297 тысяч пропусков (83%), 1183 - самый длинный. Понятно как можно восстановить, если есть информация о температуре, нужно отдельно изучать, как он соотносится с минимумом по скользящему окну.
Если совпадают, то признак бесполезный, но в целом --- его можно агрегировать для упрощения жизни нейронным сетям.
- `Tx` - максимальная температура воздуха за прошедший период. 313 тысяч пропусков (88%). Аналогично тому, что выше. Не одновременно с
Tn присутствуют или отсутствуют в данных. Также может быть триггером о происходящей аномалии / быстрых перепадах температуры.
- `Tg` - минимальная температура поверхности почвы за ночь. 350 тыс. пропусков (98%). Содержится далеко не везде, но с учетом усреднения
на сезоны и на ночь пропусков не так уж и много. Нужно поискать здесь сезонность и подумать о полезность предсказывания этого признака
за пределы имеющихся данных (или надо вставить шум и признак вида "признак Tg в данных имеется").
- `sss` - высота снежного покрова в см. 342 тыс. признаков, хорошо бы понять, что происходит со снежным покровом в зависимости от осадков
и температуры. За пределами "зимы" может ставить 0, но как проводить интерполяцию внутри зимы, сходу не так очевидно.

#### Облачные признаки, которые надо разбирать вместе
- `Cl` - текстовое описание тяжелых низких облаков. 98 тыс. пропусков (27%), 897 - самый длинный. После перегона в категориальный
признак можно забить и довольствоваться другими признаками про облака (или подумать и понять, что отсутствие этого признака = отсутствие
облачности - в любом случае отдельной категории под отсутствие будет предостаточно).
- `Cm` - текстовое описание средних облаков. 144 тыс. пропусков (40%), также 897 - самый длинный. Аналогично тому, что выше.
- `Ch` - текстовое описание высоких облаков. 181 тыс. пропусков (49%), 897 - самый длинный (коррелирует с тем, что выше). На самом деле тоже
аналогично тому, что выше.
- `Nh` - Количество всех наблюдающихся облаков вида Cl, при их отсутствии - Cm. 64 тыс. (18%) пропусков, 897 - cамый длинный. Нужно понять,
как он ведет себя при отсутствии признаков Cm и Cl --- может это ясная погода?)
- `H` - Высота нижней кромки облаков. 64 тыс. пропусков (18%), 2718 - самый длинный. Аналогично Nh - надо разбираться, не означает ли NaN
отсутствие нижней кромки вообще (64 тыс. - довольно много пропусков).





