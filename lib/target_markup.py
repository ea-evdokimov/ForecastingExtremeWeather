import json
import re
import typing as tp
from enum import Enum

import pandas as pd


class TAGS(str, Enum):
    VETER: str = 'сильный ветер'
    SHKVAL: str = 'шквал'
    METEL: str = 'метель'
    DOZD: str = 'сильный дождь'
    SNEG: str = 'снег'
    GRAD: str = 'град'
    TUMAN: str = 'туман'
    GOLOLED: str = 'гололедно-изморозевое отложение'


if __name__ == '__main__':
    with open('tags.json', 'w') as f:
        json.dump({i.name: i.value for i in TAGS}, f, ensure_ascii=False)


class Target:
    def __init__(self):
        self._values: tp.Dict = {
            k: False for k in TAGS
        }

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def export(self):
        return {k.name: v for k, v in self._values.items()}


def classify_based_on_text(row: pd.Series) -> tp.Dict:
    target = Target()

    WW, W1, W2, E, E_1 = str(row['WW']), str(row['W1']), str(row['W1']), str(row['E']), str(row["E'"])
    regexps = {
        TAGS.VETER: re.compile(r'ветер|ветр', re.I),
        TAGS.SHKVAL: re.compile(r'шквал', re.I),
        TAGS.METEL: re.compile(r'метел', re.I),
        TAGS.DOZD: re.compile(r'дожд|ливен|морос', re.I),
        TAGS.SNEG: re.compile(r'снег|снежн|круп|зерн', re.I),
        TAGS.GRAD: re.compile(r'град', re.I),
        TAGS.TUMAN: re.compile(r'туман|мгла', re.I),
        TAGS.GOLOLED: re.compile(r'голол[её]д', re.I),
    }
    for tag, regexp in regexps.items():
        if regexp.search(WW) or regexp.search(W1) or regexp.search(W2) or regexp.search(E) or regexp.search(E_1):
            target[tag] = True

    return target.export()


def classify(row: pd.Series) -> tp.Dict:
    target = Target()

    # regex parse of WW, W1, W2 on keywords
    WW, W1, W2, E, E_1 = str(row['WW']), str(row['W1']), str(row['W1']), str(row['E']), str(row["E'"])
    regexps = {
        TAGS.VETER: re.compile(r'ветер|ветр', re.I),
        TAGS.SHKVAL: re.compile(r'шквал', re.I),
        TAGS.METEL: re.compile(r'метел', re.I),
        TAGS.DOZD: re.compile(r'дожд|ливен|морос', re.I),
        TAGS.SNEG: re.compile(r'снег|снежн|круп|зерн', re.I),
        TAGS.GRAD: re.compile(r'град', re.I),
        TAGS.TUMAN: re.compile(r'туман|мгла', re.I),
        TAGS.GOLOLED: re.compile(r'голол[её]д', re.I),
    }
    for tag, regexp in regexps.items():
        if regexp.search(WW) or regexp.search(W1) or regexp.search(W2) or regexp.search(E) or regexp.search(E_1):
            target[tag] = True

    # preprocessing nan
    VV_to_float = {
        'менее 0.05': 0.05,
        'менее 0.1': 0.1,
    }
    if row['VV'] in VV_to_float.keys():
        row['VV'] = VV_to_float[row['VV']]
    else:
        row['VV'] = float(row['VV'])
    RRR_to_zero = ['Осадков нет', 'Следы осадков']
    if row['RRR'] in RRR_to_zero:
        row['RRR'] = 0
    else:
        row['RRR'] = float(row['RRR'])

    # heuristics based on meteoinfo.ru
    if row['Ff'] >= 11.0 or row['ff3'] >= 15.0:
        target[TAGS.VETER] = True
    if row['ff3'] - row['Ff'] >= 10:
        target[TAGS.SHKVAL] = True
    if row['Ff'] >= 15.0 and row['VV'] <= 0.5:
        target[TAGS.METEL] = True
    if row['RRR'] >= 50.0 and row['tR'] <= 12:
        target[TAGS.DOZD] = True
    if row['VV'] <= 0.05:
        target[TAGS.TUMAN] = True

    return target.export()
