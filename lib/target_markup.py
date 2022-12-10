import json
import typing as tp
import re

import pandas as pd
from pydantic import BaseModel


class TAGS(BaseModel):
    VETER: str = 'сильный ветер'
    SHKVAL: str = 'шквал'
    METEL: str = 'метель'
    DOZD: str = 'сильный дождь'
    SNEG: str = 'снег'
    GRAD: str = 'град'
    TUMAN: str = 'туман'
    GOLOLED: str = 'гололедно-изморозевое отложение'


TAGS_INSTANCE = TAGS()
if __name__ == '__main__':
    with open('TAGS_INSTANCE.json', 'w') as f:
        json.dump(TAGS_INSTANCE.dict(), f, ensure_ascii=False)


class Target(BaseModel):
    values = {
        TAGS_INSTANCE.VETER: False,
        TAGS_INSTANCE.SHKVAL: False,
        TAGS_INSTANCE.METEL: False,
        TAGS_INSTANCE.DOZD: False,
        TAGS_INSTANCE.SNEG: False,
        TAGS_INSTANCE.GRAD: False,
        TAGS_INSTANCE.TUMAN: False,
        TAGS_INSTANCE.GOLOLED: False,
    }


def classify(row: pd.Series) -> tp.Dict:
    target = Target()

    # regex parse of WW, W1, W2 on keywords
    WW, W1, W2 = str(row['WW']), str(row['W1']), str(row['W1'])
    regexps = {
        TAGS_INSTANCE.VETER: re.compile(r'ветер|ветр', re.I),
        TAGS_INSTANCE.SHKVAL: re.compile(r'шквал', re.I),
        TAGS_INSTANCE.METEL: re.compile(r'метел', re.I),
        TAGS_INSTANCE.DOZD: re.compile(r'дожд|ливен|морос', re.I),
        TAGS_INSTANCE.SNEG: re.compile(r'снег|снежн|круп|зерн', re.I),
        TAGS_INSTANCE.GRAD: re.compile(r'град', re.I),
        TAGS_INSTANCE.TUMAN: re.compile(r'туман|мгла', re.I),
        TAGS_INSTANCE.GOLOLED: re.compile(r'голол[её]д', re.I),
    }
    for tag, regexp in regexps.items():
        if regexp.search(WW) or regexp.search(W1) or regexp.search(W2):
            target.values[tag] = True

    # preprocessing nan
    VV_to_float = {
        'менее 0.05': 0.05,
        'менее 0.1': 0.1,
    }
    if row['VV'] in VV_to_float.keys():
        row['VV'] = VV_to_float[row['VV']]
    else:
        row['VV'] = float(row['VV'])

    # heuristics based on meteoinfo.ru
    if row['Ff'] >= 20.0 or row['ff10'] >= 25.0:  # ff10 and ff3
        target.values[TAGS_INSTANCE.VETER] = True
    if all([
        'снег' in str(row["E'"]).lower(),
        row['Ff'] >= 15.0,
        row['VV'] <= 0.5,
    ]):
        target.values[TAGS_INSTANCE.METEL] = True
    # if row['RRR'] >= 50.0 and row['tR'] <= 12:
    #     target.values[TAGS_INSTANCE.DOZD] = True

    return target.dict()
