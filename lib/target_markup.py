import json
import typing as tp

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
with open('tags.json', 'w') as f:
    json.dump(TAGS_INSTANCE.dict(), f, ensure_ascii=False)


class Target(BaseModel):
    VETER: bool = False
    SHKVAL: bool = False
    METEL: bool = False
    DOZD: bool = False
    SNEG: bool = False
    GRAD: bool = False
    TUMAN: bool = False
    GOLOLED: bool = False


def classify(row: pd.Series) -> tp.Dict:
    target = Target()

    # maybe check W1 and W2 not only WW
    if row['Ff'] >= 20.0 or row['ff10'] >= 25.0:  # ff10 and ff3
        target.VETER = True

    if 'шквал' in row['WW'].lower():
        target.SHKVAL = True

    if all([
        'снег' in row["E'"].lower(),
        row['Ff'] >= 15.0,
        row['VV'] <= 0.5,
    ]) or 'метель' in row['WW'].lower():
        target.METEL = True

    if row['RRR'] >= 50.0 and row['tR'] <= 12:  # check WW
        target.DOZD = True

    return target.dict()
