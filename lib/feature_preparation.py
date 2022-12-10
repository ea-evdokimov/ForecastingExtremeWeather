import re
import typing as tp

from tqdm import tqdm

tqdm.pandas()
import numpy as np
import pandas as pd
from pydantic import BaseModel
from datetime import datetime


class WindDescription(BaseModel):
    isnan: bool
    changed: bool
    x_rad: float
    y_rad: float


def _dd_preparation(description: tp.Optional[str]) -> WindDescription:
    '''
    :param description: string description of wind
    :return: WindDescription object
    '''
    if description is None or not isinstance(description, str):
        return WindDescription(isnan=True, changed=False, x_rad=0., y_rad=0.)

    prepared_description = description.lower()

    if 'переменное направление'.lower() in prepared_description:
        return WindDescription(isnan=False, changed=True, x_rad=0., y_rad=0.)

    if 'штиль' in prepared_description or 'безветрие' in prepared_description:
        return WindDescription(isnan=False, changed=False, x_rad=0., y_rad=0.)

    # 'восток определяется неоднозначно, поэтому определим его далее'
    dest_to_rad = {'восток': None,
                   'север': np.pi / 2,
                   'запад': np.pi,
                   'юг': 3 * np.pi / 2}
    counts = {k: prepared_description.count(k) for k in dest_to_rad if prepared_description.count(k) > 0}
    dest_to_rad['восток'] = 2 * np.pi if 'восток' in counts and 'юг' in counts else 0.

    total_count = sum((v for v in counts.values()))
    if total_count < 3:
        res_angle = sum((c * dest_to_rad[dest] for dest, c in counts.items())) / total_count
    else:
        mid_angle = sum((dest_to_rad[dest] for dest in counts)) / 2
        add_angle = dest_to_rad[[k for k, v in counts.items() if v == 2][0]]
        res_angle = (mid_angle + add_angle) / 2

    return WindDescription(isnan=False, changed=False, x_rad=np.cos(res_angle), y_rad=np.sin(res_angle))


def dd_preparation(description: tp.Optional[str]) -> tp.Dict:
    return _dd_preparation(description).dict()


def dd_pipeline(df: pd.DataFrame, column_name='DD') -> pd.DataFrame:
    dd_values = pd.json_normalize(df.DD.progress_map(dd_preparation)).add_prefix('dd_')
    df.drop(columns=[column_name], inplace=True)
    return pd.merge(df, dd_values, left_index=True, right_index=True, copy=False)


def _local_time_to_timestamp(dtime: str) -> int:
    return int(datetime.strptime(dtime, '%d.%m.%Y %H:%M').timestamp()) // 3600


def local_time_pipeline(df: pd.DataFrame, column_name='local_time') -> pd.DataFrame:
    '''
    :param df:
    :param column_name:
    :return: df with columns contained hours
    '''
    df[column_name] = df[column_name].progress_map(_local_time_to_timestamp)
    return df


def vv_preparation(x: tp.Union[str, float]) -> tp.Optional[float]:
    if isinstance(x, float) and np.isnan(x):
        return x
    try:
        return float(x)
    except:
        return float(re.match(r'\w+ ([0-9.]+)', x).group(1))


def sliding_window_features(df, feature_columns, target_columns, size):
    df_new = df.sort_index(ascending=False).copy()

    columns = target_columns + feature_columns
    for column in columns:
        for shift in range(1, size + 1):
            feature_values = df_new[column].values
            feature_values_shifted = np.roll(feature_values, -shift)
            df_new[column + '_' + str(shift)] = feature_values_shifted

    y = df_new[target_columns]

    df_new = df_new.drop(columns, axis=1)

    return df_new.iloc[:-size], y.iloc[:-size]


ch_mapper = {
    'Перисто-кучевые одни или перисто-кучевые, сопровождаемые перистыми или перисто-слоистыми, либо те и другие, но перисто-кучевые преобладают среди них.': 'Кучевые',
    'Перисто-слоистые, не распространяющиеся по небу и не покрывающие его полностью.': 'Перисто-слоистые',
    'Перисто-слоистые, покрывающие все небо.': 'Перисто-слоистые',
    'Перистые (часто в виде полос) и перисто-слоистые, распространяющиеся по небу и в целом обычно уплотняющиеся, но сплошная пелена поднимается над горизонтом менее чем на 45°.': 'Перисто-слоистые',
    'Перистые (часто в виде полос) и перисто-слоистые, распространяющиеся по небу и в целом обычно уплотняющиеся; сплошная пелена, поднимающаяся над горизонтом выше 45°, не покрывает всего неба.': 'Перисто-слоистые',
    'Перистые когтевидные или нитевидные или первые и вторые, распространяющиеся по небу и в целом обычно уплотняющиеся.': 'Нитевидные',
    'Перистые нитевидные, иногда когтевидные, не распространяющиеся по небу.': 'Нитевидные',
    'Перистые плотные в виде клочьев или скрученных склонов, количество которых обычно не увеличивается, иногда могут казаться остатками верхней части кучево-дождевых; или перистые башенкообразные, или перистые хлопьевидные.': 'Перистые плотные',
    'Перистые плотные, образовавшиеся от кучево-дождевых.': 'Перистые плотные',
    'Перистых, перисто-кучевых или перисто-слоистых нет.': 'Нет'
}


def ch_preparation(description: tp.Optional[str]) -> str:
    return ch_mapper[description] if isinstance(description, str) else description


def ch_pipeline(df: pd.DataFrame, column_name='Ch') -> pd.DataFrame:
    df[column_name] = df[column_name].progress_map(ch_preparation)
    return df


class SimpleFeatureInterpolator:
    max_nan_percent = 1.5
    max_cons_nan_percent = 0.01

    @staticmethod
    def _longest_na_seq(col: pd.Series) -> int:
        na_groups = col.notna().cumsum()[col.isna()]
        lens_cons_na = na_groups.groupby(na_groups).agg(len)
        longest_na_len = lens_cons_na.max()
        return 0 if longest_na_len is np.nan else longest_na_len

    @staticmethod
    def interpolate_column(s: pd.Series) -> pd.Series:
        return s.interpolate(method='slinear')

    @staticmethod
    def get_columns(df: pd.DataFrame) -> tp.List[str]:
        na_sum = df.isna().sum()
        max_nan = int(len(df) * SimpleFeatureInterpolator.max_nan_percent / 100)
        max_cons_nan = int(len(df) * SimpleFeatureInterpolator.max_cons_nan_percent / 100)
        cand_cols = [c for c in df.columns if na_sum[c] <= max_nan]
        columns = []
        for col in cand_cols:
            m_len = SimpleFeatureInterpolator._longest_na_seq(df[col])
            if m_len != 0 and m_len < max_cons_nan:
                columns.append(col)
        return columns

