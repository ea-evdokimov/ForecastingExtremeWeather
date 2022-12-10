import re
import typing as tp

import numpy as np
import pandas as pd
from pydantic import BaseModel


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


def vv_preparation(x: tp.Union[str, float]) -> tp.Optional[float]:
    if isinstance(x, float) and np.isnan(x):
        return x
    try:
        return float(x)
    except:
        return float(re.match(r'\w+ ([0-9.]+)', x).group(1))


def sliding_window_features(df, columns, size):
    df_new = df.sort_index(ascending=False).copy()
    for column in columns:
        for shift in range(1, size + 1):    
            feature_values = df[column].values
            feature_values_shifted = np.roll(feature_values, -shift)
            df_new[column + '_' + str(shift)] = feature_values_shifted    
    return df_new.iloc[:-size]


class FeatureInterpolator:
    max_nan_percent = 0.5
    max_cons_nan_percent = 0.01

    def _longest_na_seq(self, col: pd.Series) -> int:
        na_groups = col.notna().cumsum()[col.isna()]
        lens_cons_na = na_groups.groupby(na_groups).agg(len)
        longest_na_len = lens_cons_na.max()
        return 0 if longest_na_len is np.nan else longest_na_len

    def interpolate_column(self, s: pd.Series) -> pd.Series:
        return s.interpolate(method='slinear')

    def get_columns(self, df: pd.DataFrame) -> tp.List[str]:
        na_sum = df.isna().sum()
        max_nan = int(len(df) * FeatureInterpolator.max_nan_percent / 100)
        max_cons_nan = int(len(df) * FeatureInterpolator.max_cons_nan_percent / 100)
        cand_cols = [c for c in df.columns if na_sum[c] <= max_nan]
        columns = []
        for col in cand_cols:
            m_len = self._longest_na_seq(df[col])
            if m_len != 0 and m_len < max_cons_nan:
                columns.append(col)
        return columns
