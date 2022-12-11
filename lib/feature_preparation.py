import re
import typing as tp

from tqdm.auto import tqdm

tqdm.pandas()
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


def dd_pipeline(df: pd.DataFrame, column_name='DD') -> pd.DataFrame:
    dd_values = pd.json_normalize(df.DD.progress_map(dd_preparation)).add_prefix('dd_')
    df.drop(columns=[column_name], inplace=True)
    return pd.merge(df, dd_values, left_index=True, right_index=True, copy=False)


def local_time_pipeline(df: pd.DataFrame, column_name='local_time') -> pd.DataFrame:
    '''
    :param df:
    :param column_name:
    :return: df with columns contained hours
    '''
    df[column_name] = pd.to_datetime(df[column_name])
    return df


def vv_preparation(x: tp.Union[str, float]) -> tp.Optional[float]:
    if isinstance(x, float) and np.isnan(x):
        return x
    try:
        return float(x)
    except:
        return float(re.match(r'\w+ ([0-9.]+)', x).group(1))


def vv_pipeline(df: pd.DataFrame, column_name='VV') -> pd.DataFrame:
    df[column_name] = df[column_name].progress_map(vv_preparation)
    return df


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
    'Перисто-кучевые одни или перисто-кучевые, сопровождаемые перистыми или перисто-слоистыми, либо те и другие, но перисто-кучевые преобладают среди них.': 'Перисто-кучевые+',
    'Перисто-слоистые, не распространяющиеся по небу и не покрывающие его полностью.': 'Перисто-слоистые-50',
    'Перисто-слоистые, покрывающие все небо.': 'Перисто-слоистые-100',
    'Перистые (часто в виде полос) и перисто-слоистые, распространяющиеся по небу и в целом обычно уплотняющиеся, но сплошная пелена поднимается над горизонтом менее чем на 45°.': 'Перисто-слоистые-пелена-<45',
    'Перистые (часто в виде полос) и перисто-слоистые, распространяющиеся по небу и в целом обычно уплотняющиеся; сплошная пелена, поднимающаяся над горизонтом выше 45°, не покрывает всего неба.': 'Перисто-слоистые-пелена->45',
    'Перистые когтевидные или нитевидные или первые и вторые, распространяющиеся по небу и в целом обычно уплотняющиеся.': 'Нитевидные-по-небу',
    'Перистые нитевидные, иногда когтевидные, не распространяющиеся по небу.': 'Нитевидные-не-по-небу',
    'Перистые плотные в виде клочьев или скрученных склонов, количество которых обычно не увеличивается, иногда могут казаться остатками верхней части кучево-дождевых; или перистые башенкообразные, или перистые хлопьевидные.': 'Перистые-плотные-остатки',
    'Перистые плотные, образовавшиеся от кучево-дождевых.': 'Перистые-плотные',
    'Перистых, перисто-кучевых или перисто-слоистых нет.': 'Нет'
}

cm_mapper = {
    'Высококучевые башенкообразные или хлопьевидные.': 'Высококучевые-башенкообразные',
    'Высококучевые при хаотическом виде неба, обычно расположенные на нескольких уровнях.': 'Высококучевые-хаот',
    'Высококучевые просвечивающие, полосами, либо один или несколько слоев высококучевых просвечивающих, распространяющихся по небу; эти высококучевые в целом уплотняются.': 'Высококучевые-просвет-уплотн',
    'Высококучевые просвечивающие, расположенные на одном уровне.': 'Высококучевые-просвет-один-ур',
    'Высококучевые, образовавшиеся из кучевых (или кучево-дождевых).': 'Высококучевые-дождь',
    'Высококучевые, просвечивающие или плотные в двух или более слоях или высококучевые плотные в одном слое, не распространяющиеся но небу, либо высококучевые с высокослоистыми или слоисто-дождевыми.': 'Высококучевые-просвет',
    'Высококучевых, высокослоистых или слоисто-дождевых облаков нет.': 'Нет',
    'Высокослоистые непросвечивающие или слоисто-дождевые.': 'Высокослоистые-непросвет',
    'Высокослоистые просвечивающие.': 'Высокослоистые-просвет',
    'Клочья (часто чечевицеобразные) высококучевых просвечивающих, непрерывно изменяющихся; расположены на одном или более уровнях.': 'Клочья'
}

cl_mapper = {
    'Кучево-дождевые волокнистые (часто с наковальней), либо с кучево-дождевыми лысыми, кучевыми, слоистыми, разорванно-дождевыми, либо без них.': 'Кучево-дождевые-волокнистые',
    'Кучево-дождевые лысые с кучевыми, слоисто-кучевыми или слоистыми, либо без них.': 'Кучево-дождевые-лысые',
    'Кучевые и слоисто-кучевые (но не слоисто-кучевые, образовавшиеся из кучевых), основания расположены на разных уровнях.': 'Кучевые-раз-уровни',
    'Кучевые плоские или кучевые разорванные, или те и другие вместе, не относящиеся к облакам плохой погоды.': 'Кучевые-не-плохая-погода',
    'Кучевые средние или мощные или вместе с кучевыми разорванными, или с кучевыми плоскими, или со слоисто-кучевыми, либо без них; основания всех этих облаков расположены на одном уровне.': 'Кучевые-средние-мощные-один-уровень',
    'Слоисто-кучевые, образовавшиеся из кучевых.': 'Слоисто-кучевые-из-кучевых',
    'Слоисто-кучевые, образовавшиеся не из кучевых.': 'Слоисто-кучевые-не-из-кучевых',
    'Слоисто-кучевых, слоистых, кучевых или кучево-дождевых облаков нет.': 'Нет',
    'Слоистые разорванные или кучевые разорванные облака плохой погоды, либо те и другие вместе (разорванно-дождевые); обычно расположены под слоистыми или слоисто-дождевыми облаками.': 'Слоистые-разорванные-плохая-погода',
    'Слоистые туманообразные или слоистые разорванные, либо те и другие, но не относящиеся к облакам плохой погоды.': 'Слоистые-разорванные-не-плохая'
}


def ch_preparation(description: tp.Optional[str]) -> tp.Optional[str]:
    return ch_mapper[description] if isinstance(description, str) else None


def cm_preparation(description: tp.Optional[str]) -> tp.Optional[str]:
    return cm_mapper[description] if isinstance(description, str) else None


def cl_preparation(description: tp.Optional[str]) -> tp.Optional[str]:
    return cl_mapper[description] if isinstance(description, str) else None


def ch_pipeline(df: pd.DataFrame, column_name='Ch') -> pd.DataFrame:
    df[column_name] = df[column_name].progress_map(ch_preparation)
    return df


def cm_pipeline(df: pd.DataFrame, column_name='Cm') -> pd.DataFrame:
    df[column_name] = df[column_name].progress_map(cm_preparation)
    return df


def cl_pipeline(df: pd.DataFrame, column_name='Cl') -> pd.DataFrame:
    df[column_name] = df[column_name].progress_map(cl_preparation)
    return df


def parse_clouds(x: tp.Optional[str], no_clouds_default: float) -> tp.Optional[float]:
    if not isinstance(x, str):
        return None
    if x.lower().find('облаков нет') != -1:
        return no_clouds_default
    groups = re.findall('[0-9]+', x)
    if len(groups) == 1:
        return float(groups[0])
    if len(groups) == 2:
        return (float(groups[0]) + float(groups[1])) / 2
    return None


def pa_fill_na(df: pd.DataFrame, column_name='Pa') -> pd.DataFrame:
    g = df.groupby('station_id')
    t_diff = g.local_time.diff(-1).fillna(pd.Timedelta('03:00:00')) / pd.Timedelta('03:00:00')
    p_diff = g.P.diff(-1).fillna(0)
    pa = p_diff / t_diff
    df[column_name] = df[column_name].fillna(pa).round(4)
    return df


def sss_RRR_tr_preparation(value: tp.Union[str, float]) -> tp.Optional[float]:
    if value is None or isinstance(value, float) and np.isnan(value):
        return None

    if isinstance(value, str):
        numbers = re.findall('\d+\.?\d*', value)
        if not numbers:
            return None
        return np.mean([float(n) for n in numbers])

    if isinstance(value, (int, float)):
        return float(value)

    return None


def rolling_window_na_fill(df: pd.DataFrame, column_name: str, window_size: int, min_periods: int = 1) -> pd.DataFrame:
    rolling = df[column_name].rolling(window=window_size, min_periods=min_periods, center=True).mean()
    df[column_name] = df[column_name].fillna(rolling)
    return df

def sss_pipeline_with_fill(df: pd.DataFrame, column_name: str='sss') -> pd.DataFrame:
    df[column_name] = df[column_name].map(sss_RRR_tr_preparation)
    copy = df[column_name].copy()
    copy.loc[:] = np.nan
    copy[df['T'] > 12] = 0
    df[column_name] = df[column_name].fillna(copy)

    # зимой снег быстро не тает, можно расширить интерполяцию 
    copy = df[column_name].copy().interpolate(method='linear', limit=40, limit_direction='both')
    copy = rolling_window_na_fill(copy.to_frame(), column_name, 48, 2)
    copy.sss[df['T'] > 0] = np.nan
    df[column_name] = df[column_name].fillna(copy[column_name])

    # иначе интерполируем не больше чем на 3 дня
    df = rolling_window_na_fill(df, column_name, 16, 4)
    df[column_name] = df[column_name].interpolate(method='slinear', limit=8, limit_direction='both')
    df[column_name] = df[column_name].fillna(0)
    return df

class FloatWithNanModel(BaseModel):
    isnan: bool
    val: float

def _float_with_nan_prep(d: tp.Optional[float], default: float) -> FloatWithNanModel:
    if not isinstance(d, float) or np.isnan(d):
        return FloatWithNanModel(isnan=True, val=default)
    return FloatWithNanModel(isnan=False, val=d)


def _float_with_replace(df: pd.DataFrame, col: str, 
                        default: pd.Series, prefix_name: tp.Optional[str] = None) -> pd.DataFrame:
    def_name = default.name    
    joined = default.to_frame().join(df[col])
    data = joined.apply(lambda x: _float_with_nan_prep(x[col], x[def_name]).dict(), axis=1)
    if prefix_name is None:
        prefix_name = col + "_"
    return pd.json_normalize(data).add_prefix(prefix_name)


def _float_with_replace_merge(df: pd.DataFrame, col: str, 
                              default: pd.Series, 
                              prefix_name: tp.Optional[str] = None) -> pd.DataFrame:
    vals = _float_with_replace(df, col, default, prefix_name=prefix_name)
    df.drop(columns=[col], inplace=True)
    return pd.merge(df, vals, left_index=True, right_index=True, copy=False)
    

def tn_preparation_fill_na(df: pd.DataFrame, col: str = 'Tn', t_col: str = 'T', 
                   window_size: int = 4, prefix_name: str = 'Tn_') -> pd.DataFrame:
    roll_mins_12_h = df[t_col].rolling(window=window_size, min_periods=1, center=False).min()
    return _float_with_replace_merge(df, col, roll_mins_12_h, prefix_name=prefix_name)


def tx_preparation_fill_na(df: pd.DataFrame, col: str = 'Tx', t_col: str = 'T', 
                window_size: int = 4, prefix_name: str = 'Tx_') -> pd.DataFrame:
    roll_maxs_12_h = df[t_col].rolling(window=window_size, min_periods=1, center=False).max()
    return _float_with_replace_merge(df, col, roll_maxs_12_h, prefix_name=prefix_name)


def tg_preparation_fill_na(df: pd.DataFrame, col: str = 'Tg', prefix_name='Tg_') -> pd.DataFrame:
    last_cold_temp = df[col].fillna(method='bfill').fillna(0).rename(col + "_bfilled")
    return _float_with_replace_merge(df, col, last_cold_temp, prefix_name=prefix_name)

def ff3_fill_na(df: pd.DataFrame, col: str = 'ff3', f_col: str = 'Ff', prefix_name = 'ff3_'):
    return _float_with_replace_merge(df, col, df[f_col], prefix_name=prefix_name)


def ff10_fill_na(df: pd.DataFrame, col: str = 'ff10', f_col: str = 'Ff', prefix_name = 'ff10_'):
    return _float_with_replace_merge(df, col, df[f_col], prefix_name=prefix_name)


def n_pipeline_with_na_fill(df: pd.DataFrame, column_name='N') -> pd.DataFrame:
    default_height = 0
    lambd = lambda x: parse_clouds(x, default_height)
    df[column_name] = df[column_name].progress_map(lambd)
    default = df[column_name].fillna(default_height).rename(column_name + "_def") 
    return _float_with_replace_merge(df, column_name, default, prefix_name='N_')


def h_pipeline_with_na_fill(df: pd.DataFrame, column_name='H') -> pd.DataFrame:
    default_height = 10000
    lambd = lambda x: parse_clouds(x, default_height)
    df[column_name] = df[column_name].progress_map(lambd)
    default = df[column_name].fillna(default_height).rename(column_name + "_def")    
    return _float_with_replace_merge(df, column_name, default, prefix_name='H_')


def nh_pipeline_with_na_fill(df: pd.DataFrame, column_name='Nh') -> pd.DataFrame:
    default_vision = 0
    lambd = lambda x: parse_clouds(x, default_vision)
    df[column_name] = df[column_name].progress_map(lambd)
    default = df[column_name].fillna(default_vision).rename(column_name + "_def") 
    return _float_with_replace_merge(df, column_name, default, prefix_name='Nh_')


def string_fill_na(df: pd.DataFrame, column_names: tp.List[str], default_fill='Нет данных!') -> pd.DataFrame:
    for col in column_names:
        df[col].fillna(default_fill, inplace=True)
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
