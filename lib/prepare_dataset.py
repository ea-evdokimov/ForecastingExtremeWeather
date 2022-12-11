import typing as tp
from enum import Enum

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

import feature_preparation
import target_markup

# Now is ALL!
ALLOW_COLUMNS = ['local_time', 'T', 'Po', 'P', 'Pa', 'U', 'DD', 'Ff', 'ff10', 'ff3', 'N',
                 'WW', 'W1', 'W2', 'Tn', 'Tx', 'Cl', 'Nh', 'H', 'Cm', 'Ch', 'VV', 'Td',
                 'RRR', 'tR', 'E', 'Tg', 'E\'', 'sss', 'station_id']


NUMERIC_COLUMNS_PREPARED = [
    "T", "P", "U", "Ff", "Po", "N", "Td",
    "Pa", "ff10", "ff3", "tR", "H", 
    "RRR", "sss", "dd_x_rad", "dd_y_rad", "Tn_isnan", "Tx_isnan",
    "dd_isnan", "dd_changed", "sss", "VV"
]


CATEGORIAL_COLUMNS_PREPARED = [
    "Cl", "Cm", "Ch", "Nh"
]


class Mode(str, Enum):
    UNPROCESSED = 'unprocessed'
    PROCESSED = 'processed'


# INPLACE METHODS ONLY
pipeline = [
    feature_preparation.local_time_pipeline,
    feature_preparation.dd_pipeline,
    feature_preparation.ch_pipeline,
    feature_preparation.cm_pipeline,
    feature_preparation.cl_pipeline,
    feature_preparation.vv_pipeline
]


def fill_nans(dataset: pd.DataFrame, 
              allow_columns: tp.Optional[tp.List] = None, 
              mode=tp.Union[Mode, str]) -> pd.DataFrame:
    if mode == Mode.UNPROCESSED:
        return dataset
    
    simple_int_cand = ["T", "PP", "VV", "U", "Ff", "Po", "N", "Td"]
    if allow_columns is not None:
        simple_int_cand = list(set(simple_int_cand).intersection(set(allow_columns)))
    for col in simple_int_cand:
        dataset[col] = feature_preparation.SimpleFeatureInterpolator.interpolate_column(dataset[col])

    # PA
    dataset = feature_preparation.pa_fill_na(dataset)
    # TN and TX
    dataset = feature_preparation.tn_preparation_fill_na(dataset)
    dataset = feature_preparation.tx_preparation_fill_na(dataset)
    # Tg
    dataset = feature_preparation.tg_preparation_fill_na(dataset)
    # ff3 and ff10
    dataset = feature_preparation.ff3_fill_na(dataset)
    dataset = feature_preparation.ff10_fill_na(dataset)

    dataset.drop(['WW', 'WW1', 'WW2'], inplace=True)

    dataset = feature_preparation.string_fill_na(dataset, ['Cl', 'Cm', 'Ch', 'E', 'E\''])

    dataset = feature_preparation.n_pipeline_with_na_fill(dataset)
    dataset = feature_preparation.nh_pipeline_with_na_fill(dataset)
    dataset = feature_preparation.h_pipeline_with_na_fill(dataset)
    dataset = feature_preparation.sss_pipeline_with_fill(dataset)

    return dataset


def make_target(dataset: pd.DataFrame) -> pd.DataFrame:
    target = pd.json_normalize(dataset.progress_apply(target_markup.classify, axis=1))
    return target


def prepare_dataset(dataset: pd.DataFrame,
                    allow_columns: tp.Optional[tp.List] = None,
                    mode: tp.Union[Mode, str] = Mode.PROCESSED) -> \
        tp.Tuple[pd.DataFrame, pd.DataFrame]:
    allow_columns = allow_columns or ALLOW_COLUMNS

    tqdm.pandas()
    
    # Step 1: prepare features
    prepared_dataset = dataset.copy()
    logger.info(f'Current mode of preparation is {mode}.')
    if mode == Mode.UNPROCESSED:
        logger.info('Done (unprocessed)!')
        target = make_target(prepared_dataset)
        logger.info('Markup target...')
        return prepared_dataset, target
    
    # Step 2: sort features
    prepared_dataset.sort_values(['station_id', 'local_time'], ascending=False, 
                                inplace=True)
    prepared_dataset = prepared_dataset.reset_index(drop=True)
    target = make_target(prepared_dataset)

    # Step 3: prepare features  
    for step in tqdm(pipeline):
        logger.info(f'Step: {step.__name__}')
        prepared_dataset = step(prepared_dataset)
    
    logger.info('Filling nans!')
    prepared_dataset = fill_nans(prepared_dataset, allow_columns=allow_columns, mode=mode)

    logger.info('Done (processed)!')
    return prepared_dataset, target
