import typing as tp
from enum import Enum

import pandas as pd
from loguru import logger
from tqdm import tqdm

import feature_preparation
import target_markup

# Now is ALL!
ALLOW_COLUMNS = ['local_time', 'T', 'Po', 'P', 'Pa', 'U', 'DD', 'Ff', 'ff10', 'ff3', 'N',
                 'WW', 'W1', 'W2', 'Tn', 'Tx', 'Cl', 'Nh', 'H', 'Cm', 'Ch', 'VV', 'Td',
                 'RRR', 'tR', 'E', 'Tg', 'E\'', 'sss', 'station_id']


class Mode(str, Enum):
    UNPROCESSED = 'unprocessed'
    PROCESSED = 'processed'


# INPLACE METHODS ONLY
pipeline = [
    feature_preparation.local_time_pipeline,
    feature_preparation.dd_pipeline,
    feature_preparation.ch_pipeline
]


def prepare_dataset(dataset: pd.DataFrame,
                    allow_columns: tp.Optional[tp.List] = None,
                    mode: tp.Union[Mode, str] = Mode.PROCESSED) -> \
        tp.Tuple[pd.DataFrame, pd.DataFrame]:
    allow_columns = allow_columns or ALLOW_COLUMNS

    tqdm.pandas()
    # Step 1: markup target
    logger.info('Markup target...')
    target = pd.json_normalize(dataset.progress_apply(target_markup.classify, axis=1))

    # Step 2: prepare features
    prepared_dataset = dataset.copy()
    logger.info(f'Current mode of preparation is {mode}.')
    if mode == Mode.UNPROCESSED:
        logger.info('Done (unprocessed)!')
        return prepared_dataset, target

    for step in tqdm(pipeline):
        logger.info(f'Step: {step.__name__}')
        prepared_dataset = step(prepared_dataset)

    logger.info('Done (processed)!')
    return prepared_dataset, target
