import typing as tp

import pandas as pd

import target_markup

# Now is ALL!
ALLOW_COLUMNS = ['local_time', 'T', 'Po', 'P', 'Pa', 'U', 'DD', 'Ff', 'ff10', 'ff3', 'N',
                 'WW', 'W1', 'W2', 'Tn', 'Tx', 'Cl', 'Nh', 'H', 'Cm', 'Ch', 'VV', 'Td',
                 'RRR', 'tR', 'E', 'Tg', 'E\'', 'sss', 'station_id']


def prepare_dataset(dataset: pd.DataFrame,
                    allow_columns: tp.Optional[tp.List] = None) -> \
        tp.Tuple[pd.DataFrame, pd.DataFrame]:
    allow_columns = allow_columns or ALLOW_COLUMNS
    # Step 1: markup target
    target = target_markup.classify(dataset)

    # Step 2: prepare features
    prepared_dataset = dataset.copy()
    prepared_dataset = prepared_dataset[allow_columns]

    return prepared_dataset, target