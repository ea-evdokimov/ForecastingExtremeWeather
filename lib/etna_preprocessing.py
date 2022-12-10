import pandas as pd


# st = weather_train[weather_train.station_id == 34214].loc[:, ['local_time', 'N']]
# st.loc[:, 'timestamp'] = st.local_time
# st.loc[:, 'segment'] = 'main'
# st.loc[:, 'target'] = st.N.apply(parse_N)
# st = st.loc[:, ['timestamp', 'segment', 'target']]
# st.set_index('timestamp').reset_index()
# st = to_min_interval(st, 'timestamp')
# st = TSDataset.to_dataset(st)
# st = TSDataset(st, freq="3H")
#  
def to_min_interval(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    s = df[col_name]
    min_interval = s.diff(-1).fillna(method='bfill').min()
    new_df = df.set_index(col_name)
    index = pd.date_range(start=s.min(), end=s.max(), freq=min_interval, name=col_name)[::-1]
    return new_df.reindex(index, fill_value=None).reset_index()