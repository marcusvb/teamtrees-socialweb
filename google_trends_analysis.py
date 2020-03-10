import pandas as pd
from analysis import get_correlation_data

def get_google_data():
    df = pd.read_csv("data/google_trends_data/multiTimeline.csv", delimiter=",", header=0)
    df['date'] = pd.to_datetime(df['Day'], infer_datetime_format=True)
    df['count'] = df['teamtrees: (Worldwide)']
    return df

google_df = get_google_data()

get_correlation_data(google_df)