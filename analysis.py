import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# pd.set_option('display.max_colwidth', -1)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 150)

def get_tweet_data():
    df = pd.read_csv("data/twitter_data/tweets.csv", delimiter=";", header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    print(df.head())
    return df


def get_donation_data():
    df = pd.read_csv("data/donation_data/team-tree-donation-data.csv", delimiter="	", header=0)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S %p')
    print(df.head())
    return df


tweet_df = get_tweet_data()
donation_df = get_donation_data()


plt.figure()
sns.distplot(tweet_df['date'])
# sns.distplot(donation_df['time'])
plt.show()
