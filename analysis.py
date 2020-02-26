import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_tweet_data():
    df = pd.read_csv("data/twitter_data/tweets.csv", delimiter=";", header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    return df


def get_donation_data():
    df = pd.read_csv("data/donation_data/team-tree-donation-data.csv", delimiter="	", header=0)
    df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y %H:%M:%S %p')
    return df


tweet_df = get_tweet_data()
# donation_df = get_donation_data()


plt.figure()
sns.distplot(tweet_df['date'])
# sns.distplot(donation_df['time'])
plt.show()
