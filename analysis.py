import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set()

# pd.set_option('display.max_colwidth', -1)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 150)

def get_tweet_data():
    df = pd.read_csv("data/twitter_data/tweets.csv", delimiter=";", header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

    return df

def get_donation_data(secondly=False):
    # Read the 10s donation data from the website
    if secondly:
        df = pd.read_csv("data/donation_data/team-trees-10second.csv", delimiter=",", header=0)
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        return df

    # Read old data-set a couple of hours per tweet
    else:
        df = pd.read_csv("data/donation_data/team-tree-donation-data.csv", delimiter="	", header=0)
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S %p')

        # Cleanup rate of funding
        df['rate_of_funding'] = df['rate_of_funding'].str.replace("?", "0")
        df['rate_of_funding'] = df['rate_of_funding'].str.replace("/min", "")
        df["rate_of_funding"] = pd.to_numeric(df["rate_of_funding"])
        # df["rate_of_funding"] = df["rate_of_funding"]/df["rate_of_funding"].max() # scale rate funding between 0 and 1

        # Cleanup donation amount
        df['raised_capital'] = df['raised_capital'].str.replace("$", "")
        df['raised_capital'] = df['raised_capital'].str.replace(",", "")
        df['raised_capital'] = df['raised_capital'].str.replace("/min", "")
        df["raised_capital"] = pd.to_numeric(df["raised_capital"])
        return df

def get_tweet_count_data(timeunit):
    df = pd.read_csv('data/twitter_data/count_' + 'per_' + timeunit + '_tweets.csv', delimiter=",", header=0)
    return df

def get_donation_rate_data(timeunit):
    df = pd.read_csv('data/donation_data/av_rate_' + 'per_' + timeunit + '_donations.csv', delimiter=",", header=0)
    return df


def calc_correlation(social_data, donation_data, begin_date, end_date):
    period = (social_data['date'] > begin_date) & (social_data['date'] <= end_date)
    correlation = np.corrcoef(social_data.loc[period]['count'], donation_data.loc[period]['av_rate'])
    return correlation


# get data in raw form
tweet_df = get_tweet_data()
donation_df = get_donation_data()

# read in data per time unit
tweets_per_day = get_tweet_count_data('day')
av_donation_rate_per_day = get_donation_rate_data('day')

# plots
plt.bar(donation_df['date'], donation_df['raised_capital'])
plt.xlabel("Date")
plt.ylabel("Raised capital ($)")

fig, ax1 = plt.subplots()
ax1.set_xlabel('date')

color = 'tab:red'
ax1.set_ylabel('Frequency of tweets', color=color)
plt.hist(tweet_df['date'], alpha=0.5, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('rate of funding', color=color)  # we already handled the x-label with ax1
plt.bar(donation_df['date'], donation_df['rate_of_funding'], alpha=1, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

'''Doesn't work at the moment as the datasets need to be of the same size (so per day)'''
# calculate covariance of tweets per day and rate of funding
start_date = '2019-10-25'
end_date = '2020-02-03'
print(calc_correlation(tweets_per_day, av_donation_rate_per_day, start_date, end_date))
