import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

sns.set()

# pd.set_option('display.max_colwidth', -1)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 150)


def get_tweet_data():
    df = pd.read_csv("data/twitter_data/tweets.csv", delimiter=";", header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    return df

def get_donation_data():
    df = pd.read_csv('data/donation_data/parsed-team-trees-10second.csv', header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    return df

def parse_donation_data():
    # Read the 10s donation data from the website
    # Src: https://vps.natur-kultur.eu/trees.html
    df = pd.read_csv("data/donation_data/team-trees-10second.csv", delimiter=",", header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    print(df["date"])
    df['date'] = df['date'].apply(lambda x: x.replace(tzinfo=None))
    print(df['date'])
    df['raised_capital'] = (df['amount']/20000000)*100

    rate_of_fundings = [0]
    for i in range(0, len(df['date'])-1):
        current_date = df['date'].values[i]
        current_capital = df['amount'].values[i]

        next_date = df['date'].values[i+1]
        next_capital = df['amount'].values[i+1]

        time_diff = (next_date-current_date) / np.timedelta64(1, 's')
        delta_diff = next_capital - current_capital

        min_rate = (delta_diff/time_diff)*60
        rate_of_fundings.append(min_rate)

    df['rate_of_funding'] = rate_of_fundings

    df.to_csv('data/donation_data/parsed-team-trees-10second.csv', header=True)

    return df


def get_tweet_count_data(timeunit):
    df = pd.read_csv('data/twitter_data/count_' + 'per_' + timeunit + '_tweets.csv', delimiter=",", header=0)
    return df


def get_donation_rate_data(timeunit):
    df = pd.read_csv('data/donation_data/av_rate_' + 'per_' + timeunit + '_donations.csv', delimiter=",", header=0)
    return df


def calc_correlation(social_data, donation_data, begin_date, end_date, sentiment):
    period_social = (social_data['date'] > begin_date) & (social_data['date'] <= end_date)
    period_donation = (donation_data['date'] > begin_date) & (donation_data['date'] <= end_date)

    if not sentiment:
        correlation = np.corrcoef(social_data.loc[period_social]['count'], donation_data.loc[period_donation]['av_rate'])
    else:
        correlation = np.corrcoef(social_data.loc[period_social]['n_positive'], donation_data.loc[period_donation]['av_rate'])
    return correlation


def plot_data(tweet_df, donation_df):
    # read in data per time unit

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


def get_correlation_data(file1='regular_twitter_data', file2='donation_rate_data', sentiment=False):
    if isinstance(file1, str):
        if file1 == 'regular_twitter_data':
            tweets_per_day = get_tweet_count_data('day')
        else:
            tweets_per_day = pd.read_csv(file1, delimiter=",", header=0)
    else:
        tweets_per_day = file1

    if isinstance(file2, str):
        if file2 == 'donation_rate_data':
            av_donation_rate_per_day = get_donation_rate_data('day')
        else:
            av_donation_rate_per_day = pd.read_csv(file2, delimiter=",", header=0)
    else:
        av_donation_rate_per_day = file2


    # calculate covariance of tweets per day and rate of funding
    start_date = '2019-10-25'
    end_date = '2020-02-03'
    print(calc_correlation(tweets_per_day, av_donation_rate_per_day, start_date, end_date, sentiment))


def fit_log_model_analysis(donation_df):
    def logFunc(x, a, b):
        return a + b * np.log(x)

    # Split the donations into amounts, and add a cumsum column
    donation_data_delta = donation_df.diff(periods=1, axis=0)
    donation_df['amount'] = donation_data_delta['amount']
    donation_df['cumsum'] = donation_df['amount'].cumsum()

    # Out Y data is all but the first cumsum as it's nan
    y_data = donation_df['cumsum'].to_numpy()[1:]


    # special date scaling... day 0 starts at 1, so instead of large numbers we normalize to the data and to make logarithm happy
    x_dates = mdates.date2num(donation_df['date'])
    x_dates = x_dates - (x_dates[0] - 1)

    # fit the log model to campaign data corresponding to
    # 1-12-2019
    small_x_data = x_dates[0:30500]
    small_y_data = y_data[0:30500]
    popt, _ = curve_fit(logFunc, small_x_data, small_y_data)

    fig, ax0 = plt.subplots()

    ax0.set_xlabel('date')
    ax0.set_ylabel('cumulative donations in $')
    ax0.plot(donation_df['date'], donation_df['amount'].cumsum(), linestyle="--", label="real data", color="blue")
    ax0.tick_params(axis='y')
    ax0.tick_params(axis='x')
    ax0.axhline(20000000, label="20mil goal", color="yellow")
    ax0.legend(loc=1)

    # Dual graphs on 1 axis
    ax1 = ax0.twinx().twiny()

    ax1.axvline(small_x_data[-1], label="Boundary for prediction data", color="red")
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.plot(x_dates, logFunc(x_dates, *popt), label="Log model prediction based on donation data till 1-12-19", color="purple")

    ax1.legend(loc=0)

    plt.show()


def catagorize_donation_amounts(donation_df):
    # pop the first 11 rows which are not per 10s
    SKIP = 11

    # donation catagories
    bins = [0, 100, 5000, 50000, 9999999999]

    data_dates = donation_df['date'].iloc[SKIP:]
    donation_data_delta = donation_df.diff(periods=1, axis=0)
    donation_data_delta = donation_data_delta.iloc[SKIP:]['amount']

    merged = pd.concat([data_dates, donation_data_delta], axis=1, keys=['date', 'donated_amount'])

    # Bin the donations
    merged['bin'] = pd.cut(x=merged['donated_amount'], bins=bins)
    # cumsum the bins
    merged['cumsum'] = merged.groupby('bin')['donated_amount'].cumsum()

    # Group the tweets per catagory for v-lines
    binned = merged.groupby(['bin'])

    # Get the donors in highest interval
    TOP_DONORS_INTERVAL = pd.Interval(left=50000, right=9999999999)
    top_donor_data = binned.get_group(TOP_DONORS_INTERVAL)


    for high_donation_date in top_donor_data['date']:
        plt.axvline(high_donation_date)

    sns.lineplot(x="date", y="cumsum", hue="bin", data=merged)
    plt.yscale('log')
    plt.show()

# # get data in raw form
# tweet_df = get_tweet_data()
# donation_df = get_donation_data()

#get_correlation_data('data/twitter_data/count_sentiment_per_day_tweets.csv')
#get_correlation_data()

parse_donation_data()

# fit_log_model_analysis(donation_df)
# catagorize_donation_amounts(donation_df)
