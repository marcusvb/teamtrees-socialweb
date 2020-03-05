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
    # Read the 10s donation data from the website
    # Src: https://vps.natur-kultur.eu/trees.html
    df = pd.read_csv("data/donation_data/team-trees-10second.csv", delimiter=",", header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    df['raised_capital'] = (df['amount']/20000000)*100

    rate_of_fundings = [0]
    for i in range(0, len(df['date'])-1):
        current_date = df['date'].values[i]
        current_capital = df['amount'].values[i]

        next_date = df['date'].values[i+1]
        next_capital = df['amount'].values[i+1]

        time_diff = (next_date-current_date).total_seconds()
        delta_diff = next_capital - current_capital

        min_rate = (delta_diff/time_diff)*60
        rate_of_fundings.append(min_rate)

    df['rate_of_funding'] = rate_of_fundings
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

def get_correlation_data():
    tweets_per_day = get_tweet_count_data('day')
    av_donation_rate_per_day = get_donation_rate_data('day')

    '''Doesn't work at the moment as the datasets need to be of the same size (so per day)'''
    # calculate covariance of tweets per day and rate of funding
    start_date = '2019-10-25'
    end_date = '2020-02-03'
    print(calc_correlation(tweets_per_day, av_donation_rate_per_day, start_date, end_date))


def catagorize_donation_amounts(donation_df, compare_log_model=False):
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

    # def log_func(x, a, b, c):
    #     return a * x + b * np.power(x, 2) + c

    def logFunc(x, a, b):
        return a + b * np.log(x)

    def logFit(x, y):
        # cache some frequently reused terms
        sumy = np.sum(y)
        sumlogx = np.sum(np.log(x))

        b = (x.size * np.sum(y * np.log(x)) - sumy * sumlogx) / (x.size * np.sum(np.log(x) ** 2) - sumlogx ** 2)
        a = (sumy - b * sumlogx) / x.size

        return a, b

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    if compare_log_model:
        indexes = top_donor_data.index
        filtered_no_top_donors = merged.drop(labels=indexes, axis=0)
        filtered_no_top_donors['cumsum'] = filtered_no_top_donors['donated_amount'].cumsum()

        y_data = filtered_no_top_donors['cumsum'].to_numpy()
        y_data = NormalizeData(y_data)

        # Keep the dates for the ticking
        x_dates = mdates.date2num(filtered_no_top_donors['date'])
        x_dates = x_dates - (x_dates[0]-1) # day 0 starts at 1, so instead of large numbers we normalize to the data
        # x_dates = NormalizeData(x_dates)

        # fit to around december 2019, to see the campaign progress
        small_x_data = x_dates[0:4000]
        small_y_data = y_data[0:4000]

        print("xdata")
        print(small_x_data)
        print("ydata")
        print(small_y_data)

        # popt, _ = curve_fit(log_func, small_x_data, small_y_data)

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('date')
        ax1.set_ylabel('cumulative donations (scaled)', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', labelcolor=color)
        ax1.plot(x_dates, logFunc(x_dates, *logFit(small_x_data, small_y_data)))
        # ax1.plot(x_dates[2:], log_func(x_dates[2:], *popt), label="fitted line")
        plt.legend()

        ax2 = ax1.twinx().twiny()

        color = 'tab:blue'
        ax2.set_xlabel('date')
        ax2.set_ylabel('cumulative donations', color=color)
        ax2.plot(filtered_no_top_donors['date'], filtered_no_top_donors['cumsum'], linestyle="--", label="real data", color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.tick_params(axis='x', labelcolor=color)

        plt.legend()
        # plt.savefig("done.pdf")
        plt.show()

    else:
        for high_donation_date in top_donor_data['date']:
            plt.axvline(high_donation_date)

        sns.lineplot(x="date", y="cumsum", hue="bin", data=merged)
        plt.yscale('log')
        plt.show()

# get data in raw form
tweet_df = get_tweet_data()
donation_df = get_donation_data()

catagorize_donation_amounts(donation_df, compare_log_model=True)
