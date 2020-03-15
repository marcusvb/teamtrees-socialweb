import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import copy

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
        correlation = np.corrcoef(social_data.loc[period_social]['n_negative'], donation_data.loc[period_donation]['av_rate'])
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


def get_correlation_data(file1='regular_twitter_data', file2='donation_rate_data', time_unit='day', sentiment=False):
    if isinstance(file1, str):
        if file1 == 'regular_twitter_data' and time_unit == 'day':
            tweets_per_day = get_tweet_count_data('day')
        elif file1 == 'regular_twitter_data' and time_unit == 'hour':
            tweets_per_day = get_tweet_count_data('hour')
        else:
            tweets_per_day = pd.read_csv(file1, delimiter=",", header=0)
    else:
        tweets_per_day = file1

    if isinstance(file2, str):
        if file2 == 'donation_rate_data' and time_unit == 'day':
            av_donation_rate_per_day = get_donation_rate_data('day')
        elif file2 == 'donation_rate_data' and time_unit == 'hour':
            av_donation_rate_per_day = get_donation_rate_data('hour')
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


def fix_intervals_for_data(df):
    # 3120 hours in this range
    START_RANGE = "2019-10-24"
    END_RANGE = "2020-03-02"

    # prep our arrays for filling
    hourly_range = pd.date_range(START_RANGE, END_RANGE, periods=3120)
    hourly_donation_data = np.zeros(len(hourly_range))

    holder = 0
    for index, row in df.iterrows():
        time_stamp = row['date']
        for i in range(holder, len(hourly_range)):
            hour_gen = hourly_range[i]
            if hour_gen > time_stamp:
                holder = i-1
                hourly_donation_data[i-1] = row['cumsum']
                break

    # Fill the rest of the data with the last entry, in other words no more summing
    last_entry = np.where(hourly_donation_data == 0)[0][-1]
    for i in range(last_entry, len(hourly_donation_data)):
        hourly_donation_data[i] = hourly_donation_data[i-1]

    df_1 = pd.DataFrame(hourly_donation_data.T).replace(to_replace=0, method='ffill')
    return df_1.values.flatten()  # after the zero fill return this df as flattened an numpy style


def plot_hourly_runned_summed_data(binned, bins):
    for i in range(1, len(bins)):
        left = bins[i - 1]
        right = bins[i]
        interval = pd.Interval(left=left, right=right)
        data = binned.get_group(interval)
        data = data.drop("donated_amount", axis=1)
        data = data.drop("bin", axis=1)
        data = fix_intervals_for_data(data)

        # get time
        START_RANGE = "2019-10-24"
        END_RANGE = "2020-03-02"
        hourly_range = pd.date_range(START_RANGE, END_RANGE, periods=3120)

        sns.lineplot(x=hourly_range, y=data, label=str(interval), drawstyle="steps-pre")
    plt.ylabel("Binned Cumulative Sum Hourly")
    plt.xlabel("Date")
    plt.legend()
    plt.show()

def correlate_binned_data(top_donor_data, binned, bins):
    # PREP TOP DONATORS
    plot_hourly_runned_summed_data(binned, bins)

    print("TOP DONATION DF")
    print(top_donor_data.head())

    top_donor_data = top_donor_data.drop("donated_amount", axis=1)
    top_donor_data = top_donor_data.drop("bin", axis=1)
    top_donor_data_per_hour = fix_intervals_for_data(top_donor_data)

    corrs = []
    hours_to_shift = 300

    # get and print correlations for binned groups
    for i in range(1, len(bins)-1):
        left = bins[i-1]
        right = bins[i]
        interval = pd.Interval(left=left, right=right)
        print("Interval to compare with top-donors", interval)

        data_to_comp = binned.get_group(interval)

        print(data_to_comp.head())
        data_to_comp = data_to_comp.drop("donated_amount", axis=1)
        data_to_comp = data_to_comp.drop("bin", axis=1)
        data_to_comp = fix_intervals_for_data(data_to_comp)

        # Assuming top donor data is the most influential

        corr_per_range = []
        for i in range(-hours_to_shift, hours_to_shift):
            # hours are actually reversed
            data_to_comp_mod = copy.deepcopy(data_to_comp)
            top_donor_data_per_hour_mod = copy.deepcopy(top_donor_data_per_hour)
            if i > 0:
                top_donor_data_per_hour_mod = top_donor_data_per_hour_mod[:-i]
                data_to_comp_mod = data_to_comp_mod[i:]
            elif i < 0:
                top_donor_data_per_hour_mod = top_donor_data_per_hour_mod[-i:]
                data_to_comp_mod = data_to_comp_mod[:i]
            else:
                pass # its 0
            corr = np.corrcoef(top_donor_data_per_hour_mod, data_to_comp_mod)[0,1] # grab the compared correlation
            corr_per_range.append(corr)

        corrs.append((interval, corr_per_range)) # append tuple-> inteval, corrs over the hours

    # init new plot
    fig, ax = plt.subplots()
    for data_brick in corrs:
        interval = data_brick[0]
        corrs_shifted = np.asarray(data_brick[1])

        # reverse the hour amount as this is logical for the graph
        x = np.asarray(range(-hours_to_shift, hours_to_shift)) * -1
        ax.plot(x, corrs_shifted, label="Interval: "+str(interval), alpha=0.5)
        xmax = x[np.argmax(corrs_shifted)]
        ymax = corrs_shifted.max()
        ax.plot(xmax, ymax, marker="o", ls="", ms=3)

    plt.ylabel("Correlation coefficient")
    plt.xlabel("Hours shifted")
    plt.legend()
    plt.show()



def derivate_binned_data(binned, bins):
    fig, ax = plt.subplots()
    for i in range(1, len(bins)):
        left = bins[i - 1]
        right = bins[i]
        interval = pd.Interval(left=left, right=right)

        data_to_comp = binned.get_group(interval)
        data_to_comp = data_to_comp.drop("donated_amount", axis=1)
        data_to_comp = data_to_comp.drop("bin", axis=1)
        data_to_comp = fix_intervals_for_data(data_to_comp)[75:2700]

        derivative_of_data = []
        for i in range(1, len(data_to_comp)):
            new = data_to_comp[i]
            old = data_to_comp[i-1]
            # hourly derivative is always an hour spaced data scheme
            derivative_of_data.append(new - old)
        ax.plot(derivative_of_data, label="interval " + str(left) + " - " + str(right), alpha=0.2)
    plt.yscale("log")
    plt.legend()
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
    
    # UGLY v lines for displaying intervals
    # for high_donation_date in top_donor_data['date']:
    #     plt.axvline(high_donation_date)

    ax = sns.lineplot(x="date", y="cumsum", hue="bin", data=merged, drawstyle="steps-pre")
    ax.set(xlabel='Date', ylabel='Binned Cumulative Sum')
    # plt.yscale('log')
    plt.show()

    correlate_binned_data(top_donor_data, binned, bins)
    # derivate_binned_data(binned, bins)

# # get data in raw form
# tweet_df = get_tweet_data()
# donation_df = get_donation_data()

#get_correlation_data('data/twitter_data/count_sentiment_per_day_tweets.csv')
#get_correlation_data()

# parse_donation_data()

# fit_log_model_analysis(donation_df)
# catagorize_donation_amounts(donation_df)
