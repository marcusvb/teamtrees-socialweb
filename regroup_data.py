from analysis import get_tweet_data, get_donation_data
import pandas as pd
import datetime
import math
import matplotlib.pyplot as plt

days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# groups the tweet data set per time unit
def tweet_count_per_unit(df, group_per_timeunit):
    df.drop(["retweets", "favorites",  "text", "geo", "mentions", "hashtags", "id", "permalink"], axis=1)
    count_df = pd.DataFrame(columns=['date', 'count'])

    if group_per_timeunit == "day":
        for year in [2019, 2020]:
            for month in range(1, 13):
                for day in range(1, days_per_month[month-1]+1):
                    print(year, month, day)
                    count_df.loc[0 if pd.isnull(count_df.index.max()) else count_df.index.max() + 1] = \
                        [pd.Period(str(year) + '-' + str(month) + '-' + str(day), 'D'),
                         len(df.loc[(df['date'].dt.day == day) &
                            (df['date'].dt.month == month) &
                            (df['date'].dt.year == year)])]

    if group_per_timeunit == "month":
        for year in [2019, 2020]:
            for month in range(1, 12):
                print(year, month)
                count_df.loc[0 if pd.isnull(count_df.index.max()) else count_df.index.max() + 1] = \
                    [pd.Period(str(year) + '-' + str(month), 'M'), len(df.loc[(df['date'].dt.month == month) &
                                                                    (df['date'].dt.year == year)])]

    if group_per_timeunit == "year":
        for year in [2019, 2020]:
            print(year)
            count_df.loc[0 if pd.isnull(count_df.index.max()) else count_df.index.max() + 1] = \
                [pd.Period(str(year), 'Y'), len(df.loc[(df['date'].dt.year == year)])]

    if group_per_timeunit == "hour":
        for year in [2019, 2020]:
            for month in range(1, 13):
                for day in range(1, days_per_month[month-1]+1):
                    for hour in range(0, 24):
                        print(year, month, day, hour)
                        count_df.loc[0 if pd.isnull(count_df.index.max()) else count_df.index.max() + 1] = \
                            [pd.Period(str(year) + '-' + str(month) + '-' + str(day) + ' ' + str(hour) + ":00", 'H'),
                             len(df.loc[(df['date'].dt.day == day) &
                                        (df['date'].dt.month == month) &
                                        (df['date'].dt.year == year) &
                                        (df['date'].dt.hour == hour)])]

    count_df.to_csv('data/twitter_data/count_' + 'per_' + group_per_timeunit + '_tweets.csv', header=True)

    return count_df


def tree_donation_rate_per_unit(df, group_per_timeunit):
    # df = get_donation_data()
    # df['rate_of_funding'] = df['rate_of_funding'].apply(lambda x: float(x.strip('/min')))
    print(df['rate_of_funding'].head())
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

    # df.drop(["percent_of_goal", "raised_capital", "predicted_goal_reach","predicted_campaign_end"], axis=1)
    av_rate_df = pd.DataFrame(columns=['date', 'av_rate'])

    if group_per_timeunit == "day":
        for year in [2019, 2020]:
            for month in range(1, 13):
                for day in range(1, days_per_month[month-1]+1):
                    print(year, month, day)
                    mean = df.loc[(df['date'].dt.day == day) &
                            (df['date'].dt.month == month) &
                            (df['date'].dt.year == year)]['rate_of_funding'].mean()
                    if math.isnan(mean):
                        mean = 0

                    av_rate_df.loc[0 if pd.isnull(av_rate_df.index.max()) else av_rate_df.index.max() + 1] = \
                        [pd.Period(str(year) + '-' + str(month) + '-' + str(day), 'D'), mean]

    if group_per_timeunit == "month":
        for year in [2019, 2020]:
            for month in range(1, 12):
                print(year, month)
                mean = df.loc[(df['date'].dt.month == month) &
                              (df['date'].dt.year == year)]['rate_of_funding'].mean()
                if math.isnan(mean):
                    mean = 0

                av_rate_df.loc[0 if pd.isnull(av_rate_df.index.max()) else av_rate_df.index.max() + 1] = \
                    [pd.Period(str(year) +'-'+ str(month), 'M'), mean]

    if group_per_timeunit == "year":
        for year in [2019, 2020]:
            print(year)
            mean = df.loc[(df['date'].dt.year == year)]['rate_of_funding'].mean()
            if math.isnan(mean):
                mean = 0

            av_rate_df.loc[0 if pd.isnull(av_rate_df.index.max()) else av_rate_df.index.max() + 1] = \
                [pd.Period(str(year), 'Y'), mean]

    if group_per_timeunit == "hour":
        for year in [2019, 2020]:
            for month in range(1, 13):
                for day in range(1, days_per_month[month-1]+1):
                    for hour in range(0, 24):
                        print(year, month, day, hour)
                        mean = df.loc[(df['date'].dt.day == day) &
                                (df['date'].dt.month == month) &
                                (df['date'].dt.year == year) &
                                (df['date'].dt.hour == hour)]['rate_of_funding'].mean()
                        if math.isnan(mean):
                            mean = 0

                        av_rate_df.loc[0 if pd.isnull(av_rate_df.index.max()) else av_rate_df.index.max() + 1] = \
                            [pd.Period(str(year) + '-' + str(month) + '-' + str(day) + ' ' + str(hour) + ":00", 'H'), mean]

    av_rate_df.to_csv('data/donation_data/av_rate_' + 'per_' + group_per_timeunit + '_donations_merged.csv', header=True)

    return av_rate_df

#tweet_df = get_tweet_data()
# donation_df = get_donation_data()

dft = pd.read_csv('data/donation_data/parsed-team-trees-10second-merged-20min-data.csv', header=0, delimiter=",")

donation_rate_df = tree_donation_rate_per_unit(dft, 'day')

#count_df = tweet_count_per_unit(tweet_df, "hour")
# donation_rate_df = tree_donation_rate_per_unit(donation_df, 'day')

#count_df.plot(x='date', y='count')
# donation_rate_df.plot(x='date', y='av_rate')

# plt.show()
