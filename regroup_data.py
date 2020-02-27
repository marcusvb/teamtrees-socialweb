from analysis import get_tweet_data, get_donation_data
import pandas as pd
import datetime
import matplotlib.pyplot as plt

days_per_month = [31, 2, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def tweet_count_per_unit(df, group_per_timeunit):
    df.drop(["retweets", "favorites",  "text", "geo", "mentions", "hashtags", "id", "permalink"], axis=1)
    count_df = pd.DataFrame(columns=['date', 'count'])

    if group_per_timeunit == "day":
        for year in [2019, 2020]:
            for month in range(1, 12):
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
                    [pd.Period(str(year) +'-'+ str(month), 'M'), len(df.loc[(df['date'].dt.month == month) &
                                                                    (df['date'].dt.year == year)])]

    if group_per_timeunit == "year":
        for year in [2019, 2020]:
            print(year)
            count_df.loc[0 if pd.isnull(count_df.index.max()) else count_df.index.max() + 1] = \
                [pd.Period(str(year), 'Y'), len(df.loc[(df['date'].dt.year == year)])]

    count_df.to_csv('data/twitter_data/count_' + 'per_' + group_per_timeunit + '_tweets.csv', header=True)

    return count_df

tweet_df = get_tweet_data()
donation_df = get_donation_data()

count_df = tweet_count_per_unit(tweet_df, "year")

count_df.plot(x='date', y='count')
plt.show()