import pandas as pd
import matplotlib.pyplot as plt
from analysis import get_correlation_data
from analysis import get_tweet_data, get_donation_rate_data, get_tweet_count_data

def get_google_data():
    df = pd.read_csv("data/google_trends_data/multiTimeline.csv", delimiter=",", header=0)
    df['date'] = pd.to_datetime(df['Day'], infer_datetime_format=True)
    df['count'] = df['teamtrees: (Worldwide)']
    return df


google_df = get_google_data()
tweet_df = get_tweet_count_data('day')
donation_df = get_donation_rate_data('day')

get_correlation_data(google_df)

plt.figure()
start_date = '2019-10-25'
end_date = '2020-02-03'
period_google = (google_df['date'] >= start_date) & (google_df['date'] <= end_date)

period_social = (tweet_df['date'] >= start_date) & (tweet_df['date'] <= end_date)

print(google_df.loc[period_google]['date'])
print(tweet_df.loc[period_social]['date'])
print(donation_df.loc[period_social]['date'])


plt.plot(google_df.loc[period_google]['Day'], google_df['count']*100,  label='google')
plt.plot(tweet_df.loc[period_social]['date'], tweet_df.loc[period_social]['count'],  label='tweet count')
plt.plot(donation_df.loc[period_social]['date'], donation_df.loc[period_social]['av_rate']*3,  label='donations')
plt.legend()
# ax = google_df.plot(x='date', y='count')
# tweet_df.plot(x='date', y='count')
# donation_df.plot(x='date', y='av_rate')

plt.show()
