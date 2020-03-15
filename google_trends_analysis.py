import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from analysis import get_correlation_data
from analysis import get_tweet_data, get_donation_rate_data, get_tweet_count_data
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def get_google_data():
    df = pd.read_csv("data/google_trends_data/google_trends_longer_period_daily.csv", delimiter=",", header=0)
    df['date'] = pd.to_datetime(df. iloc[1:-1, 0], infer_datetime_format=True)  # first column is the date
    df['count'] = df. iloc[1:-1, 1]  # second column is the search score
    return df


# go read in the data, you rascal
google_df = get_google_data()
tweet_df = get_tweet_count_data('day')
donation_df = get_donation_rate_data('day')

# get correlation between google data
get_correlation_data(google_df)

# set period for comparison
start_date = '2019-10-25'
end_date = '2020-02-03'
period_google = (google_df['date'] >= start_date) & (google_df['date'] <= end_date)
period_social = (tweet_df['date'] >= start_date) & (tweet_df['date'] <= end_date)

times = pd.date_range(start_date, end_date, periods=15)
times = [str(times[i].date()) for i in range(len(times))]

print(times)


fig, ax = plt.subplots()
fig.autofmt_xdate()
ax.xaxis_date()
xfmt = dates.DateFormatter('%y-%m-%d')
# ax.xaxis.set_major_formatter(xfmt)
# ax.plot_date(times.to_pydatetime(), y, 'v-')

plt.xlabel("Date", fontsize=18)
plt.ylabel("Rate (a.u.)", fontsize=18)
plt.plot_date(google_df.loc[period_google]['Dag'], google_df[period_google]['count']*100, '-',  label='google trends')
plt.plot_date(tweet_df.loc[period_social]['date'], tweet_df.loc[period_social]['count'], '-',  label='tweet count')
plt.plot_date(donation_df.loc[period_social]['date'], donation_df.loc[period_social]['av_rate']*3, '-',  label='donations')
plt.xticks(times[0:-1])
ax.legend()
# ax = google_df.plot(x='date', y='count')
# tweet_df.plot(x='date', y='count')
# donation_df.plot(x='date', y='av_rate')

plt.show()
