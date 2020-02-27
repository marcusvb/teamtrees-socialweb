import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


tweet_df = get_tweet_data()
donation_df = get_donation_data()

plt.bar(donation_df['date'], donation_df['raised_capital'])
plt.xlabel("Date")
plt.ylabel("Raised capital ($)")
plt.show()

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
