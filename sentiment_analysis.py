import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from analysis import get_tweet_data, get_correlation_data
import pandas as pd
import math
import matplotlib.pyplot as plt

days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
import ast

vader = SentimentIntensityAnalyzer()

def get_sentiment(tweet_df):
    tweet_date = tweet_df['date']
    tweet_text = tweet_df['text'].apply(lambda x: str(x))

    sentiment_df = pd.DataFrame(columns=['date', 'tweet', 'neg', 'neu', 'pos', 'compound'])

    for i in range(len(tweet_df)):
        sentiment = vader.polarity_scores(tweet_text[i])
        sentiment_df.loc[0 if pd.isnull(sentiment_df.index.max()) else sentiment_df.index.max() + 1] = \
            [tweet_date[i], tweet_text[i],  sentiment['neg'], sentiment['neu'], sentiment['pos'], sentiment['compound']]
        print(i)

    print(sentiment_df)
    sentiment_df.to_csv('data/twitter_data/sentiment_of_tweets.csv', header=True)


def get_sentiments_trends(file):
    df = pd.read_csv(file, delimiter=",", header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    # df['sentiment'] = df.apply(ast.literal_eval, columns=['sentiment'])
    # df['sentiment'] = df['sentiment'].apply(lambda x: ast.literal_eval(x))
    # print(df['sentiment']['compound'])
    count_df = pd.DataFrame(columns=['date', 'count', 'n_positive', 'neutral', 'n_negative'])

    for year in [2019, 2020]:
        for month in range(1, 13):
            for day in range(1, days_per_month[month - 1] + 1):
                print(year, month, day)
                count_df.loc[0 if pd.isnull(count_df.index.max()) else count_df.index.max() + 1] = \
                    [pd.Period(str(year) + '-' + str(month) + '-' + str(day), 'D'),
                     len(df.loc[(df['date'].dt.day == day) &
                                (df['date'].dt.month == month) &
                                (df['date'].dt.year == year)]),
                     len(df.loc[(df['date'].dt.day == day) &
                                (df['date'].dt.month == month) &
                                (df['date'].dt.year == year) &
                                (df['compound'] > 0)]),
                     len(df.loc[(df['date'].dt.day == day) &
                                (df['date'].dt.month == month) &
                                (df['date'].dt.year == year) &
                                (df['compound'] == 0)]),
                     len(df.loc[(df['date'].dt.day == day) &
                                (df['date'].dt.month == month) &
                                (df['date'].dt.year == year) &
                                (df['compound'] < 0)])
                     ]

    count_df.to_csv('data/twitter_data/count_sentiment_' + 'per_day_tweets.csv', header=True)


# tweet_df = get_tweet_data()
# get_sentiment(tweet_df)
#
# get_sentiments_trends('data/twitter_data/sentiment_of_tweets.csv')

get_correlation_data(file1='data/twitter_data/count_sentiment_per_day_tweets.csv', sentiment=True)

df = pd.read_csv('data/twitter_data/count_sentiment_per_day_tweets.csv', delimiter=",", header=0)
#
df['n_positive'].plot()
df['n_negative'].plot()
df['neutral'].plot()
plt.legend(['Positive', 'Negative', 'Neutral'])
plt.xlabel('Time (days)', fontsize=18)
plt.ylabel('Tweets (#)', fontsize=18)
plt.show()