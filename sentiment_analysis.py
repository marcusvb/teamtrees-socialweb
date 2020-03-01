import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from analysis import get_tweet_data
import pandas as pd

def get_tweet_data():
    df = pd.read_csv("data/twitter_data/tweets.csv", delimiter=";", header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    df['text'] = df['text'].astype(str)
    return df



vader = SentimentIntensityAnalyzer()

tweet_df = get_tweet_data()
tweet_date = tweet_df['date']
tweet_text = tweet_df['text']

sentiment_df = pd.DataFrame(columns=['date', 'tweet', 'sentiment'])

for i in range(len(tweet_df)):
    sentiment_df.loc[0 if pd.isnull(sentiment_df.index.max()) else sentiment_df.index.max() + 1] = \
        [tweet_date[i], tweet_text[i], vader.polarity_scores(tweet_text[i])]
    print(i)

print(sentiment_df)
sentiment_df.to_csv('data/twitter_data/sentiment_of_tweets.csv', header=True)

