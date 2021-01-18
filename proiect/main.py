import tweepy
import json
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

consumer_key = 'j8kG8SZRw53Igb5sdyx0AXKk3'
consumer_secret = 'YMSpwyFQOYxGOEGTeNxqIjd14g5hkIJ8T1zbapgno8PqkWv7eM'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)
search = api.search(q='#facebook', result_type='mixed', count=500, until='2021-01-17')

tweets = []
i = 0

for tweet in search:
    print(tweet)
    if tweet.retweet_count != 0:
        i += 1
        blob = TextBlob(tweet.text, analyzer=NaiveBayesAnalyzer())
        sentiment = blob.sentiment
        objectivity = 1 - blob.subjectivity
        tweets.append({'retweets': tweet.retweet_count, 'favorites': tweet.user.favourites_count,
                       'created_at': int(tweet.created_at.strftime('%H')),
                       'id': i, 'sentiment': sentiment.p_pos-0.5, 'objectivity': objectivity-0.5,
                       'followers': tweet.user.followers_count,
                       'friends': tweet.user.friends_count, 'listed': tweet.user.listed_count,
                       'statuses': tweet.user.statuses_count})
print(i)

with open("sample.json", "w") as outfile:
    json.dump(tweets, outfile)

