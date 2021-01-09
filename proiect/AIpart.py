import json
import kmeans1d
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import datetime
from fbprophet import Prophet
import pandas as pd


with open("sample.json", "r") as infile:
    tweets = json.load(infile)

nr_of_tweets = len(tweets)

favorites_list = []
friends_list = []
listed_list = []
followers_list = []
statuses_list = []
time_list = []
sentiment_list = []
objectivity_list = []
retweet_list = []

for tweet in tweets:
    favorites_list.append(tweet['favorites'])
    listed_list.append(tweet['listed'])
    followers_list.append(tweet['followers'])
    friends_list.append(tweet['friends'])
    statuses_list.append(tweet['statuses'])
    time_list.append(tweet['created_at'])
    sentiment_list.append(tweet['sentiment'])
    objectivity_list.append(tweet['objectivity'])
    retweet_list.append(tweet['retweets'])


favorites_clusters, favorites_centroids = kmeans1d.cluster(favorites_list, 16)
listed_clusters, listed_centroids = kmeans1d.cluster(listed_list, 7)
followers_clusters, followers_centroids = kmeans1d.cluster(followers_list, 11)
friends_clusters, friends_centroids = kmeans1d.cluster(friends_list, 16)
statuses_clusters, statuses_centroids = kmeans1d.cluster(statuses_list, 11)
time_clusters, time_centroids = kmeans1d.cluster(statuses_list, 24)
sentiment_clusters, sentiment_centroids = kmeans1d.cluster(sentiment_list, 5)
objectivity_clusters, objectivity_centroids = kmeans1d.cluster(objectivity_list, 5)

print(time_list)
print(time_clusters)
print(time_centroids)

x = []

for i in range(0, (nr_of_tweets-2)):
    tweet = [favorites_clusters[i], listed_clusters[i], followers_clusters[i], friends_clusters[i],
             statuses_clusters[i], time_clusters[i], sentiment_clusters[i], objectivity_clusters[i]]
    x.append(tweet)

x, y = np.array(x), np.array(retweet_list[0:-2])

x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

model = LinearRegression().fit(x_, y)

r_sq = model.score(x_, y)
intercept, coefficients = model.intercept_, model.coef_

y_pred = model.predict(x_)

print(retweet_list)
#print(y_pred)
max = 0
argmax = 0
i = 0
for nr in y_pred:
    if nr > max:
        max = nr
        argmax = i
    print(i)
    print(nr)
    i += 1
#print(y_pred.where(a=max(y_pred)))
print(max)
print(argmax)
print(time_list[argmax])


print(time_list)
print(len(time_list))
print(retweet_list)
print(len(retweet_list))

'''
    partea 2
'''

data = []

for i in range(0, len(time_list)):
    data.append((datetime.datetime.strptime(str(time_list[i]), '%H'), y[i]))

print(data)

df = pd.DataFrame.from_records(data, columns=['ds', 'y'])

print(df)

m = Prophet()
m.fit(df)

da = m.make_future_dataframe(periods=24, freq='h')
dada = m.predict(da)
print(dada.ds[2])

i = 0

for x in dada.yhat:
    if x == max(dada.yhat):
        break
    i += 1


besthour = int(dada.ds[i].strftime('%H'))
print(besthour)
