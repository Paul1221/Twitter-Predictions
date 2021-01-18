import json
import kmeans1d
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import datetime
import pandas as pd

jsons = []

# with open("sample.json", "r") as infile:
#     jsons.append(json.load(infile))

# with open("sample_bucharest.json", "r") as infile:
#     jsons.append(json.load(infile))

# with open("sample_rezist.json", "r") as infile:
#     jsons.append(json.load(infile))

with open("sample_selfie.json", "r") as infile:
    jsons.append(json.load(infile))

tweets = []


for el in jsons:
    tweets.extend(el)

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

for i in range(0, (nr_of_tweets-10)):
    tweet = [favorites_clusters[i], listed_clusters[i], followers_clusters[i], friends_clusters[i],
             statuses_clusters[i], time_clusters[i], sentiment_clusters[i], objectivity_clusters[i]]
    x.append(tweet)

x, y = np.array(x), np.array(retweet_list[0:-10])

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
    i += 1
#print(y_pred.where(a=max(y_pred)))
print(max)
print(argmax)
print("best hour", time_list[argmax])

norm_data = {}

for x,y in zip(time_list[0:-10], y_pred):
    if x not in norm_data:
        norm_data[x] = [y]
    else:
        norm_data[x].append(y)

norm_data_f = {}

for x in norm_data:
    norm_data_f[x] = sum(norm_data[x])/len(norm_data[x])

print("normdataf", norm_data_f)

import collections

od = collections.OrderedDict(sorted(norm_data_f.items()))

import matplotlib.pyplot as plt

plt.bar(range(len(od)), list(od.values()), align='center')
plt.xticks(range(len(od)), list(od.keys()))

plt.show()



print(time_list)
print(len(time_list))
print(retweet_list)
print(len(retweet_list))


