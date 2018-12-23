# -*- coding: utf-8 -*-
# Simple/example Python 3 script to perform sentiment analysis on an individual's Tweets
# ***Very much a WIP***
# Adapted from RodolfoFerro on GitHub
# All libraries available via pip install 

import tweepy 
import pandas as pd
import numpy as np 
from IPython.display import display

# Prompt user for Twitter handle to analyze
print('Enter the Twitter handle (without the @) to analyze: ')
handle_to_analyze = input()
print('Enter the number of tweets to analyze (e.g. 300)')
tweet_quantity = input()

print('Now analyzing ' + tweet_quantity + ' of @' + handle_to_analyze + ' s tweets...')


# Reference seperate .py file containing Twitter API authentication data
# For end user, could prompt for info
from twitcred_sample import * # Use keys as variables

# Create function to setup Twitter API
def twitter_setup():
    """
    Utility function to setup the Twitter API
    with our access keys provided.
    """
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    
    # Return API with authentication:
    api = tweepy.API(auth)
    return api

# Use above function to create extractor object
extractor = twitter_setup()

# Create tweet list
tweets = extractor.user_timeline(screen_name=handle_to_analyze, count=200)
print("Number of tweets extracted: {}.\n".format(len(tweets)))

# We print the most recent 5 tweets:
print("5 recent tweets:\n")
for tweet in tweets[:5]:
    print(tweet.text)
    print()


# Create pandas DataFrame to manipulate the data pulled by tweepy
data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

# Display first 10 entries of the dataframe
display(data.head(10))


# Internal methods of a single tweet object:
# i.e. view all the elements (metadata) available in single tweet
print(dir(tweets[0])) 

# e.g. Print certain data from first tweet
print(tweets[0].id)
print(tweets[0].created_at)
print(tweets[0].source)
print(tweets[0].favorite_count)
print(tweets[0].retweet_count)
print(tweets[0].geo)
print(tweets[0].coordinates)
print(tweets[0].entities)


# Lots of data avail in each tweet, not all useful
# Add relevant data to dataframe using Pythons list comprehension
data['len'] = np.array([len(tweet.text) for tweet in tweets])
data['ID'] = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes'] = np.array([tweet.favorite_count for tweet in tweets])
data['RTs'] = np.array ([tweet.retweet_count for tweet in tweets])

# Display first 10 elements from dataframe:
# (To view changes)
display(data.head(10))

### Part 2: Basic Statistics using numpy
### Part 2: Data Visualization using matplotlib

# Extract the mean of the lengths of tweets
mean = np.mean(data['len'])

print("The mean character length of tweets: {}".format(mean))


# Additional pandas features

# Extract tweet with the most FAVs and the most RTs

fav_max = np.max(data['Likes'])
rt_max = np.max(data['RTs'])

fav = data[data.Likes == fav_max].index[0]
rt = data[data.RTs == rt_max].index[0]

# Max FAVs
print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
print("Number of likes: {}".format(fav_max))
print("{} characters.\n".format(data['len'][fav]))

# Max RTs
print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
print("The number of retweets: {}".format(rt_max))
print("{} characters.\n".format(data['len'][rt]))

# Explanation copied direct from RodolfoFerro's GitHub;
# "This is common, but it won't necessarily happen: the tweet with 
# more likes is the tweet with more retweets. What we're doing is 
# that we find the maximum number of likes from the 'Likes' column 
# and the maximum number of retweets from the 'RTs' using numpy's max
# function. With this we just look for the index in each of both columns 
# that satisfy to be the maximum. Since more than one could have the same 
# number of likes/retweets (the maximum) we just need to take the first 
# one found, and that's why we use .index[0] to assign the index to the 
# variables favand rt. To print the tweet that satisfies, we access the 
# data in the same way we would access a matrix or any indexed object."


# Create a time series for the data

tlen = pd.Series(data=data['len'].values, index=data['Date'])
tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
tret = pd.Series(data=data['RTs'].values, index=data['Date'])

# Tweet lengths over time (time = 200 tweets)
tlen.plot(figsize=(16,4), label="Tweet Length", color='r', legend=True);


# Tweet Likes vs Retweets over time (time being last 200 tweets)
tfav.plot(figsize=(16,4), label="Likes", color='b', legend=True);
tret.plot(figsize=(16,4), label="Retweets", color='g', legend=True);

# Important not note what elements are worth comparing (esp. visually)
# e.g. plot below makes 'Tweet Length' irrelevant visually

tlen.plot(figsize=(16,4), label="Tweet Length", color='r', legend=True);
tfav.plot(figsize=(16,4), label="Likes", color='b', legend=True);
tret.plot(figsize=(16,4), label="Retweets", color='g', legend=True);

# Not every tweet is from the same source so...

# Find all possible sources
sources = []
for source in data['Source']:
    if source not in sources:
        sources.append(source)
        
# Print list of source
print("Creation of content sources: ")
for source in sources:
    print("*{}".format(source))


# Create numpy vector mapped to labels:
percent = np.zeros(len(sources))

for source in data['Source']:
    for index in range(len(sources)):
        if source == sources[index]:
            percent[index] += 1
            pass
        
percent /= 100


# Create pandas pie chart showing percent of tweets per source
pie_chart = pd.Series(percent, index=sources, name='Sources')
pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(6,6));


### Part 3: Sentiment Analysis using textblob
### Part 3: Using re (regular expressions) to clean text and
### create classifier to determine polarity of each tweet after cleaned


from textblob import TextBlob
import re

def clean_tweet(tweet):
    """
    Create function to 'clean text'
    i.e. remove links, special characters, etc. using regex
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    """
    Create function to classify 'polarity' of a tweet
    using pre-trained analyzer in textblob
    (can use different ML NLP models with textblob)
    """
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


# Create column with the result of the analysis
data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets']])

# Show updated dataframe with SA value added (far right)
display(data.head(10))


# Analyze results, determine percentage of positive, negative, neutral tweets

# Create lists with classified tweets:
pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]

# Print percentages:
print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
print("Percentage of negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))

