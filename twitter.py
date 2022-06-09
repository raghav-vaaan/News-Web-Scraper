import time
from nltk.util import pr
import pandas as pd
import logging
from bs4 import BeautifulSoup
import os

import numpy as np
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
from config import *
from datetime import datetime

from tweepy import *
import tweepy
import pickle

import threading
import _thread   # import thread in python2
import time



class timeout():
  def __init__(self, time):
    self.time= time
    self.exit=False

  def __enter__(self):
    threading.Thread(target=self.callme).start()

  def callme(self):
    time.sleep(self.time)
    if self.exit==False:
       _thread.interrupt_main()  # use thread instead of _thread in python2
  def __exit__(self, a, b, c):
       self.exit=True


consumer_key = consumer_key
consumer_secret = consumer_secret
access_key = access_key
access_secret = access_secret
with open(path_naive_bayes, 'rb') as handle:
        naive_bayes = pickle.load(handle)
with open(path_count_vec, 'rb') as handle:
    count_vector = pickle.load(handle)
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

try:
    inference_dataframe = {'headline': [],
    'category': [] }
except Exception as e:
    logging.error(e, exc_info = True)



def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def remove_stopwords(text):
    stopword = set(stopwords.words('english'))
    text=[word.lower()+' ' for word in text.split() if word.lower() not in stopword]
    return ''.join(text)

def lemmitize(text):
  lemmatizer = WordNetLemmatizer()
  return ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)])

def remove_dig(text):
  tmp = ''
  i = 0
  while i < len(text):
    #Removing all digits
    #Removing all 's
    if text[i] == "'" and i < len(text) - 1 and text[i+1] == "s":
      i += 1
    elif not text[i].isdigit():
      tmp += text[i]
    i += 1
  return tmp


def inference(S, count_vector, model):
  # S = pd.DataFrame(S, columns=['text'])
  S = count_vector.transform(S.text)
  pred = model.predict(S)
  out = []
  for i in pred:
    
    out.append(i)
  return out


import re
def remove_meta_data(S):
  for i in range(len(S)):
    s = str(S[i][0])
    s = re.sub('<[^>]+>', '', s)
    S[i] = s
  return S


def Acc_data(text ,model):
    text = [[i] for i in text]

    df = remove_meta_data(text)
    df =  pd.DataFrame(text, columns = ['text'])
    df['text'] = df['text'].apply(lambda x: remove_stopwords(str(x)))
    df['text'] = df['text'].apply(lambda x: lemmitize(x))
    df['text'] = df['text'].apply(lambda x: remove_dig(x))
    out = inference(df,count_vector, model)
    return out


def sentiment(text):
    #text: list of strings
    for i in range(len(text)):
        sent = TextBlob(text[i]).sentiment.polarity
        if sent < .2 and sent > -.2:
            text[i] = 'Neutral'
        elif sent > .2:
            text[i] = 'Positive'
        else:
            text[i] = 'Negative'
    
    pos,neg,neut = 0, 0, 0
    for i in range(len(text)):
        if text[i] == 'Positive':
            pos += 1
        elif text[i] == 'Negative':
            neg += 1
        else:
            neut += 1
    return len(text), pos,neg,neut



#searching by keywords

def twitter_keywords( search_keyword, count, comment):
    search_words = "#" + search_keyword + ' -filter:retweets'
    date_since = "2018-11-16"

    tweets = tweepy.Cursor(api.search,
    q=search_words,
    lang="en",
    since=date_since).items(count)
    users_locs = [[tweet.id ,tweet.author.screen_name, tweet.source, tweet.text] for tweet in tweets] 
    tweet_text = pd.DataFrame(data=users_locs, columns=['tweet id', "twitter handle",  'Source', 'desc'])
    tweet_text['headline'] = 0
    tweet_text['category'] = search_keyword    
    tweet_text['timestamp'] = datetime.now()

    DF = pd.read_csv('Acc_Data.csv', usecols= ['headline','desc', 'category', 'pred_category', 'timestamp', 'Source','Total comments','Pos comments',
        'Neg comments', 'Neut comments','tweet id','twitter handle'])
    y = tweet_text.desc
    y = Acc_data(y, naive_bayes)
    tweet_text['pred_category'] = y
    df = tweet_text[['headline','desc','category','pred_category','timestamp', 'Source']]
    df['pred_keyword'] = y

    total,pos,neg,neut = [],[],[],[]
    if comment:
        for i in df.index:      
            print(tweet_text['twitter handle'][i], tweet_text['tweet id'][i])
            t = twitter_comments(tweet_text['twitter handle'][i], str(tweet_text['tweet id'][i]), comment_count).text
            # try:
            #     with timeout(3):
            #         t = twitter_comments(tweet_text['twitter handle'][i], tweet_text['tweet id'][i], comment_count).text
            # except:
                
            #     total.append(0)
            #     pos.append(0)
            #     neg.append(0)
            #     neut.append(0)
            #     continue

            t = t.to_list()
            tot,po,ne,nut= sentiment(t)
            print(tot,po,ne,nut)
            total.append(tot)
            pos.append(po)
            neg.append(ne)
            neut.append(nut)
    if len(total) == 0:
        total = 0
    if len(pos) == 0:
        pos = 0
    if len(neg) == 0:
        neg = 0
    if len(neut) == 0:
        neut = 0
    
    df['Total comments'] = total
    df['Pos comments'] = pos
    df['Neg comments'] = neg
    df['Neut comments'] = neut        
    df['tweet id'] = tweet_text['tweet id']
    df['twitter handle'] = tweet_text['twitter handle']    
    
    df =pd.concat([df, DF])
    df = df[['headline','desc', 'category', 'pred_category', 'timestamp', 'Source','Total comments','Pos comments',
        'Neg comments', 'Neut comments','tweet id','twitter handle']]
    df.drop_duplicates(subset= 'desc', keep= 'first', inplace=True)
    df.reset_index(inplace=True)
    df.to_csv('Acc_Data.csv')


#searching by user id
def twitter_id(username, count, comment):
    tweets = api.user_timeline(screen_name=username)
    
    users_locs = [[tweet.id ,tweet.author.screen_name, tweet.source, tweet.text] for tweet in tweets] 
    tweet_text = pd.DataFrame(data=users_locs, columns=['tweet id', "twitter handle",  'Source', 'desc'])
    tweet_text['headline'] = 0
    tweet_text['category'] = username    
    tweet_text['timestamp'] = datetime.now()
    tweet_text = tweet_text.head(count)

    DF = pd.read_csv('Acc_Data.csv', usecols= ['headline','desc', 'category', 'pred_category', 'timestamp', 'Source','Total comments','Pos comments',
        'Neg comments', 'Neut comments','tweet id','twitter handle'])
    y = tweet_text.desc
    y = Acc_data(y, naive_bayes)
    tweet_text['pred_category'] = y
    df = tweet_text[['headline','desc','category','pred_category','timestamp', 'Source']]
    df['pred_keyword'] = y

    total,pos,neg,neut = [],[],[],[]
    if comment:
        for i in df.index:      
            print(tweet_text['twitter handle'][i], tweet_text['tweet id'][i])
            t = twitter_comments(tweet_text['twitter handle'][i], str(tweet_text['tweet id'][i]), comment_count).text
            # try:
            #     with timeout(30):
            #         t = twitter_comments(tweet_text['twitter handle'][i], tweet_text['tweet id'][i], comment_count).text
            #         print(t)
            # except:
            #     print(1)
            #     total.append(0)
            #     pos.append(0)
            #     neg.append(0)
            #     neut.append(0)
            #     continue
            
            t = t.to_list()
            print(len(t),t)
            tot,po,ne,nut= sentiment(t)
            print(tot,po,ne,nut)
            total.append(tot)
            pos.append(po)
            neg.append(ne)
            neut.append(nut)
    if len(total) == 0:
        total = 0
    if len(pos) == 0:
        pos = 0
    if len(neg) == 0:
        neg = 0
    if len(neut) == 0:
        neut = 0

    df['Total comments'] = total
    df['Pos comments'] = pos
    df['Neg comments'] = neg
    df['Neut comments'] = neut        
    df['tweet id'] = tweet_text['tweet id']
    df['twitter handle'] = tweet_text['twitter handle']    
    
    df =pd.concat([df, DF])
    df = df[['headline','desc', 'category', 'pred_category', 'timestamp', 'Source','Total comments','Pos comments',
        'Neg comments', 'Neut comments','tweet id','twitter handle']]
    df.drop_duplicates(subset= 'desc', keep= 'first', inplace=True)
    df.reset_index(inplace=True)
    df.to_csv('Acc_Data.csv')


# #extracting replies from tweet id
def twitter_comments(name, tweet_id, count):    

    replies=[]
    replies = tweepy.Cursor(api.search, q='to:{}'.format(name),
                                    since_id=tweet_id, tweet_mode='extended',result_type='recent').items(count)
    text = []
    for i in replies:
        if i.in_reply_to_status_id_str == tweet_id:
            text.append([i.full_text])
    return pd.DataFrame(text, columns= ['text'])


# twitter_keywords(keyword,keyword_count,True)
# twitter_id('RahulGandhi',10,False)
# print(twitter_comments('AdonicaB', '1438566893850284035',100))
 
