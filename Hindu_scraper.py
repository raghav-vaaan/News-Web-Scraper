from operator import ne
import pandas as pd
import logging
import time
import requests
from bs4 import BeautifulSoup
import os
import pickle
import numpy as np
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from config import *
from datetime import datetime
logging.basicConfig(filename='Hindu_scraper.log',
                    filemode='a',
                    format='%(asctime)s:%(msecs)d-%(name)s-%(levelname)s-%(message)s',
                    datefmt='%d-%m-%y %H:%M:%S',level=logging.DEBUG)


try:
    with open(path_naive_bayes, 'rb') as handle:
        naive_bayes = pickle.load(handle)
    with open(path_count_vec, 'rb') as handle:
        count_vector = pickle.load(handle)

    new = []
    inference_dataframe = {'headline': [],
    'category': [], 'desc':[] }
except Exception as e:
    logging.error(e, exc_info = True)


def Politics():
        
    #Creating a set to remove duplicates
    last_20 = pd.read_csv('Hindu_Politics_Data.csv', usecols= ['title'])
    added = set(last_20.title)

    # Extracting latest news
    url="https://frontline.thehindu.com/politics/feeder/default.rss"
    resp=requests.get(url)
    soup=BeautifulSoup(resp.content,features="xml")
    items=soup.findAll('item')
    item=items[0]
    news_items=[]
    for item in items:
        news_item={}
        if item.title.text not in added:
            news_item['title']=item.title.text
            news_item['description']=item.description.text
            news_item['Date']= datetime.now()
            news_item['Source']= 'Hindu'
            news_items.append(news_item)
            new.append([news_item['title'] + news_item['description']])
            inference_dataframe['category'].append('Politics')
            inference_dataframe['headline'].append( news_item['title'] )
            inference_dataframe['desc'].append(news_item['description'])

    DF = pd.read_csv('Hindu_Politics_Data.csv')
    df = pd.DataFrame(news_items)
    df =pd.concat([df, DF])
    df = df[['title','desc','Date','Source']]
    return df


# def Education():
        
    # #Creating a set to remove duplicates
    # last_20 = pd.read_csv('Hindu_Edu_Data.csv', usecols= ['title'])
    # added = set(last_20.title)

    # # Extracting latest news
    # url="https://frontline.thehindu.com/the-nation/education/feeder/default.rss"
    # resp=requests.get(url)
    # soup=BeautifulSoup(resp.content,features="xml")
    # items=soup.findAll('item')
    # item=items[0]
    # news_items=[]
    # for item in items:
    #     news_item={}
    #     if item.title.text not in added:
    #         news_item['title']=item.title.text
    #         news_item['description']=item.description.text
    #         news_item['Date']= datetime.now()
    #         news_item['Source']= 'Hindu'
    #         news_items.append(news_item)
    #         new.append([news_item['title'] + news_item['description']])
    #         inference_dataframe['category'].append('Education')
    #         inference_dataframe['headline'].append( news_item['title'] )
    #     # else:
    #     #     news_items = {'title': [],
    #     #     'description': [],
    #     #     'Date':[], 'Source': []}

    # DF = pd.read_csv('Hindu_Edu_Data.csv')
    # df = pd.DataFrame(news_items)
    # df =pd.concat([df, DF])
    # df = df[['title','description','Date','Source']]
    # return df


def Entertainment():
        
    #Creating a set to remove duplicates
    last_20 = pd.read_csv('Hindu_Ent_Data.csv', usecols= ['title'])
    added = set(last_20.title)

    # Extracting latest news
    url="https://frontline.thehindu.com/arts-and-culture/feeder/default.rss"
    resp=requests.get(url)
    soup=BeautifulSoup(resp.content,features="xml")
    items=soup.findAll('item')
    item=items[0]
    news_items=[]
    for item in items:
        news_item={}
        if item.title.text not in added:
            news_item['title']=item.title.text
            news_item['description']=item.description.text
            news_item['Date']= datetime.now()
            news_item['Source']= 'Hindu'
            news_items.append(news_item)
            new.append([news_item['title'] + news_item['description']])
            inference_dataframe['category'].append('Entertainment')
            inference_dataframe['headline'].append( news_item['title'] )
            inference_dataframe['desc'].append(news_item['description'])
        # else:
        #     news_items = {'title': [],
        #     'description': [],
        #     'Date':[], 'Source': []}

    DF = pd.read_csv('Hindu_Ent_Data.csv')
    df = pd.DataFrame(news_items)
    df =pd.concat([df, DF])
    df = df[['title','desc','Date','Source']]
    return df

# def Health():
        
#     #Creating a set to remove duplicates
#     last_20 = pd.read_csv('Hindu_Health_Data.csv', usecols= ['title'])
#     added = set(last_20.title)

#     # Extracting latest news
#     url="https://frontline.thehindu.com/the-nation/public-health/feeder/default.rss"
#     resp=requests.get(url)
#     soup=BeautifulSoup(resp.content,features="xml")
#     items=soup.findAll('item')
#     item=items[0]
#     news_items=[]
#     for item in items:
#         news_item={}
#         if item.title.text not in added:
#             news_item['title']=item.title.text
#             news_item['description']=item.description.text
#             news_item['Date']= datetime.now()
#             news_item['Source']= 'Hindu'
#             new.append([news_item['title'] + news_item['description']])
#             new.append([news_item['title']])
#             inference_dataframe['category'].append('Health')
#             inference_dataframe['headline'].append( news_item['title'] )
#         # else:
#         #     news_items = {'title': [],
#         #     'description': [],
#         #     'Date':[], 'Source': []}

#     DF = pd.read_csv('Hindu_Health_Data.csv')
#     df = pd.DataFrame(news_items)
#     df =pd.concat([df, DF])
#     df = df[['title','description','Date','Source']]
#     return df


def Science():
        
    #Creating a set to remove duplicates
    last_20 = pd.read_csv('Hindu_Sci_Data.csv', usecols= ['title'])
    added = set(last_20.title)

    # Extracting latest news
    url="https://frontline.thehindu.com/science-and-technology/feeder/default.rss"
    resp=requests.get(url)
    soup=BeautifulSoup(resp.content,features="xml")
    items=soup.findAll('item')
    item=items[0]
    news_items=[]
    for item in items:
        news_item={}
        if item.title.text not in added:
            news_item['title']=item.title.text
            news_item['description']=item.description.text
            news_item['Date']= datetime.now()
            news_item['Source']= 'Hindu'
            news_items.append(news_item)
            new.append([news_item['title'] + news_item['description']])
            inference_dataframe['category'].append('Science')
            inference_dataframe['headline'].append( news_item['title'] )
            inference_dataframe['desc'].append(news_item['description'])
        #     'description': [],
        #     'Date':[], 'Source': []}

    DF = pd.read_csv('Hindu_Sci_Data.csv')
    df = pd.DataFrame(news_items)
    df =pd.concat([df, DF])
    df = df[['title','desc','Date','Source']]
    return df


# def Sports():
        
#     #Creating a set to remove duplicates
#     last_20 = pd.read_csv('Hindu_Sports_Data.csv', usecols= ['title'])
#     added = set(last_20.title)

#     # Extracting latest news
#     url="https://frontline.thehindu.com/other/sport/feeder/default.rss"
#     resp=requests.get(url)
#     soup=BeautifulSoup(resp.content,features="xml")
#     items=soup.findAll('item')
#     item=items[0]
#     news_items=[]
#     for item in items:
#         news_item={}
#         if item.title.text not in added:
#             news_item['title']=item.title.text
#             news_item['description']=item.description.text
#             news_item['Date']= datetime.now()
#             news_item['Source']= 'Hindu'
#             news_items.append(news_item)
#             new.append([news_item['title'] + news_item['description']])
#             inference_dataframe['category'].append('Sports')
#             inference_dataframe['headline'].append( news_item['title'] )
#         # else:
#         #     news_items = {'title': [],
#         #     'description': [],
#         #     'Date':[], 'Source': []}

#     DF = pd.read_csv('Hindu_Sports_Data.csv')
#     df = pd.DataFrame(news_items)
#     df =pd.concat([df, DF])
#     df = df[['title','description','Date','Source']]
#     return df



# def Travel():
        
#     #Creating a set to remove duplicates
#     last_20 = pd.read_csv('Hindu_Travel_Data.csv', usecols= ['title'])
#     added = set(last_20.title)

#     # Extracting latest news
#     url="https://frontline.thehindu.com/other/travel/feeder/default.rss"
#     resp=requests.get(url)
#     soup=BeautifulSoup(resp.content,features="xml")
#     items=soup.findAll('item')
#     item=items[0]
#     news_items=[]
#     for item in items:
#         news_item={}
#         if item.title.text not in added:
#             news_item['title']=item.title.text
#             news_item['description']=item.description.text
#             news_item['Date']= datetime.now()
#             news_item['Source']= 'Hindu'
#             news_items.append(news_item)
#             new.append([news_item['title'] + news_item['description']])
#             inference_dataframe['category'].append('Travel')
#             inference_dataframe['headline'].append( news_item['title'] )
#         #     news_items = {'title': [],
#         #     'description': [],
#         #     'Date':[], 'Source': []}

#     DF = pd.read_csv('Hindu_Travel_Data.csv')
#     df = pd.DataFrame(news_items)
#     df =pd.concat([df, DF])
#     df = df[['title','description','Date','Source']]
#     return df


# def Environment():
        
#     #Creating a set to remove duplicates
#     last_20 = pd.read_csv('Hindu_Env_Data.csv', usecols= ['title'])
#     added = set(last_20.title)

#     # Extracting latest news
#     url="https://frontline.thehindu.com/environment/feeder/default.rss"
#     resp=requests.get(url)
#     soup=BeautifulSoup(resp.content,features="xml")
#     items=soup.findAll('item')
#     item=items[0]
#     news_items=[]
#     for item in items:
#         news_item={}
#         if item.title.text not in added:
#             news_item['title']=item.title.text
#             news_item['description']=item.description.text
#             news_item['Date']= datetime.now()
#             news_item['Source']= 'Hindu'
#             news_items.append(news_item)
#             new.append([news_item['title'] + news_item['description']])
#             inference_dataframe['category'].append('Environment')
#             inference_dataframe['headline'].append( news_item['title'] )
#         #     news_items = {'title': [],
#         #     'description': [],
#         #     'Date':[], 'Source': []}

#     DF = pd.read_csv('Hindu_Env_Data.csv')
#     df = pd.DataFrame(news_items)
#     df =pd.concat([df, DF])
#     df = df[['title','description','Date','Source']]
#     return df


# def World():
        
#     #Creating a set to remove duplicates
#     last_20 = pd.read_csv('Hindu_World_Data.csv', usecols= ['title'])
#     added = set(last_20.title)

#     # Extracting latest news
#     url="https://frontline.thehindu.com/world-affairs/feeder/default.rss"
#     resp=requests.get(url)
#     soup=BeautifulSoup(resp.content,features="xml")
#     items=soup.findAll('item')
#     item=items[0]
#     news_items=[]
#     for item in items:
#         news_item={}
#         if item.title.text not in added:
#             news_item['title']=item.title.text
#             news_item['description']=item.description.text
#             news_item['Date']= datetime.now()
#             news_item['Source']= 'Hindu'
#             news_items.append(news_item)
#             new.append([news_item['title'] + news_item['description']])
#             inference_dataframe['category'].append('World')
#             inference_dataframe['headline'].append( news_item['title'] )
#         #     news_items = {'title': [],
#         #     'description': [],
#         #     'Date':[], 'Source': []}

#     DF = pd.read_csv('Hindu_World_Data.csv')
#     df = pd.DataFrame(news_items)
#     df =pd.concat([df, DF])
#     df = df[['title','description','Date','Source']]
#     return df



# def get_sequences(texts, tokenizer, train=True, max_seq_length=0):
#     sequences = tokenizer.texts_to_sequences(texts)
    
#     if train == True:
#         max_seq_length = np.max(list(map(lambda x: len(x), sequences)))
    
#     sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    
#     return sequences

# def inference(S, model, index_word):
#     out = []
  
#     for i in S:
#         s = get_sequences(i, tokenizer, False, X_train.shape[1])
#         y_pred = np.argmax(AttentionLSTM.predict(s), axis=1)
#         out.append(index_word[y_pred[0]])
#     print(out)
#     return out

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


def Acc_data(model):
    if new != []:
        for i in range(len(new)):
            new[i] = [remove_stopwords(new[i][0])]
        for i in range(len(new)):
            new[i] = [lemmitize(new[i][0])]
        for i in range(len(new)):
            new[i] = [remove_dig(new[i][0])]
        # print(new)
        df = remove_meta_data(new)
        descc = df
        df =  pd.DataFrame(new, columns = ['text'])
        df['text'] = df['text'].apply(lambda x: remove_stopwords(str(x)))
        df['text'] = df['text'].apply(lambda x: lemmitize(x))
        df['text'] = df['text'].apply(lambda x: remove_dig(x))

        inference_dataframe['desc'] = descc
        inference_dataframe['pred_category'] = inference(df, count_vector, model)
        inference_dataframe['timestamp'] = datetime.now()
        inference_dataframe['Source'] = 'Hindu'
        inference_dataframe['Total comments'] = 0
        inference_dataframe['Pos comments'] = 0
        inference_dataframe['Neg comments'] = 0
        inference_dataframe['Neut comments'] = 0
        inference_dataframe['tweet id'] = 0
        inference_dataframe['twitter_handle'] = 0


        DF = pd.read_csv('Acc_Data.csv', usecols= ['headline','desc', 'category', 'pred_category', 'timestamp', 'Source','Total comments','Pos comments',
        'Neg comments', 'Neut comments','tweet id','twitter handle'])
        df = pd.DataFrame(inference_dataframe)
        df =pd.concat([df, DF])
        df = df[['headline','desc', 'category', 'pred_category', 'timestamp', 'Source','Total comments','Pos comments',
        'Neg comments', 'Neut comments','tweet id','twitter handle']]
        df.reset_index(inplace=True)
        return df


if __name__ == "__main__":
    # try:
    #     # X_train = get_sequences(X_train, tokenizer, train=True)
    #     # print(X_train.shape, y_train.shape)
    #     # inputs = Input(shape=(X_train.shape[1],))
    #     # embedding = tf.keras.layers.Embedding(
    #     #     input_dim=10000,
    #     #     output_dim=64
    #     # )(inputs)

    #     # x = (Bidirectional(LSTM(300, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))(embedding)
    #     # # x = Dropout(0.25)(x)
    #     # x = (Bidirectional(LSTM(200, dropout=0.25, recurrent_dropout=0.25)))(embedding)
    #     # x = Dense(150, activation='relu')(x)
    #     # # x = Dropout(0.25)(x)
    #     # x = BatchNormalization()(x)
    #     # x = Dense(60, activation='relu')(x)
    #     # # x = Dropout(0.25)(x)
    #     # x = BatchNormalization()(x)
    #     # outp = Dense(len(set(y_train)), activation='softmax')(x)

    #     # AttentionLSTM = Model(inputs= inputs, outputs=outp)
    #     # AttentionLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # except Exception as e:
    #     logging.error(e, exc_info = True)


    # try:
    #     AttentionLSTM.load_weights(model_path)
    # except Exception as e:
    #     logging.error(e, exc_info = True)


    try:
        if 'Hindu_Politics_Data.csv' not in os.listdir():
            d = pd.DataFrame({'title': [],'desc': [] ,'Date': [],'Source': []})
            d.to_csv('Hindu_Politics_Data.csv')

        # if 'Hindu_Edu_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('Hindu_Edu_Data.csv')
        
        if 'Hindu_Ent_Data.csv' not in os.listdir():
            d = pd.DataFrame({'title': [],'desc': [] ,'Date': [],'Source': []})
            d.to_csv('Hindu_Ent_Data.csv')

        # if 'Hindu_Health_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('Hindu_Health_Data.csv')

        if 'Hindu_Sci_Data.csv' not in os.listdir():
            d = pd.DataFrame({'title': [],'desc': [] ,'Date': [],'Source': []})
            d.to_csv('Hindu_Sci_Data.csv')

        # if 'Hindu_Sports_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('Hindu_Sports_Data.csv')
            
        # if 'Hindu_Travel_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('Hindu_Travel_Data.csv')

        # if 'Hindu_Env_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('Hindu_Env_Data.csv')

        # if 'Hindu_World_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('Hindu_World_Data.csv')

        if 'Acc_Data.csv' not in os.listdir():
            d = pd.DataFrame({'headline':[],'desc':[], 'category':[], 'pred_category':[], 'timestamp':[], 'Source':[],'Total comments':[],
            'Pos comments':[], 'Neg comments':[], 'Neut comments':[], 'tweet id': [],'twitter handle': [] })
            d.to_csv('Acc_Data.csv')

        
    except Exception as e:
        logging.error(e, exc_info = True)

    import pandas as pd

    try:
        Politics().to_csv('Hindu_Politics_Data.csv')
        # Education().to_csv('Hindu_Edu_Data.csv')
        Entertainment().to_csv('Hindu_Ent_Data.csv')
        # Health().to_csv('Hindu_Health_Data.csv')
        Science().to_csv('Hindu_Sci_Data.csv')
        # Sports().to_csv('Hindu_Sports_Data.csv')
        # Travel().to_csv('Hindu_Travel_Data.csv')  
        # Environment().to_csv('Hindu_Env_Data.csv')
        # World().to_csv('Hindu_World_Data.csv')    

        if len(new) != 0:
             Acc_data(naive_bayes).to_csv('Acc_Data.csv')
        else:
            print('No new news yet.')
        new = []
        inference_dataframe = {'headline': [],
        'category': [], 'desc':[] }


    except Exception as e:
        logging.error(e, exc_info = True)


