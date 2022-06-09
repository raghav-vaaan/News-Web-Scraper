from operator import ne         #     
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
logging.basicConfig(filename='TOI_category.log',
                    filemode='a',
                    format='%(asctime)s:%(msecs)d-%(name)s-%(levelname)s-%(message)s',
                    datefmt='%d-%m-%y %H:%M:%S',level=logging.DEBUG)

try:
    with open(path_naive_bayes, 'rb') as handle:			# Opening Path file naive_bayes present in desktop
        naive_bayes = pickle.load(handle)
    with open(path_count_vec, 'rb') as handle:			    # Opening another count_vec file present in desktop
        count_vector = pickle.load(handle)

    new = []
    inference_dataframe = {'headline': [],
    'category': [], 'desc':[] }
except Exception as e:
    logging.error(e, exc_info = True)


def find_strings(keywords , news_string):
  """keywords: arr or keywords
  news_string: one news string
  """
  for j in keywords:
    sub = j
    if str(news_string).find(sub) != -1:
      return True
    return False


# def Buisness():
        
#     #Creating a set to remove duplicates
#     last_20 = pd.read_csv('Buis_Data.csv', usecols= ['title'])
#     added = set(last_20.title)

#     # Extracting latest news
#     url="https://timesofindia.indiatimes.com/rssfeeds/1898055.cms"        # Fetching data from url
#     resp=requests.get(url)												# Get request  from url
#     soup=BeautifulSoup(resp.content,features="xml")						# Get features
#     items=soup.findAll('item')											
#     item=items[0]
#     news_items=[]
#     for item in items:
#         news_item={}
#         if item.title.text not in added: 									#Agar item ke title ka text not in added 
#             news_item['title']=item.title.text							# Toh update karenge 
#             news_item['description']=item.description.text
#             news_item['Date']= datetime.now()								# Abhi ka date time
#             news_item['Source']= 'TOI'								     # Times of india hi rahega
#             news_items.append(news_item)										# List mein append kar denge 
#             new.append([news_item['title'] + news_item['description']])		# Phir new mein append karenge title aur description
#             inference_dataframe['category'].append('Buisness')                 # Ek nya dataframe inference mein category naam se append karenge business  
#             inference_dataframe['headline'].append( news_item['title'] )			
#         # else:
#         #     news_item = {'title': [],
#         #     'description': [],
#         #     'Date':[], 'Source': []}

#     DF = pd.read_csv('Buis_Data.csv')
#     df = pd.DataFrame(news_items)
#     df =pd.concat([df, DF])
#     df = df[['title','description','Date','Source']]
#     return df

# def Cricket():
#     #Creating a set to remove duplicates
#     last_20 = pd.read_csv('Crick_Data.csv', usecols= ['title'])
#     added = set(last_20.title)

#     # Extracting latest news
#     url="https://timesofindia.indiatimes.com/rssfeeds/54829575.cms"
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
#             news_item['Source']= 'TOI'
#             news_items.append(news_item)
#             new.append([news_item['title'] + news_item['description']])
#             inference_dataframe['category'].append('Cricket')
#             inference_dataframe['headline'].append( news_item['title'] )

#         # else:
#         #     news_item = {'title': [],
#         #     'description': [],
#         #     'Date':[], 'Source': []}
    
#     DF = pd.read_csv('Crick_Data.csv')
#     df = pd.DataFrame(news_items)
#     df =pd.concat([df, DF])
#     df = df[['title','description','Date','Source']]
#     return df

# def Education():
        
#     #Creating a set to remove duplicates
#     last_20 = pd.read_csv('Edu_Data.csv', usecols= ['title'])
#     added = set(last_20.title)

#     # Extracting latest news
#     url="https://timesofindia.indiatimes.com/rssfeeds/913168846.cms"
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
#             news_item['Source']= 'TOI'
#             news_items.append(news_item)
#             new.append([news_item['title'] + news_item['description']])
#             inference_dataframe['category'].append('Education')
#             inference_dataframe['headline'].append( news_item['title'] )
#         # else:
#         #     news_items = {'title': [],
#         #     'description': [],
#         #     'Date':[], 'Source': []}

#     DF = pd.read_csv('Edu_Data.csv')
#     df = pd.DataFrame(news_items)
#     df =pd.concat([df, DF])
#     df = df[['title','description','Date','Source']]
#     return df
    
def Entertainment():
        
    #Creating a set to remove duplicates
    last_20 = pd.read_csv('Ent_Data.csv', usecols= ['title'])
    added = set(last_20.title)

    # Extracting latest news
    url="https://timesofindia.indiatimes.com/rssfeeds/54829575.cms"
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
            news_item['Source']= 'TOI'
            news_items.append(news_item)
            new.append([news_item['title'] + news_item['description']])
            inference_dataframe['category'].append('Entertainment')
            inference_dataframe['headline'].append( news_item['title'] )
            inference_dataframe['desc'].append(news_item['description'])
        # else:
        #     news_items = {'title': [],
        #     'description': [],
        #     'Date':[], 'Source': []}

    DF = pd.read_csv('Ent_Data.csv')
    df = pd.DataFrame(news_items)
    df =pd.concat([df, DF])
    df = df[['title','desc','Date','Source']]
    return df

# def Health():
        
#     #Creating a set to remove duplicates
#     last_20 = pd.read_csv('Health_Data.csv', usecols= ['title'])
#     added = set(last_20.title)

#     # Extracting latest news
#     url="https://timesofindia.indiatimes.com/rssfeeds/3908999.cms"
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
#             news_item['Source']= 'TOI'
#             news_items.append(news_item)
#             new.append([news_item['title'] + news_item['description']])
#             inference_dataframe['category'].append('Health')
#             inference_dataframe['headline'].append( news_item['title'] )
#         # else:
#         #     news_items = {'title': [],
#         #     'description': [],
#         #     'Date':[], 'Source': []}

#     DF = pd.read_csv('Health_Data.csv')
#     df = pd.DataFrame(news_items)
#     df =pd.concat([df, DF])
#     df = df[['title','description','Date','Source']]
#     return df

# def Life_Style():
        
#     #Creating a set to remove duplicates
#     last_20 = pd.read_csv('LS_Data.csv', usecols= ['title'])
#     added = set(last_20.title)

#     # Extracting latest news
#     url = 'https://timesofindia.indiatimes.com/rssfeeds/2886704.cms'
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
#             news_item['Source']= 'TOI'
#             news_items.append(news_item)
#             new.append([news_item['title'] + news_item['description']])
#             inference_dataframe['category'].append('Lifestyle')
#             inference_dataframe['headline'].append( news_item['title'] )
#         # else:
#         #     news_items = {'title': [],
#         #     'description': [],
#         #     'Date':[], 'Source': []}

#     DF = pd.read_csv('LS_Data.csv')
#     df = pd.DataFrame(news_items)
#     df =pd.concat([df, DF])
#     df = df[['title','description','Date','Source']]
#     return df

def Science():
        
    #Creating a set to remove duplicates
    last_20 = pd.read_csv('Sci_Data.csv', usecols= ['title'])
    added = set(last_20.title)

    # Extracting latest news
    url="https://timesofindia.indiatimes.com/rssfeeds/-2128672765.cms"
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
            news_item['Source']= 'TOI'
            news_items.append(news_item)
            new.append([news_item['title'] + news_item['description']])
            inference_dataframe['category'].append('Science')
            inference_dataframe['headline'].append( news_item['title'] )
            inference_dataframe['desc'].append(news_item['description'])
            
        #     news_items = {'title': [],
        #     'description': [],
        #     'Date':[], 'Source': []}

    DF = pd.read_csv('Sci_Data.csv')
    df = pd.DataFrame(news_items)
    df =pd.concat([df, DF])
    df = df[['title','desc','Date','Source']]
    return df

# def Sports():
        
#     #Creating a set to remove duplicates
#     last_20 = pd.read_csv('Sports_Data.csv', usecols= ['title'])
#     added = set(last_20.title)

#     # Extracting latest news
#     url="https://timesofindia.indiatimes.com/rssfeeds/4719148.cms"
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
#             news_item['Source']= 'TOI'
#             news_items.append(news_item)
#             new.append([news_item['title'] + news_item['description']])
#             inference_dataframe['category'].append('Sports')
#             inference_dataframe['headline'].append( news_item['title'] )
#         # else:
#         #     news_items = {'title': [],
#         #     'description': [],
#         #     'Date':[], 'Source': []}

#     DF = pd.read_csv('Sports_Data.csv')
#     df = pd.DataFrame(news_items)
#     df =pd.concat([df, DF])
#     df = df[['title','description','Date','Source']]
#     return df

# def Tech():
        
#     #Creating a set to remove duplicates
#     last_20 = pd.read_csv('Tech_Data.csv', usecols= ['title'])
#     added = set(last_20.title)

#     # Extracting latest news
#     url="https://timesofindia.indiatimes.com/rssfeeds/66949542.cms"
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
#             news_item['Source']= 'TOI'
#             news_items.append(news_item)
#             new.append([news_item['title'] + news_item['description']])
#             inference_dataframe['category'].append('Tech')
#             inference_dataframe['headline'].append( news_item['title'] )
#         # else:
#         #     news_items = {'title': [],
#         #     'description': [],
#         #     'Date':[], 'Source': []}

#     DF = pd.read_csv('Tech_Data.csv')
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
#         print(y_pred)
#         # print(y_pred)
#         out.append(index_word[y_pred[0]])
#     print(out)
#     return out


import re
def remove_meta_data(S):
  for i in range(len(S)):
    s = str(S[i][0])
    s = re.sub('<[^>]+>', '', s)
    S[i] = s
  return S

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



def Acc_data(model):
    if new!= []:
        for i in range(len(new)):
            new[i] = [remove_stopwords(new[i][0])]
        for i in range(len(new)):
            new[i] = [lemmitize(new[i][0])]
        for i in range(len(new)):
            new[i] = [remove_dig(new[i][0])]
        tmp = []
        for i in inference_dataframe['desc']:
            tmp.append([i])
        descc = remove_meta_data(tmp)
        df = remove_meta_data(new)
        df =  pd.DataFrame(new, columns = ['text'])
        df['text'] = df['text'].apply(lambda x: remove_stopwords(str(x)))
        df['text'] = df['text'].apply(lambda x: lemmitize(x))
        df['text'] = df['text'].apply(lambda x: remove_dig(x))
        print(inference_dataframe)
        inference_dataframe['desc'] = descc
        inference_dataframe['pred_category'] = inference(df, count_vector, model)
        inference_dataframe['timestamp'] = datetime.now()
        inference_dataframe['Source'] = 'TOI'
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
    #     X_train = get_sequences(X_train, tokenizer, train=True)
    #     print(X_train.shape, y_train.shape)
    #     inputs = Input(shape=(X_train.shape[1],))
    #     embedding = tf.keras.layers.Embedding(
    #         input_dim=10000,
    #         output_dim=64
    #     )(inputs)

    #     x = (Bidirectional(LSTM(300, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))(embedding)
    #     # x = Dropout(0.25)(x)
    #     x = (Bidirectional(LSTM(200, dropout=0.25, recurrent_dropout=0.25)))(embedding)
    #     x = Dense(150, activation='relu')(x)
    #     # x = Dropout(0.25)(x)
    #     x = BatchNormalization()(x)
    #     x = Dense(60, activation='relu')(x)
    #     # x = Dropout(0.25)(x)
    #     x = BatchNormalization()(x)
    #     outp = Dense(len(set(y_train)), activation='softmax')(x)

    #     AttentionLSTM = Model(inputs= inputs, outputs=outp)
    #     AttentionLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # except Exception as e:
    #     logging.error(e, exc_info = True)


    # try:
    #     AttentionLSTM.load_weights(model_path)


    # except Exception as e:
    #     logging.error(e, exc_info = True)


    try:
        # if 'Buis_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('Buis_Data.csv')

        # if 'Crick_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('Crick_Data.csv')

        # if 'Edu_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('Edu_Data.csv')

        if 'Ent_Data.csv' not in os.listdir():
            d = pd.DataFrame({'title': [],'desc': [] ,'Date': [],'Source': []})
            d.to_csv('Ent_Data.csv')

        # if 'Health_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('Health_Data.csv')

        # if 'LS_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('LS_Data.csv')

        if 'Sci_Data.csv' not in os.listdir():
            d = pd.DataFrame({'title': [],'desc': [] ,'Date': [],'Source': []})
            d.to_csv('Sci_Data.csv')

        # if 'Tech_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('Tech_Data.csv')

        # if 'Sports_Data.csv' not in os.listdir():
        #     d = pd.DataFrame({'title': [],'description': [] ,'Date': [],'Source': []})
        #     d.to_csv('Sports_Data.csv')

        if 'Acc_Data.csv' not in os.listdir():
            d = pd.DataFrame({'headline':[],'desc':[], 'category':[], 'pred_category':[], 'timestamp':[], 'Source':[],'Total comments':[],
            'Pos comments':[], 'Neg comments':[], 'Neut comments':[], 'tweet id': [],'twitter handle':[] })
            
            d.to_csv('Acc_Data.csv')

    except Exception as e:
        logging.error(e, exc_info = True)

    import pandas as pd


    try:
        # Buisness().to_csv('Buis_Data.csv')
        # Cricket().to_csv('Crick_Data.csv')
        # Education().to_csv('Edu_Data.csv')
        Entertainment().to_csv('Ent_Data.csv')
        # Health().to_csv('Health_Data.csv')
        # Life_Style().to_csv('LS_Data.csv')
        Science().to_csv('Sci_Data.csv')
        # Sports().to_csv('Sports_Data.csv')
        # Tech().to_csv('Tech_Data.csv')       
        if len(new) != 0:
            Acc_data(naive_bayes).to_csv('Acc_Data.csv')
        else:
            print('No new news yet')
        new = []
        inference_dataframe = {'headline': [],
        'category': [], 'desc':[] }

    except Exception as e:
        logging.error(e, exc_info = True)


