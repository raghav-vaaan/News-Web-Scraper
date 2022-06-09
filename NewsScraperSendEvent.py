from datetime import datetime
import logging
import json
logging.basicConfig(filename=datetime.now().strftime('%d_%m_%Y.log'),
                    filemode='a',
                    format='%(asctime)s:%(msecs)d-%(name)s-%(levelname)s-%(message)s',
                    datefmt='%d_%m_%Y_%H:%M:%S',level=logging.DEBUG)

try:
    import requests
    import pandas as pd
    import time 
    import json
    import xml.etree.ElementTree as ET
except Exception as e:
    logging.error(e, exc_info=True)
    
try:
    tree = ET.parse('WebScraper.xml')
    root = tree.getroot()
    config_data = []
    for elem in root:
        for subelem in elem:
            config_data.append(subelem.text)
    DestinationURL = config_data[0]

except Exception as e:
    logging.error(e, exc_info=True)

print(DestinationURL)
LastCount = 0


def SendEvent():   
    for index, row in df1.iterrows():
        time.sleep(1)
        SNo = row['SNo']
        #Time = row['timestamp']
        Time = datetime.now().strftime('%d-%b-%Y %H:%M:%S')
        Source = row['Source']
        Title = row['headline']
        Description = row['headline']
        Category = row['pred_category']
        Totalreview = row['Total']
        PositiveReview = row['Positive']
        NegativeReview = row['Negative']
        NeutralReview = row['Neutral']
        Totalreview = 100
        PositiveReview = 60
        NegativeReview = 20
        NeutralReview = 20
        header={'Content-Type': 'text/json'}
        #header = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"} 
        my_json_string = json.dumps(dict({'SN': SNo, 'Timestamp': Time,'Source':Source,'Title':Title,'Description':Description,'Category':Category,'TotalReview':Totalreview,'PositiveReview':PositiveReview,'NegativeReview':NegativeReview,'NutralReview':NeutralReview}))
        #print(my_json_string)
        #my_json_string = {"SN": 1000, "Timestamp": "15-Sep-2021 13:17:56", "Source": "TOI", "Title": "5th Test: Bumrah's workload, Rahane's form big concerns as India eye history", "Description": "5th Test: Bumrah's workload, Rahane's form big concerns as India eye history", "Category": "Entertainment", "TotalReview": 100, "PositiveReview": 60, "NegativeReview": 20, "NutralReview": 20}
        #r = requests.post(DestinationURL, data=my_json_string,headers=header)
        print(my_json_string)
        #header = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        #r = requests.post(DestinationURL, json.dumps(my_json_string),headers=header)
        r = requests.post(DestinationURL,my_json_string,headers=header)
        print(r.text)
        print(r.status_code, r.reason)
        time.sleep(1)
    
while True:
    df = pd.read_csv("Acc_Data.csv")
    Count = len(df)
    new_text = Count-LastCount
    #print(df.head(2)) 
    print(df.dtypes)
    LastCount = Count
    df1 = df.tail(LastCount)
    #print(df1.head(10))
    if new_text>0:
        #pass
        SendEvent()
    time.sleep(1000)
   