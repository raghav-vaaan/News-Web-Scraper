import twitter
import os
from  config import *
import logging
import time

logging.basicConfig(filename='twiiter.log',
                    filemode='a',
                    format='%(asctime)s:%(msecs)d-%(name)s-%(levelname)s-%(message)s',
                    datefmt='%d-%m-%y %H:%M:%S',level=logging.DEBUG)
while True:
    if run_TOI:
        print( )
        print('Executing TOI ...............................................................')
        print( )
        os.system('python TOI_category.py')

    if run_Hindu:
        print( )
        print('Executing Hindu ...............................................................')
        print( )
        os.system('python Hindu_scraper.py')


    if run_twitter_id:
        print( )
        print('Executing twitter id................................................................')
        try:
            twitter.twitter_id(id_user, id_count, True )
        except Exception as e:
            logging.error(e, exc_info = True)
        

    if run_twitter_keywords:
        print('Executing twitter keyword..................................................................')
        print( )
        
        try:
            twitter.twitter_keywords(keyword, keyword_count, True)
        except Exception as e:
            logging.error(e, exc_info = True)
    
    print('All done for this slot.')
    time.sleep(3600)

