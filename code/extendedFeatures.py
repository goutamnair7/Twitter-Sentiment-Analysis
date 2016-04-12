import csv
import re
import nltk
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment import SentimentIntensityAnalyzer


#output file is in the following format
'''tweet label positive negative neutral totalpos totalneg totalneu negation hashtag ? * ! capitalized capsPos capsNeg capsNeu'''

with open('../data/temppreprocessedTraining.data','rt') as f:
    reader=csv.reader(f, delimiter='\t')
    l=list(reader)

sid = SentimentIntensityAnalyzer()

f = open("../data/featuresTraining.data", 'w+')


for row in l:
    sentiment=row[2]
    tweet=row[3]
    tweet=tweet[:-2] 
    ss = sid.polarity_scores(tweet)
    f.write(tweet+" "+sentiment+" ")
    #for k in sorted(ss):
    #	print(k, ss[k])

    # positive negative and neatral polarities
    if ss['pos']>0.0:
    	f.write('1 ')
    else:
    	f.write('0 ')	
    if ss['neg']>0.0:
    	f.write('1 ')
    else:
    	f.write('0 ')	
    if ss['neu']>0.0:
    	f.write('1 ')
    else:
    	f.write('0 ')
    
    # 1 0 0 - positive # 0 1 0 - negative # 0 0 1 - neutral
    if ss['pos']>ss['neg']:
    	f.write('1 0 0 ')
    elif ss['neg']>ss['pos']:
    	f.write('0 1 0 ')	
    elif ss['neg']>0 and ss['pos']>0:
    	f.write('0 1 0 ')			
    else:
    	f.write('0 0 1 ')	

    # negation	
    if tweet.find(' not ')>-1 or tweet.find(" n't ")>-1:
    	f.write('1 ')
    else:
    	f.write('0 ')	

    #hashtag	
    if tweet.find('#')>-1:
    	f.write('1 ')	
    else:
    	f.write('0 ') 	

    
    #special characters	
    if tweet.find('?')>-1:
    	f.write('1 ')	
    else:
    	f.write('0 ')     	
    if tweet.find('*')>-1:
    	f.write('1 ')	
    else:
    	f.write('0 ')
    if tweet.find('!')>-1:
    	f.write('1 ')	
    else:
    	f.write('0 ')


    #	capitalized
    flag=0
    caps=""
    for word in tweet.split():
    	if word=='||U||' or word=='||T||':
    		continue
    	if word.isupper():
    		flag=1
    		caps=caps+word+" "
    		
    if flag:
    	f.write('1 ')	
    else:
    	f.write('0 ')    		 	 	

    if caps!="":
   	    ss = sid.polarity_scores(caps)
   	    if ss['pos']>0.0:
   	    	f.write('1 ')
   	    else:
   	    	f.write('0 ')
   	    if ss['neg']>0.0:
   	    	f.write('1 ')
   	    else:
   	    	f.write('0 ')
   	    if ss['neu']>0.0:
   	    	f.write('1 ')
   	    else:
   	    	f.write('0 ')
    else:
    	f.write('0 0 0 ')		

    f.write("\n")	

