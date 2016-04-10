import csv
import re
import nltk
import pickle
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import wordnet as wn

stopWords=['||T||','||U||']

featureList = []
tweets = []

def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:   #Unigram model
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        if(w in stopWords or val is None):
            continue
        featureVector.append(w.lower())

    bigram_finder = BigramCollocationFinder.from_words(words)
    score_fn = BigramAssocMeasures.chi_sq
    bigrams = bigram_finder.nbest(score_fn, 10)

    for tokens in bigrams:  #Bi-gram model
        word1 = tokens[0]
        word2 = tokens[1]
        val1 = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word1)
        val2 = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word2)
        
        if((word1 in stopWords or word2 in stopWords) or (val1 is None or val2 is None)):
            continue
        token = word1.lower() + ' ' + word2.lower()
	print token
        featureVector.append(token)
    
    return featureVector


def getFeatureListAndLabels(tweets, featureList):
    sortedFeatures = sorted(featureList)
    map = {}
    feature_vector = []
    labels = []
    for t in tweets:
        label = 0
        map = {}
        #Initialize empty map
        for w in sortedFeatures:
            map[w] = 0

        tweet_words = t[0]
        tweet_opinion = t[1]
        #Fill the map
        for word in tweet_words:
            word = word.strip('\'"?,.')
            #set map[word] to 1 if word exists
            if word in map:
                map[word] = 1
	    #look for synonyms    
            else:
                flag=0
                for i,j in enumerate(wn.synsets(word)):
                    syns=j.lemma_names
                    for syn in syns:
                        if syn in map:
                            map[syn]=1
                            flag=1
                            break        
                    if flag==1:
                        break      

        #end for loop
        values = map.values()
        feature_vector.append(values)
        if(tweet_opinion == 'positive'):
            label = 0
        elif(tweet_opinion == 'negative'):
            label = 1
        elif(tweet_opinion == 'neutral'):
            label = 2
        labels.append(label)
    #return the list of feature_vector and labels
    return {'feature_vector' : feature_vector, 'labels': labels}


def main():
    global featureList
    with open('../data/preprocessedTraining.data','rb') as f:   
        reader=csv.reader(f, delimiter='\t')
        l=list(reader)

    for row in l:
        sentiment=row[2]
        tweet=row[3]
        tweet=tweet[:-2]            # ignoring hashcounts
        featureVector = getFeatureVector(tweet)
        featureList.extend(featureVector)
        tweets.append((featureVector, sentiment))

    # Remove featureList duplicates
    featureList = list(set(featureList))    
    result = getFeatureListAndLabels(tweets, featureList)
    return result

if __name__ == '__main__':
    main()
