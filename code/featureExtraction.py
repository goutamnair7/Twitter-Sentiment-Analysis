import csv
import re
import nltk
import pickle

stopWords=['||T||','||U||']

featureList = []
tweets = []

def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        if(w in stopWords or val is None):
            continue
        featureVector.append(w.lower())
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
