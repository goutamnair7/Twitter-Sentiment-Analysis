import svm
import csv
from svmutil import *
import featureExtraction

classifier = ''

def train_classifier(train_data):
    ''' Train SVM classifier using training data'''
    
    global classifier
    labels = train_data['labels']
    feature_vector = train_data['feature_vector']

    problem = svm_problem(labels, feature_vector)
    param = svm_parameter('-q')  #suppress console output
    param.kernel_type = LINEAR
    classifier = svm_train(problem, param)

def obtain_test_data():
    ''' Obtain testing data for the classifier'''
    
    featureList = []
    tweets = []
    with open('../data/preprocessedTesting.data','rb') as f:   
        reader = csv.reader(f, delimiter='\t')
        l = list(reader)

    for row in l:
        sentiment = row[2]
        tweet = row[3]
        tweet = tweet[:-2]            # ignoring hashcounts
        featureVector = featureExtraction.getFeatureVector(tweet)
        #featureList.extend(featureVector)
        tweets.append((featureVector, sentiment))

    with open('../data/preprocessedTraining.data','rb') as f:   
        reader = csv.reader(f, delimiter='\t')
        l = list(reader)
    for row in l:
        sentiment = row[2]
        tweet = row[3]
        tweet = tweet[:-2]            # ignoring hashcounts
        featureVector = featureExtraction.getFeatureVector(tweet)
        featureList.extend(featureVector)
    # Remove featureList duplicates
    featureList = list(set(featureList))    
    result = featureExtraction.getFeatureListAndLabels(tweets, featureList)
    return result

def test_classifier(test_data):
    ''' Classification of test data'''

    global classifier
    l = len(test_data['labels'])
    y = [0]*l
    test_feature_vector = test_data['feature_vector']
    p_labels, p_accs, p_vals = svm_predict(y, test_feature_vector, classifier)
    return p_labels, p_accs, p_vals

def main():
    ''' Control function for classification of tweets '''

    train_data = featureExtraction.main()
    train_classifier(train_data)
    test_data = obtain_test_data()
    labels, accuracy, values = test_classifier(test_data)
    print accuracy 	

if __name__ == '__main__':
    main()
