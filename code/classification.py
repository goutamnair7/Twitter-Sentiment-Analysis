import svm
import csv
from svmutil import *
import featureExtraction
import nltk

classifier = ''

def train_classifier(train_data):
    ''' Train SVM classifier using training data'''
    
    global classifier
    labels = train_data['labels']
    feature_vector = train_data['feature_vector']

    # extending the feature vector
    with open('../data/featuresTraining.data','rt') as f:   
        reader=csv.reader(f, delimiter=' ')
        l=list(reader)

    rownum=0
    for row in l:
        features=[int(i) for i in row[-16:-1]]
        feature_vector[rownum].extend(features)
        rownum+=1


    problem = svm_problem(labels, feature_vector)
    param = svm_parameter('-q')  #suppress console output
    param.C = 10
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
#        featureList.extend(featureVector)
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
    y = []
    for i in test_data['labels']:
        y.append(float(i))
    test_feature_vector = test_data['feature_vector']

     # extending the feature vector
    with open('../data/featuresTesting.data','rt') as f:   
        reader=csv.reader(f, delimiter=' ')
        l=list(reader)

    rownum=0
    for row in l:
        features=[int(i) for i in row[-16:-1]]
        test_feature_vector[rownum].extend(features)
        rownum+=1

    p_labels, p_accs, p_vals = svm_predict(y, test_feature_vector, classifier)
    return p_labels, p_accs, p_vals

def get_confusion_matrix(labels, test_data):
    ''' Confusion Matrix '''

    matrix = {0:{0:0, 1:0, 2:0}, 1:{0:0, 1:0, 2:0}, 2:{0:0, 1:0, 2:0}}
    l = len(labels)
    for i in range(l):
        matrix[int(test_data['labels'][i])][int(labels[i])] += 1

    return matrix

def main():
    ''' Control function for classification of tweets '''

    train_data = featureExtraction.main()
    train_classifier(train_data)
    test_data = obtain_test_data()
    labels, accuracy, values = test_classifier(test_data)
    matrix = get_confusion_matrix(labels, test_data)

    print "\nConfusion Matrix: "
    print "\t%6d\t|%2d   |%2d   |" % (0, 1, 2)
    print "\t-----------------------------"
    for i in range(0, 3):
        print "\t", i, "|",
        for j in range(0, 3):
            print "%4d|" % matrix[i][j],
        print
    
    print "\nPrecision: "
    sum_precision = 0
    p = []
    for i in range(0, 3):
        print "\tLabel " + str(i) + ":",
        tp = matrix[i][i]
        fp = 0
        for j in range(0, 3):
            if j != i:
                fp += matrix[j][i] 
        precision = float((tp*1.0)/(tp + fp))
        sum_precision += precision
        p.append(precision)
        print precision
    avg_precision = float(sum_precision)/3.0
    print "Average precision: ", avg_precision

    print "\nRecall: "
    sum_recall = 0
    r = []
    for i in range(0, 3):
        print "\tLabel " + str(i) + ":",
        tp = matrix[i][i]
        fn = 0
        for j in range(0, 3):
            if j != i:
                fn += matrix[i][j] 
        recall = float((tp*1.0)/(tp + fn))
        sum_recall += recall
        r.append(recall)
        print recall
    avg_recall = float(sum_recall)/3.0
    print "Average recall: ", avg_recall

    print "\nF1 score: "
    sum_score = 0
    for i in range(0, 3):
        print "\tLabel " + str(i) + ":",
        den = p[i] + r[i]
        num = 2 * p[i] * r[i]
        score = float(num)/den
        sum_score += score
        print score
    avg_score = float(sum_score)/3.0
    print "Average F1 score: ", avg_score

if __name__ == '__main__':
    main()
