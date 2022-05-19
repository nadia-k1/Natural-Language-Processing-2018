# coding: utf-8

import csv                               # csv reader
import re
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.metrics import accuracy_score,average_precision_score,f1_score,recall_score,classification_report
from sklearn.metrics import precision_recall_fscore_support


# load data from a file and append it to the rawData
def loadData(path, Text=None):
    with open(path, encoding="utf8") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for line in reader:
            (Id, Text, Label) = parseReview(line)
            rawData.append((Id, Text, Label))
            preprocessedData.append((Id, preProcess(Text), Label))

def splitData(percentage):
    dataSamples = len(rawData)
    halfOfData = int(len(rawData)/2)
    trainingSamples = int((percentage*dataSamples)/2)
    for (_, Text, Label) in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
        trainData.append((toFeatureVector(preProcess(Text)),Label))
    for (_, Text, Label) in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
        testData.append((toFeatureVector(preProcess(Text)),Label))


################
## QUESTION 1 ##
################

# Convert line from input file into an id/text/label tuple
def parseReview(reviewLine):
    # Should return a triple of an integer, a string containing the review, and a string indicating the label
    Id, Text, Label = reviewLine[0], reviewLine[8], reviewLine[1]
    return (Id, Text, Label)


# TEXT PREPROCESSING AND FEATURE VECTORIZATION

# Input: a string of one review
def preProcess(text):
    # Should return a list of tokens
    # normalisation
    text = text.lower()
    # word tokenisation
    tokens = re.split(r"\s+",text)#output a list of tokens
    return tokens


################
## QUESTION 2 ##
################
featureDict = Counter({}) # A global dictionary of features

def toFeatureVector(tokens):
    # Should return a dictionary containing features as keys, and weights as values
    #token as key, number of occurences as values
    featureVec =  Counter({word: tokens.count(word) for word in tokens})
    featureDict.update(featureVec) #update dictionary of features
    return featureVec


# TRAINING AND VALIDATING OUR CLASSIFIER
def trainClassifier(trainData):
    print("Training Classifier...")
    pipeline =  Pipeline([('svc', LinearSVC())])
    return SklearnClassifier(pipeline).train(trainData)

################
## QUESTION 3 ##
################

def crossValidate(dataset, folds):
    shuffle(dataset)
    cv_results = []
    foldSize = len(dataset)//folds
    for i in range(0,len(dataset),foldSize):
       # continue # Replace by code that trains and tests on the 10 folds of data in the dataset
        #split into train and test data
        print ("fold start %d foldSize %d" % (i, foldSize))
        myTestData = dataset[i:i+foldSize]
        myTrainData = dataset[:i] + dataset[i+foldSize:]

        #train classifier
        classifier = trainClassifier(myTrainData)

        #store correct target values
        y_true = list(map(lambda x: x[1], myTestData))

        #estimated targets as predicted
        y_pred = predictLabels(myTestData, classifier)
        #print(y_pred)
        
        #model performance metrics
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

    cv_results.append("Average precision: " + str(round(sum(precision)/len(precision),4)))
    cv_results.append("Average Recall: "  + str(round(sum(recall)/len(recall),4)))
    cv_results.append("Average F1 score:" + str(round(sum(fscore)/len(fscore),4)))
    cv_results.append("Accuracy score: " + str(round(accuracy,4)))
    return cv_results

# PREDICTING LABELS GIVEN A CLASSIFIER

def predictLabels(reviewSamples, classifier):
    return classifier.classify_many(map(lambda t: t[0], reviewSamples))


def predictLabel(reviewSample, classifier):
    return classifier.classify(toFeatureVector(preProcess(reviewSample)))


# MAIN

# loading reviews
rawData = []          # the filtered data from the dataset file (should be 21000 samples)
preprocessedData = [] # the preprocessed reviews (just to see how your preprocessing is doing)
trainData = []        # the training data as a percentage of the total dataset (currently 80%, or 16800 samples)
testData = []         # the test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# the output classes
fakeLabel = 'fake'
realLabel = 'real'

# references to the data files
reviewPath = 'amazon_reviews.txt'

## Do the actual stuff
# We parse the dataset and put it in a raw data list
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Preparing the dataset...",sep='\n')
loadData(reviewPath)
# We split the raw dataset into a set of training data and a set of test data (80/20)
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Preparing training and test data...",sep='\n')
splitData(0.8)
# We print the number of training samples and the number of features
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Training Samples: ", len(trainData), "Features: ", len(featureDict), sep='\n')

folds = 10
cv_results = crossValidate(trainData, folds)
for x in cv_results:
 print(x)
