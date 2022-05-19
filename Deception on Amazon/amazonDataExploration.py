# coding: utf-8

import csv                               # csv reader
import re
import nltk
import nltk.stem
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.metrics import accuracy_score,average_precision_score,f1_score,recall_score,classification_report
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from textstat.textstat import textstat
from nltk.corpus import stopwords
from string import punctuation


################
## QUESTION 3 ##
################


# load data from a file and append it to the rawData
def loadData(path, Text=None):
    with open(path, encoding="utf8") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for line in reader:
            (Id, Text, Label, Rating, Verified, ProductTitle, ReviewTitle) = parseReview(line)
            rawData.append((Id, Text, Label, Rating, Verified, ProductTitle, ReviewTitle))
            preprocessedData.append((Id, preProcess(Text), Label, Rating, Verified, ProductTitle, ReviewTitle))

positiveRating = 'positive'
negativeRating = 'negative'

def splitData(percentage):
    dataSamples = len(rawData)
    halfOfData = int(len(rawData)/2)
    trainingSamples = int((percentage*dataSamples)/2)
    for (_, Text, Label, Rating, Verified, ProductTitle, ReviewTitle) in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
        if int(Rating)>3:
         trainData.append((toFeatureVector(preProcess(Text), Rating, Verified, ProductTitle, ReviewTitle),positiveRating))
        elif int(Rating)<3:
         trainData.append((toFeatureVector(preProcess(Text), Rating, Verified, ProductTitle, ReviewTitle),negativeRating))
    for (_, Text, Label, Rating, Verified, ProductTitle, ReviewTitle) in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
        if int(Rating)>3:
         testData.append((toFeatureVector(preProcess(Text), Rating, Verified, ProductTitle, ReviewTitle),positiveRating))
        elif int(Rating)<3:
         testData.append((toFeatureVector(preProcess(Text), Rating, Verified, ProductTitle, ReviewTitle),negativeRating))



# Convert line from input file into an id/text/rating/verified/producttitle/label tuple
def parseReview(reviewLine):
    # Should return a triple of an integer, a string containing the review, and a string indicating the label
    Id, Text, Label = reviewLine[0], reviewLine[8], reviewLine[1]
    Rating, Verified, ProductTitle, ReviewTitle = reviewLine[2], reviewLine[3], reviewLine[6], reviewLine[7]
    return (Id, Text, Label, Rating, Verified, ProductTitle, ReviewTitle)



# TEXT PREPROCESSING AND FEATURE VECTORIZATION

# Input: a string of one review
def preProcess(text):
    # Should return a list of tokens

    #word tokenisation
    text = re.sub(r"(^|\s)(http|www\.)\S+", r" ", text) #remove URLs
    text = re.sub(r"(^|\s)@\S+", r" ", text) #remove usernames 
    text = re.sub(r"[\\.,;:!?'\“””\(\)]", r" ", text) #remove punctuation
    text = re.sub(r"[^\w#]", r" ", text) #remove any non-alphanumeric characters

    #normalisation
    text = text.lower()

    #lematization,bigrams
    lem_vectorizer = LemmatizedCountVectorizer(ngram_range=(1,4),min_df=1,stop_words='english')# instance CountVectorizer class

    X = lem_vectorizer.fit_transform([text]) #extract bag of words representation
    
    tokens = lem_vectorizer.get_feature_names() #output list of tokens
    return tokens

class LemmatizedCountVectorizer(CountVectorizer):

   def build_analyzer(self):
     english_stemmer = nltk.stem.WordNetLemmatizer()
     analyzer = super(LemmatizedCountVectorizer,self).build_analyzer()
     return lambda doc:(english_stemmer.lemmatize(w)for w in analyzer(doc))


####################

featureDict = Counter({}) # A global dictionary of features

def toFeatureVector(tokens,rating,verified,producttitle,reviewtitle):
    # Should return a dictionary containing features as keys, and weights as values
    #token as key, number of occurences as values
    featureVec =  Counter({word: (tokens.count(word)/len(tokens)) for word in tokens})
    featureVec['Rating'] = rating
    featureVec['Verified'] = verified
    featureVec['ProductTitle'] = producttitle
    featureVec['ReviewTitle'] = reviewtitle
    featureDict.update(featureVec) #update dictionary of features
    return featureVec


# TRAINING AND VALIDATING OUR CLASSIFIER
def trainClassifier(trainData):
    print("Training Classifier...")
    pipeline =  Pipeline([('svc', LinearSVC())])
    return SklearnClassifier(pipeline).train(trainData)


####################

def crossValidate(dataset, folds):
    shuffle(dataset)
    cv_results = []
    foldSize = len(dataset)//folds
    for i in range(0,len(dataset),foldSize):
        #continue # Replace by code that trains and tests on the 10 folds of data in the dataset
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


#############################
## PART B DATA EXPLORATION ##
#############################


################
## QUESTION 2 ##
################

# List of fake reviews and their product names, and real reviews and their product names

fakeReview = []
realReview = []
fakeProductName = []
realProductName = []

for x in rawData:
 if x[2] =="__label1__":
  fakeReview.append(x[1])
  fakeProductName.append(x[5])
 elif x[2] =="__label2__":
  realReview.append(x[1])
  realProductName.append(x[5])

#Review length
#Complex vocabulary, Flesch-Kincaid readability test


reviewLength1 = 0
reviewLength2 = 0
readingScore1 = 0
readingScore2 = 0
stopCount1 = 0
stopCount2 = 0

stopWords = set(stopwords.words('english'))

for text in fakeReview:
 reviewLength1 = reviewLength1 + len(text)
 readingScore1  = readingScore1 + textstat.flesch_reading_ease(text)
 words = text.split(" ")
 for word in words:
  if word in stopWords:
    stopCount1 = stopCount1 + 1

for text in realReview:
 reviewLength2 = reviewLength2 + len(text)
 readingScore2  = readingScore2 + textstat.flesch_reading_ease(text)
 words = text.split(" ")
 for word in words:
  if word in stopWords:
    stopCount2 = stopCount2 + 1


print("Average review length for label 1:" + str(reviewLength1/len(fakeReview)))
print("Average review length for label 2:" + str(reviewLength2/len(realReview)))
print("Average readability test for label 1:" + str(readingScore1/len(fakeReview)))
print("Average readability test for label 2:" + str(readingScore2/len(realReview)))
print("Number of stopwords for label 1:" + str(stopCount1))
print("Number of stopwords for label 2:" + str(stopCount2))

#capitalisation

capitalCount1 = 0
capitalCount2 = 0

for text in fakeReview:
 words = text.split(" ")
 for word in words:
  for character in word:
   if character.isupper():
    capitalCount1 = capitalCount1 + 1
   
for text in realReview:
 words = text.split(" ")
 for word in words:
  for character in word:
   if character.isupper():
    capitalCount2 = capitalCount2 + 1

print("Number of capitals for label 1:" + str(capitalCount1))
print("Number of capitals for label 2:" + str(capitalCount2))

#punctuation

punctuationCount1 = 0
punctuationCount2 = 0

for text in fakeReview:
 words = text.split(" ")
 for word in words:
  for character in word:
   if character in punctuation:
    punctuationCount1 = punctuationCount1 + 1
   
for text in realReview:
 words = text.split(" ")
 for word in words:
  for character in word:
   if character in punctuation:
    punctuationCount2 = punctuationCount2 + 1

print("Number of punctuations for label 1:" + str(punctuationCount1))
print("Number of punctuations for label 2:" + str(punctuationCount2))


#product name

wordcount1 = 0
wordcount2 = 0

for text in fakeReview:
 for name in fakeProductName:
  if name in text:
   wordcount1 = wordcount1 + 1
   
for text in realReview:
 for name in realProductName:
  if name in text:
   wordcount2 = wordcount2 + 1
   break

print("Number times product name in review for label 1:" + str(wordcount1))
print("Number times product name in review for label 2:" + str(wordcount2))

