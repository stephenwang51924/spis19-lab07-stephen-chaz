import nltk 
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.classify.util import accuracy

# "Stop words" that you might want to use in your project/an extension
stop_words = set(stopwords.words('english'))

def format_sentence(sent):
    ''' format the text setence as a bag of words for use in nltk'''
    tokens = nltk.word_tokenize(sent)
    return({word: True for word in tokens})

def getReviews(data, rating):
    ''' Return the reviews from the rows in the data set with the
        given rating '''
    rows = data['Rating']==rating
    return list(data.loc[rows, 'Review'])


def splitTrainTest(data, trainProp):
    ''' input: A list of data, trainProp is a number between 0 and 1
              specifying the proportion of data in the training set.
        output: A tuple of two lists, (training, testing)
    '''
    return (data[0 : int(len(data) * trainProp)], data[int((len(data) * trainProp)) : len(data)])

def formatForClassifier(dataList, label):
    ''' input: A list of documents represented as text strings
               The label of the text strings.
        output: a list with one element for each doc in dataList,
                where each entry is a list of two elements:
                [format_sentence(doc), label]
    '''
    newList = []
    for sentence in dataList:
        newList.append([format_sentence(sentence), label])
    return newList

def classifyReviews():
    ''' Perform sentiment classification on movie reviews ''' 
    # Read the data from the file
    data = pd.read_csv("data/movieReviews.csv")

    # get the text of the positive and negative reviews only.
    # positive and negative will be lists of strings
    # For now we use only very positive and very negative reviews.
    positive = getReviews(data, 4)
    negative = getReviews(data, 0)

    # Split each data set into training and testing sets.
    # You have to write the function splitTrainTest
    (posTrainText, posTestText) = splitTrainTest(positive, 0.8)
    (negTrainText, negTestText) = splitTrainTest(negative, 0.8)

    # Format the data to be passed to the classifier.
    # You have to write the formatForClassifer function
    posTrain = formatForClassifier(posTrainText, 'pos')
    negTrain = formatForClassifier(negTrainText, 'neg')

    # Create the training set by appending the pos and neg training examples
    training = posTrain + negTrain

    # Format the testing data for use with the classifier
    posTest = formatForClassifier(posTestText, 'pos')
    negTest = formatForClassifier(negTestText, 'neg')
    # Create the test set
    test = posTest + negTest


    # Train a Naive Bayes Classifier
    # Uncomment the next line once the code above is working
    classifier = NaiveBayesClassifier.train(training)

    # Uncomment the next two lines once everything above is working
    print("Accuracy of the classifier is: " + str(accuracy(classifier, test)))
    classifier.show_most_informative_features()

    # Calculate and print the accuracy on the positive and negative
    # documents separately
    # You will want to use the function classifier.classify, which takes
    # a document formatted for the classifier and returns the classification
    # of that document ("pos" or "neg").  For example:
    # classifier.classify(format_sentence("I love this movie. It was great!"))
    # will (hopefully!) return "pos"
    numPos = 0
    numNeg = 0
    for review in positive:
        if classifier.classify(format_sentence(review)) == "pos":
            numPos += 1
    for review in negative:
        if classifier.classify(format_sentence(review)) == "neg":
            numNeg += 1
    print("Accuracy of Positive: " + str(numPos / len(positive)))
    print("Accuracy of Negative: " + str(numNeg / len(negative)))

    # Prints two lists with all of the misclassified positive reviews and misclassified negative reviews.
    wrongPosList = []
    wrongNegList = []
    for review in positive:
        if classifier.classify(format_sentence(review)) == "neg":
            wrongPosList.append(review)
    for review in negative:
        if classifier.classify(format_sentence(review)) == "pos":
            wrongNegList.append(review)
    print("Misclassified Positive Reviews: " + str(wrongPosList))
    print("Misclassified Negative Reviews: " + str(wrongNegList))
            


if __name__ == "__main__":
    classifyReviews()


