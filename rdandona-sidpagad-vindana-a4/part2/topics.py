'''
Author :- Prudhvi Indana, Siddu Pagadala, Rohit Dandona
AI Assignment - 4
Topic classification
'''

"""
Accuracy for
fraction   1 :- 72.0525756771
         0.8 :- 58.1386086033
         0.6 :- 47.0525756771
         0.4 :- 18.1093998938
         0.2 :- 22.2517259692
         0.1 :- 15.0026553372
         0   :- 6.14710568242
"""


import os
import math
import sys
import numpy as np
import cPickle as pickle
import random
from collections import Counter

# Evaluataion

#this list of stops words is prepared by looking at top 20 words generated in each topic.
stopwords = set(['a','and','from:','for','that','i','of','is','it','|','to', 'have','in','the','on','with','you','subject:','>','are','as','they','if','at','be','but','your','what','not','about','can','from','would','there','had','do','we','an','any','will','has','my','or','re:','all','>>','his','who','--',
                 '/*','|>','>>','me','like','just','this',"don't",'than','when','so','one','was','by','out','lines:','use','get','how','were','more','no','some','go','he','only','may','us','up','been','am','which','should','also','does','know','then','them','these','why','their','other','used','must','those',
                 'could','many','because','very','could','these','its','new','*','-',':',"it's",'.',''])

def getaccuracy(actual, predictions):
    '''
    This function generated accuracy based on actual topics and predicted topics
    :param actual: list of actual topic documents belongs to
    :param predictions: list of predicted documents belong to
    :return: accuracy value as float
    '''
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            correct += 1
    return (correct/float(len(actual))) * 100.0

def confusion_matrix(actual,predictions,uniquetopics):
    '''
    generated confusion matrix based on actual and predicted list of topics
    :param actual: list of actual topic documents belongs to
    :param predictions: list of predicted documents belong to
    :param uniquetopics: list of unique topics present in test path
    :return: a two dimensional array having confusion matrix values
    '''
    length = len(uniquetopics)
    confusion = [[0]*length for x in range(length)]
    length = len(actual)
    i = 0
    for i in range(length):
        index1 = uniquetopics.index(actual[i])
        index2 = uniquetopics.index(predictions[i])
        confusion[index1][index2] += 1
    return confusion

#Training
def get_word_dict(train_directory_path,category):
    '''
    creates a counter of words occured in the file as dictionary with out including stop words.
    :param train_directory_path: location at which current file can be read
    :param category: Not much use at this moment - might help if planning to use it in future.
    :return: counter dictionary, count of all the words present in the file.
    '''
    tempdict = {}
    filedir = os.listdir(os.path.join(train_directory_path))
    number_docs = len(filedir)
    for fileName in filedir:
        with open(os.path.join(train_directory_path,fileName),'r') as f:
            for line in f:
                for word in line.split():
                    lower_case_word = word.lower().strip()
                    if lower_case_word not in stopwords:
                        if lower_case_word in tempdict:
                            tempdict[lower_case_word] += 1
                        else:
                            tempdict[lower_case_word] = 1
            f.close()
    return tempdict, number_docs



def train_model(tempchoose):
    '''
    main method that creates python objects needed for future testing
    This method creates prior probability dictionary of all topics, a list of list object having topic,dictionary of word frequency present in each topic, total number of words present in all documents
    :param tempchoose: fraction of testpoints that need to be considered
    :return: creates a file having python objects needed for predicting topic.
    '''
    train_directory_path = datasetDirectory
    temptopics = [x for x in os.listdir(train_directory_path) if os.path.isdir(os.path.join(train_directory_path,x)) and x[0] != "."]
    topics = []
    testtopics = []
    for topic in os.listdir(train_directory_path):
        path = os.path.join(train_directory_path,topic)
        if os.path.isdir(path) and topic[0] != ".":
            if tempchoose == 0.0:
                tempdict, number_docs = get_word_dict(path, topics)
                topics.append([random.choice(temptopics), tempdict, number_docs])
            else:
                temp = np.random.choice([0, 1], 1, p=[1 - fraction, fraction])
                if temp == 1:
                    tempdict,number_docs = get_word_dict(path,topics)
                    topics.append([topic,tempdict,number_docs])
                else:
                    tempdict, number_docs = get_word_dict(path, topics)
                    testtopics.append([tempdict,number_docs])


    #topics_probs = {}
    if fraction != 1 or fraction != 0:
        predicted = []
        for test_file_words,nowords in testtopics:
            predict = []
            for temptopic, topicdict, noFiles in topics:
                p_topic = 0
                for key in test_file_words:
                    if key in topicdict:
                        p_topic = p_topic + (math.log(topicdict[key]) * test_file_words[key])
                predict.append([temptopic, p_topic])
            maxkey = predict.index(max(predict, key=lambda x: x[1]))
            predicted.append(predict[maxkey][0])

            for i in range(len(predict)):
                for test_file_words,nowords in testtopics:
                    for j in range(len(topics)):
                        temptopic = topics[j][0]
                        if temptopic == predict[i]:
                            topicdict = topics[j][1]
                            topics[j][2] += nowords
                            for key in test_file_words:
                                if key in topicdict:
                                    topicdict[key] += test_file_words[key]
                                else:
                                    topicdict[key] = 1
                            topics[j][1] = topicdict


    totalwords = 0
    priorprop = {}
    topics_key_prob = {}
    for topic,topicdict,nowords in topics:
        temp = sum(topicdict.values())
        totalwords += nowords
        topic_prob = {}
        for key in topicdict:
            topic_prob[key] = float(topicdict[key]+1)/float(temp+20)
        topics_key_prob[topic] = topic_prob


    for topic, topicdict, noFiles in topics:
        priorprop[topic] = float(noFiles)/float(totalwords)
    topwords = []
    for topic, topicdict,nowords in topics:
        temptopwords = [topic]
        temptopwords.append(dict(Counter(topicdict).most_common(10)).keys())
        topwords.append(temptopwords)

    fileformat = "{item0} :- {item1} \n"
    fileout = open("distinctive_words.txt","w")

    for line in topwords:
        contant0 = line[0]
        contant1 = " ".join(line[1])
        fileout.write(fileformat.format(item0 = contant0,item1=contant1))
        #print fileformat.format(item0 = contant0,item1=contant1)

    pickle_out = open(modelFile,"wb")
    pickle.dump((topics_key_prob,priorprop,topics),pickle_out)
    pickle_out.close()


# Testing
def test_model():
    '''
    reads python objects from previously pickled file and predicts topics of all the test documents
    :return: prints out confusion matrix based in unique topics in test documents
    '''
    test_directory_path = datasetDirectory
    actual = []
    predicted = []
    for topic in os.listdir(test_directory_path):
        for fileName in os.listdir(os.path.join(test_directory_path,topic)):

            test_file_words = {}
            with open(os.path.join(test_directory_path,topic,fileName),'r') as f:

                for line in f:
                    for word in line.split():
                        lower_case_word = word.lower().strip()
                        if lower_case_word not in test_file_words:
                            test_file_words[lower_case_word] = 1
                        else:
                            test_file_words[lower_case_word] = test_file_words[lower_case_word] + 1

            predict = []
            for temptopic, topicdict, noFiles in topics:
                p_topic = 0
                for key in test_file_words:
                    if key in topicdict:
                        p_topic = p_topic + (math.log(topicdict[key]) * test_file_words[key])
                predict.append([temptopic,p_topic])
            actual.append(topic)
            maxkey = predict.index(max(predict,key= lambda x:x[1]))
            predicted.append(predict[maxkey][0])

    acc = getaccuracy(actual, predicted)
    print "accuracy is ",(str(acc))
    uniquetopics = list(set(actual+predicted))
    confusion = confusion_matrix(actual,predicted,uniquetopics)
    #print "\n".join([" ".join(map(str,x)) for x in confusion])

    print "confusion matrix \n"
    print "---"*10 + "Predicted" + "---" * 30+"\n"
    for item in confusion:
        x = map(str,item)
        for j in range(len(x)):
            print x[j]," "*(5-len(x[j])),
        print

if __name__ == "__main__":
    '''
    python topics.py mode dataset-directory model-file [fraction]
    where mode is either test or train, dataset-directory is a directory containing directories for each of the
    topics (you can assume there are exactly 20), and model-file is the filename of your trained model. In training
    mode, an additional parameter fraction should be a number between 0.0 and 1.0 indicating the fraction of
    labeled training examples that your training algorithm is allowed to see.

    In training mode, dataset-directory will be the training dataset and the program should write the trained
    model to model-file.
    '''
    mode,datasetDirectory,modelFile,fraction = sys.argv[1],sys.argv[2],sys.argv[3],float(sys.argv[4])
    if mode == "train":
        train_model(fraction)

    else:
        with open(modelFile,"rb") as fh:
            tup = pickle.load(fh)
        topics_key_prob = tup[0]
        priorprop = tup[1]
        topics = tup[2]

        test_model()
