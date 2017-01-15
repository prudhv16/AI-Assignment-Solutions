'''
Assignment 5
Author :- Prudhvi Indana, Siddu pagadala, Rohit Dandona.
'''
import sys
import os
from numpy import random
'''
Reused confusion matrix code from previous assignment.

Assumptions
1) All the train and test points have a length of 192 fixed. got this from current courpes
See commented code to find unique length of different points.


'''


def euclidiendistance(point1,point2):
    '''
    Calculates euclidian distance between given two points
    :param point1: image 1 passed as a point
    :param point2: image 2 passed as a point
    :return: eculidian distance between point 1 and point 2
    '''
    return sum([abs(point1[x]-point2[x])**2 for x in range(192)])


def predict1(train_data,trainorientation,test_data):
    '''

    :param train_data:
    :param trainorientation:
    :param test_data:
    :return:
    '''
    predicted = []
    for testpoint in test_data:
        mindist = float("inf");currentprediction = None
        for i in range(len(train_data)):
            dist = euclidiendistance(train_data[i],testpoint)
            if dist < mindist:
                currentprediction = trainorientation[i]
        predicted.append(currentprediction)
    return predicted

def count_target(neighbors):
    '''
    method that gets the max frequent neighbors from a list of lists having distance and label
    :param neighbors: list of k nearest neighbors lists having distance and label
    :return: predicted label
    '''
    counter = {}
    for dist,prediction in neighbors:
        if prediction in counter:
            counter[prediction] += 1
        else:
            counter[prediction] = 0
    return max(counter,key=lambda x:counter[x])

def predict2(train_data,trainorientation,test_data,k):
    '''
    predict the label for train data by running knn
    :param train_data: all the training images passed as a list
    :param trainorientation: actual orientation of the images as a list
    :param test_data: data on which we need to predict the labels
    :param k: orbitary parameter currently set as 200 (hardcoded)
    :return: predictions for all the images in test set.
    '''
    predicted = []
    for testpoint in test_data:
        neighbors = [];
        for i in range(len(train_data)):
            dist = euclidiendistance(train_data[i],testpoint)
            neighbors.append([dist,trainorientation[i]])
        neighbors = sorted(neighbors,key=lambda x:x[0])[:200]
        #print len(neighbors)
        #print neighbors
        #break
        currentprediction = count_target(neighbors)
        predicted.append(currentprediction)
    return predicted

'''
def normalize(data):
    tra
    for item in data
'''

def getaccuracy(actual, predictions):
    '''
    This function generated accuracy based on actual topics and predicted topics
    :param actual: list of actual orientations
    :param predictions: list of predicted orientations
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


def createtrain(trainfilepath,prop):
    '''
    creates data as list of lists by reading data of this assignment
    :param trainfilepath: path ot which data set is present
    :return: a list of lists having train data
    '''
    cwd = os.getcwd()
    train_data = [];orientation = [];images = []
    with open(os.path.join(cwd,trainfilepath),'r') as f:
        for line in f:
            temptoss = random.choice([0, 1], 1, p=[1 - prop, prop])
            if temptoss == 1:
                temp = line.strip().split()
                image,temporientaion,templine = temp[0],temp[1],map(int,temp[2:])
                #print templine,sum(templine)
                #tempmean = sum(templine)/len(templine)
                #templine = [float(x)/tempmean for x in templine]
                train_data.append(templine);orientation.append(int(temporientaion))
                images.append(image)
                #print train_data,orientation,images
        f.close()
        return images,train_data,orientation

def knn_classifier(trainfilename, testfilename, algo):
    if algo == "nearest":
        trainimages, train_data, trainorientation = createtrain(trainfilename,1)
        testimages, test_data, actualorientation = createtrain(testfilename,1)
        #train_data = normalize(train_data)
        #train_data = [[float(x)/sum(elem) for x in elem] for elem in train_data]
        #test_data = [[float(x)/sum(elem) for x in elem] for elem in test_data]
        predictedorientation = predict2(train_data,trainorientation,test_data,200)
        #print train_data,orientation
        #print len(train_data),len(trainorientation)
        uniquelabels = [0,90,180,270]
        line = "{0} {1}\n"
        file = open('nearest_output.txt','w')
        for i in range(len(testimages)):
            file.write(line.format(testimages[i],predictedorientation[i]))
        file.close()
        acc = getaccuracy(actualorientation, predictedorientation)
        print "accuracy is ", (str(acc))
        confusionmat = confusion_matrix(actualorientation,predictedorientation,uniquelabels)
        #print "\n".join([" ".join(map(str, x)) for x in confusionmat])
        print "confusion matrix \n"
        print "---" * 10 + "Predicted" + "---" * 30 + "\n"
        for item in confusionmat:
            x = map(str, item)
            for j in range(len(x)):
                print x[j], " " * (5 - len(x[j])),
            print
        '''
        length1 = [];
        length2 = []
        for i in train_data:
            length1.append(len(i))
        for j in test_data:
            length2.append(len(j))
        print set(length1), set(length2)
        print set(actualorientation),set(trainorientation)
        '''

    else:
        pass
if __name__ == "__main__":
    '''
    Main program for learning , testing and generating confusion matrix.
    '''
    labels = [0,90,180,270]
    trainfilename, testfilename, algo = sys.argv[1], sys.argv[2], sys.argv[3]

    knn_classifier(trainfilename, testfilename, algo)







