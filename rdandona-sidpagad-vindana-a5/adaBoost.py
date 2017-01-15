from __future__ import division
import sys
import numpy as np
import itertools
import operator
import random
import math
import os
import time

#class for storing Nodes of Decision Stump
class TreeNode:
    def __init__(self,col1=None,col2=None,left_label=None,right_label=None):
        self.col1 = col1
        self.col2 = col2
        self.left_label = left_label
        self.right_label = right_label
        self.error_rate = None

#returns left branch and right branch conditions
#returns error rate and finds best pair of attributes
# from 100 randomly chosen pairs from 192 total attributes
def get_error_rate_pair(col_1,col_2,data,classifier_class,weights):
    #left branch true and right branch false
    #col_1 > col_2
    W = sum(weights)

    num_data_points = data.shape[0]
    cond_satisfy_ind_true = np.where(data[:, col_1].astype(int) > data[:, col_2].astype(int))[0]

    cond_satisfy_ind_false = np.where(data[:, col_1].astype(int) <= data[:, col_2].astype(int))[0]

    fp = sum([weights[i] for i in cond_satisfy_ind_true[np.where(data[cond_satisfy_ind_true, 1] != classifier_class)[0]]])

    fn =  sum([weights[i] for i  in cond_satisfy_ind_false[np.where(data[cond_satisfy_ind_false, 1] == classifier_class)[0]]])

    err_rate = (fp + fn)

    # left branch false and right branch true
    # col_1 > col_2
    err_rate_1 = W - err_rate

    if err_rate <= err_rate_1:

        return True,False,err_rate
    else:
        return False, True, err_rate_1



#gets best pair of attributes
def get_best_pair(pairs_list,data,classifier_class,weights):
    pairs_list_err_rates = [get_error_rate_pair(col_1,col_2,data,classifier_class,weights) for (col_1,col_2) in pairs_list]
    index, value = min(enumerate([e for (_,_,e) in pairs_list_err_rates]), key=operator.itemgetter(1))
    return pairs_list[index],pairs_list_err_rates[index]

#learns a decision stump for the given class and weights
def learn_tree(data,classifier_class,weights):
    tree_learn = TreeNode()
    random_pairs = random.sample(list(itertools.combinations(range(2, 194), 2)),100)
    (tree_learn.col1,tree_learn.col2),(left_branch_cond,right_branch_cond,tree_learn.error_rate) = get_best_pair(random_pairs, data, classifier_class,weights)
    if left_branch_cond == True:
        tree_learn.left_label = classifier_class
    else:
        tree_learn.right_label = classifier_class
    return tree_learn

#classifies all the test data into one of the four categories(rotation) and returns list of predicted labels
def classify(learned_tree,test_data):

    pred_labels = np.empty(test_data.shape[0], dtype='int')
    pred_labels.fill(-1)
    if learned_tree.left_label != None:
        ind = np.where(test_data[:, learned_tree.col1].astype(int) > test_data[:, learned_tree.col2].astype(int))[0]
        pred_labels[ind] = int(learned_tree.left_label)

    else:
        ind = np.where(test_data[:, learned_tree.col1].astype(int) <= test_data[:, learned_tree.col2].astype(int))[0]
        pred_labels[ind] = int(learned_tree.right_label)
    return pred_labels

#calculates the accuracy and returns
def cal_accuracy(test_data,tree_learn,entype,classifier_class=None):
    predicted_boolean = []
    predicted_result = []
    predicted = []
    FP = 0
    TP = 0
    FN = 0
    TN = 0

    if entype == 'boost_train':
        temp = np.array([int(val) if val == classifier_class else -1 for val in list(test_data[:, 1])])

        result_maxVote = classify(tree_learn,test_data)

        predicted_boolean = result_maxVote == temp
        predicted_boolean_list = predicted_boolean.tolist()
        accuracy = float(predicted_boolean_list.count(True)) / float(len(predicted_boolean_list))


        return predicted_boolean,accuracy
    elif entype == 'boost_test':
        temp = np.array([int(val) if val == classifier_class else -1 for val in list(test_data[:, 1])])
        result_ensemble = np.array([ [1 if data_point==int(test_data[ind,1]) else 0 for ind,data_point in enumerate(classify(tree,test_data).tolist())] for tree in tree_learn])
        result_ensemble = np.sum(result_ensemble,axis=0)/len(tree_learn)

        return result_ensemble
        result_maxVote = []
    elif entype == 'boost_final_test':
        result_maxVote = tree_learn

        predicted_boolean = result_maxVote == test_data[:,1].astype(int)
        predicted_boolean = predicted_boolean.tolist()
        accuracy = float(predicted_boolean.count(True)) / float(len(predicted_boolean))
        print "Accuracy:",accuracy
        return predicted_boolean


    return predicted_boolean


#Learns ensemble of decision stumps for a given class and number of stumps
def learn_boosted(num_trees, data, classifier_class):

    training_data_size = data.shape[0]

    # Intial weight distribution for training examples(uniform distribution)
    d_t = [1 / training_data_size] * training_data_size

    tree_learn_boost = {}
    for _ in range(num_trees):

        tree_learn = None
        tree_learn = learn_tree(data,classifier_class,d_t)

        predicted_result_t,accuracy = cal_accuracy(data, tree_learn, 'boost_train',classifier_class)

        error_rate_t = 1-accuracy
        weight_factor = (1 - error_rate_t+1) / (error_rate_t+1)
        alpha_t = 0.5 * math.log(weight_factor)
        d_t1 = [d_t[ind]*np.exp(-alpha_t) if p == True else d_t[ind] *np.exp(alpha_t)  for ind,p in enumerate(predicted_result_t.tolist())]
        normalizingFactor = sum(d_t1)
        d_t =  [x/normalizingFactor for x in d_t1]
        tree_learn_boost[tree_learn] = alpha_t

    return tree_learn_boost

#Trains all the ensenbles for each class
def train_all_classifiers(num_trees, train_data):
    ensemble = {}
    ensemble_0 = learn_boosted(num_trees,train_data,'0')
    ensemble_90 = learn_boosted(num_trees,train_data,'90')
    ensemble_180 = learn_boosted(num_trees,train_data,'180')
    ensemble_270 = learn_boosted(num_trees,train_data,'270')
    ensemble['0'] = ensemble_0
    ensemble['90'] = ensemble_90
    ensemble['180'] = ensemble_180
    ensemble['270'] = ensemble_270

    return ensemble


def classify_ensemble(test_data,ensemble_train):

    result_ensemble_0 = cal_accuracy(test_data, ensemble_train['0'], 'boost_test','0')
    result_ensemble_90 = cal_accuracy(test_data, ensemble_train['90'], 'boost_test', '90')
    result_ensemble_180 = cal_accuracy(test_data, ensemble_train['180'], 'boost_test', '180')
    result_ensemble_270 = cal_accuracy(test_data, ensemble_train['270'], 'boost_test', '270')

    result_predicted = np.array([result_ensemble_0,result_ensemble_90,result_ensemble_180,result_ensemble_270])

    result_predicted = result_predicted.argmax(axis=0)
    result_predicted[ result_predicted == 1] = 90
    result_predicted[result_predicted == 2] = 180
    result_predicted[result_predicted == 3] = 180

    return result_predicted


def write_to_output(test_data, result_predicted,test_file):
    dataset = np.loadtxt(test_file, dtype=str)
    image_names = dataset[:, 0:1]
    file_name = "output.txt"
    if os.path.isfile(file_name):
        open(file_name, 'w').close()
    file = open(file_name, "a")
    for index in range(len(image_names)):
        file.write(test_data[index,0]+" "+str(int(result_predicted[index]))+"\n")
    file.close()


def confusion_matrix(actual, predictions, uniquetopics):
    '''
    generated confusion matrix based on actual and predicted list of topics
    :param actual: list of actual topic documents belongs to
    :param predictions: list of predicted documents belong to
    :param uniquetopics: list of unique topics present in test path
    :return: a two dimensional array having confusion matrix values
    '''
    length = len(uniquetopics)
    confusion = [[0] * length for x in range(length)]
    length = len(actual)
    i = 0
    for i in range(length):
        index1 = uniquetopics.index(actual[i])
        index2 = uniquetopics.index(predictions[i])
        confusion[index1][index2] += 1
    return confusion


## Selects fraction of training dataset randomly
def get_test_subset(dataset, percent):
    fraction = float(percent)/float(100)
    dataset_new = dataset[(np.random.choice(len(dataset), (int(len(dataset) * fraction)), replace=False))]
    return dataset_new


def adaboost_classifier(train_file,test_file,classifier_type,stump_count):

    start_time = time.time()
    (train_file,test_file,classifier_type,stump_count) = sys.argv[1:]

    train_data = np.genfromtxt(train_file, dtype='|S50')
    test_data = np.genfromtxt(test_file, dtype='|S50')

    train_data = get_test_subset(train_data,100)

    ensemble = train_all_classifiers(int(stump_count),train_data)
    result_predicted = classify_ensemble(test_data,ensemble)

    cal_accuracy(test_data,result_predicted,'boost_final_test')
    write_to_output(test_data,result_predicted,test_file)

    uniquetopics = [0,90,180,270]
    uniquetopics_str = ['0','90','180','270']
    confusion = confusion_matrix(test_data[:,1].astype(int).tolist(), result_predicted, uniquetopics)

    print "confusion matrix \n"
    print "Predicted".center(30,'*')
    print '      '.join(uniquetopics_str)
    print ''.join(['*']*30)
    for item in confusion:
        x = map(str, item)
        for j in range(len(x)):
            print x[j], " " * (5 - len(x[j])),
        print

    print "Execution time:",time.time()-start_time



if __name__ == "__main__":
    input_args = sys.argv[1:]
    train_file = input_args[0]
    test_file = input_args[1]
    classifier_type = input_args[2]
    stump_count = input_args[3]
    adaboost_classifier(train_file, test_file, classifier_type, stump_count)
