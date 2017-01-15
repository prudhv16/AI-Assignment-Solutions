import csv
import numpy as np
from math import log
from copy import deepcopy
import matplotlib as plt

class TreeNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}
        self.split_attr = None
        self.answer = None
        self.attribute_child = {}
        self.attribute_label = None
        self.subsets={}
        self.depth=0

def majorityClass(training_set):
    categories={}
    for row in training_set:
        categories.setdefault(row[0],0)
        categories[row[0]] +=1
    max_class = 0

    for key,val in categories.items():
        if val > max_class:
            max_class = val
            majorityclass = key
    #print majorityclass
    return majorityclass

def sameClassLabel(data):
    count = len(data)
    categories = {}
    for row in data:
        categories.setdefault(row[0],0)
        categories[row[0]] += 1
    #print "length",len(set(d.keys()))
    return (len(set(categories.keys())) <= 1)


def weighted_entropy(training_set, weights):
    total_rows = len(training_set)
    W = sum(weights)
    #for weight in weights:
    #    W += int(weight)

    # converts two list into dictionary
    # eg:training_set =[[1,2],[0,3]],weights =[0.4,0.9]
    # dict_example_weight ={[1,2]:0.4,[0,3]:0.9}
    dict_example_weight = dict(zip(training_set, weights))
    y_val = {}
    ent = 0.0

    for rowIndex,row in enumerate(training_set):
        y_val.setdefault(row[0], 0)
        y_val[row[0]] += weights[rowIndex]

    for val in y_val.values():
        prob = val / float(W)
        ent -= prob * log(prob, 2)
    return ent

def calc_entropy(training_set):
    total_rows = len(training_set)
    #print "total rows in the set",total_rows
    y_val = {}
    ent = 0.0
    for row in training_set:
        y_val.setdefault(row[0],0)
        y_val[row[0]] += 1
    for val in y_val.values():
        prob = val/ float(total_rows)
        ent -= prob * log(prob,2)
    return ent


def plot_accuracy(x_axis,y_axis):
    plt.plot(x_axis, y_axis, '-o', color='#3F5D7D', alpha=0.7)
    plt.ylabel("Accuracy")
    plt.xlabel("Depth")
    plt.show()

def information_gain(attr_mapped_name, training_set, threshold,ensembleType=None):
    if ensembleType == 'boost':
        entropy = weighted_entropy(training_set[:-1],training_set[-1])
    else:
        entropy = calc_entropy(training_set)

    attr_vals = {}
    total_rows = len(training_set)
    entropy_vals = 0.0
    for row in training_set:

        attr_vals.setdefault(row[attr_mapped_name], [])
        attr_vals[row[attr_mapped_name]].append(row)

    for val in attr_vals.keys():
        ent_val = calc_entropy(attr_vals[val])
        p = len(attr_vals[val]) / total_rows
        entropy_vals += p*ent_val
    info_gain = entropy - entropy_vals
    return info_gain

def get_best_attr(training_set, threshold, ensembleType=None):
    info_gain_attr_dict = {}
    numFeatures = len(training_set[0])-1
    for attr in range(1,numFeatures):
        info_gain_attr_dict[attr] = 0

        #calculate information gain based on entropy
        info_gain_attr_dict[attr] = information_gain(attr, training_set, threshold, ensembleType)
    #print(info_gain_attr_dict)
    attr_max_gain = max(info_gain_attr_dict, key=info_gain_attr_dict.get)
    if info_gain_attr_dict[attr_max_gain] < threshold:
        return -1
    else:
        return attr_max_gain

def split(training_set, attribute):
    subsets = {}
    for row in training_set:
        subsets.setdefault(row[attribute] , [])
        subsets[row[attribute]].append(row)
    return subsets

def print_confusion_matrix(tp,tn,fp,fn,cl):
    fig = plt.figure(figsize=(8, 8))
    cm = fig.add_subplot(111)

    # Draw the grid boxes
    cm.set_xlim(-0.5, 2.5)
    cm.set_ylim(2.5, -0.5)
    cm.plot([-0.5, 2.5], [0.5, 0.5], '-k', lw=1)
    cm.plot([-0.5, 2.5], [1.5, 1.5], '-k', lw=1)
    cm.plot([0.5, 0.5], [-0.5, 2.5], '-k', lw=1)
    cm.plot([1.5, 1.5], [-0.5, 2.5], '-k', lw=1)

    # Set xlabels
    cm.set_xlabel('Predicted Label', fontsize=16)
    cm.set_xticks([0, 1, 2])
    cm.set_xticklabels(cl + [''])
    cm.xaxis.set_label_position('top')
    cm.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    cm.xaxis.set_label_coords(0.34, 1.06)

    # Set ylabels
    cm.set_ylabel('True Label', fontsize=16, rotation=90)
    cm.set_yticklabels(cl + [''], rotation=90)
    cm.set_yticks([0, 1, 2])
    cm.yaxis.set_label_coords(-0.09, 0.65)

    # Fill in initial metrics: tp, tn, etc...
    cm.text(0, 0,
            'True Neg: %d' % tn,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    cm.text(0, 1,
            'False Neg: %d' % fn,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    cm.text(1, 0,
            'False Pos: %d' % fp,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    cm.text(1, 1,
            'True Pos: %d' % tp,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    cm.text(2, 0,
            'False Pos Rate: %.2f' % ((fp+1.0) / (fp + tn + 1.0)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    cm.text(2, 1,
            'True Pos Rate: %.2f' % ((tp+1.0) / (tp + fn + 1.0)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    cm.text(2, 2,
            'Accuracy: %.2f' % ((tp + tn + 0.) / (tp+tn+fp+fn)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    cm.text(0, 2,
            'Neg Pre Val: %.2f' % (1 - fn / (fn + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    cm.text(1, 2,
            'Pos Pred Val: %.2f' % (tp / (tp + fp + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    plt.tight_layout()
    plt.show()

def buildTree(training_set, currentNode, depth, ensembleType = None):
    depth_check=0
    threshold = 0.00001
    if ensembleType == 'boost':
        default_val = majorityClass(training_set[:-1])
    else:
        default_val = majorityClass(training_set)
    if depth > currentNode.depth :
        #base case when all data has the same label
        if sameClassLabel(training_set):
            #print("same class data")
            currentNode.answer = training_set[0][0]
            return currentNode

        best_attr = get_best_attr(training_set, threshold, ensembleType)

        if best_attr == -1:
            currentNode.answer = default_val
            return currentNode
        else:
            currentNode.attribute_label = best_attr

            #print("attribute label",currentNode.attribute_label)

            currentNode.data = None
            if ensembleType == 'boost':
                currentNode.subsets = split(training_set[:-1], best_attr)
            else:
                currentNode.subsets= split(training_set,best_attr)

            for key in currentNode.subsets.keys():
                child = TreeNode(currentNode)
                child.depth = currentNode.depth+1
                currentNode.children[key] = child
                #print("key is ",key)

                buildTree(currentNode.subsets[key], currentNode.children[key],depth)

            return currentNode
    else:
        currentNode.answer = majorityClass(training_set)
        return currentNode

def classify(test_row, tree):

    curr_tree = tree

    while curr_tree.answer == None:
        if curr_tree.attribute_label >= len(test_row) or test_row[curr_tree.attribute_label-1] not in curr_tree.children:
            return -1

        curr_tree = curr_tree.children[test_row[curr_tree.attribute_label-1]]

    return curr_tree.answer
