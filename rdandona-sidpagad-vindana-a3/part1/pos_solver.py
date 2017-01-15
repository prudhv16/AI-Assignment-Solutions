###################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids:
# vindana (Venkata Prudhvi Raj Indana)
# sidpagad (Siddhartha Pagadala)
# rdandona (Rohit Dandona)
#
# (Based on skeleton code by D. Crandall)
#
#
####
'''
Problem 1

parts of speech tagging by using below techniques

Initial probability is calculated by counting the number of sentences in training set which start by a particular part of speech.

Emission probability :- for each word in training set we calculated the ratio of count of word belonging to that particular POS/count of total words in that particular POS. This is done for each part of speech.

Transition probability:- for POS in training set we calculated the ratio of each POS transition to a particular POS by the total count of such transitions.

We applied laplace smoothing to all the above matrices to avoid following scenarios

	- to avoid zeros in transition and emission probabilities.
	- if a word in test data point(sentence) is not observed in training data.

1)simplified model

Best POS was returned for each words in test sentence using emission probabilities i.e. best POS for given word is chosen based on its probability value

2)An HMM

since current word POS depends on previous word POS, viterbi algorithm is being used to calculated the best possible state(POS) sequence in a sentence.

3)Complex HMM

since current word POS depends on previous two words POS, a modified version of viterbi algorithm is being used to calculated the best possible state(POS). sequence in a sentence.

here an additional two step transition probabilities is calculated by considering POS of last two words(two grams).

While calculating viterbi values we are considering the previous two state values instead of just previous state as in viterbi.

Design decisions:
1) Laplace smoothing is used over emission probabilities, transition probabilities and second order transition probabilities
2) In the complex algo negative logs of the viterbi probabilities are taken

'''
####

import random
import math
import operator
import copy
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    def __init__(self):
        self.part_of_speech = {}
        self.word_dict = {}
        self.sentence_dict = {}
        self.transition_dict = {}
        self.transition_dict_second_order = {}
        self.total_number_of_sentences = 0
        self.total_words = 0
        self.initial_probabilities = {}
        self.transition_probabilities = {}
        self.transition_probabilities_second_order = {}
        self.emission_probabilities = {}


    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        prob = 0
        for index in range(len(sentence)):
            if index == 0:
                prob = prob + math.log(self.emission_probabilities[sentence[index] + "_" + label[index]])
            else:
                if self.transition_probabilities[label[index] + "|" + label[index - 1]] == 0:
                    prob = prob + math.log(self.emission_probabilities[sentence[index] + "_" + label[index]]) + 0
                else:
                    prob = prob + math.log(self.emission_probabilities[sentence[index] + "_" + label[index]]) + math.log(self.transition_probabilities[label[index] + "|" + label[index - 1]])
        prob = prob + math.log(self.initial_probabilities[label[0]])
        return prob

    # The train function determines the emission probabilities, transition probabilities,
    # second order transition probabilities and initial probabilities.
    #
    # Various count are determined to calculate these probabilities.
    #
    # This function does not return anything.
    def train(self, data):
        self.total_number_of_sentences = len(data)
        for sentence in data:
            if sentence[1][0] in self.sentence_dict:
                self.sentence_dict[sentence[1][0]] = self.sentence_dict[sentence[1][0]] + 1
            else:
                self.sentence_dict[sentence[1][0]] = 1

            for index in range(len(sentence[0])):
                self.total_words = self.total_words + 1
                if (index + 1) < len(sentence[0]):
                    tran_key = sentence[1][index] + '_' + sentence[1][index + 1]
                    if tran_key in self.transition_dict:
                        self.transition_dict[tran_key] = self.transition_dict[tran_key] + 1
                    else:
                        self.transition_dict[tran_key] = 1

                word_key = sentence[0][index] + '_' + sentence[1][index]
                if word_key in self.word_dict:
                    self.word_dict[word_key] = self.word_dict[word_key] + 1
                else:
                    self.word_dict[word_key] = 1

                if sentence[1][index] in self.part_of_speech:
                    self.part_of_speech[sentence[1][index]] = self.part_of_speech[sentence[1][index]] + 1
                else:
                    self.part_of_speech[sentence[1][index]] = 1

            for index in range(len(sentence[0])):
                if (index + 2) < len(sentence[0]):
                    tran_key = sentence[1][index] + '_' + sentence[1][index + 1]+ '_' + sentence[1][index + 2]
                    if tran_key in self.transition_dict_second_order:
                        self.transition_dict_second_order[tran_key] = self.transition_dict_second_order[tran_key] + 1
                    else:
                        self.transition_dict_second_order[tran_key] = 1


        for key, value in self.sentence_dict.iteritems():
            self.initial_probabilities[key] = float(value) / float(self.total_number_of_sentences)

        for key1 in list(self.sentence_dict.keys()):
            for key2 in list(self.sentence_dict.keys()):
                tran_key = key1 + "_" + key2
                if tran_key in self.transition_dict:
                    self.transition_probabilities[key2 + "|" + key1] = float(self.transition_dict[tran_key]) / float(
                        self.part_of_speech[key1])
                else:
                    self.transition_probabilities[key2 + "|" + key1] = 0

        min_tranprobkey = [[self.transition_probabilities[x], x] for x in self.transition_probabilities if self.transition_probabilities[x] != 0]
        min_tranprob = min(min_tranprobkey, key=lambda x: x[0])[0] / 10.0
        for key in self.transition_probabilities.keys():
            if self.transition_probabilities[key] == 0:
                self.transition_probabilities[key] = min_tranprob


        for key1 in list(self.sentence_dict.keys()):
            for key2 in list(self.sentence_dict.keys()):
                for key3 in list(self.sentence_dict.keys()):

                    tran_key = key1 + "_" + key2+ "_" + key3
                    if tran_key in self.transition_dict_second_order:
                        self.transition_probabilities_second_order[key3 + "|" + key1 +"_"+key2] = float(self.transition_dict_second_order[tran_key]) / float(self.transition_dict[key1+"_"+key2])
                    else:
                        self.transition_probabilities_second_order[key3 + "|" + key1 +"_"+key2] = 0

        min_tranprobkey = [[self.transition_probabilities_second_order[x], x] for x in self.transition_probabilities_second_order if self.transition_probabilities_second_order[x] != 0]
        min_tranprob = min(min_tranprobkey, key=lambda x: x[0])[0] / 10.0
        for key in self.transition_probabilities_second_order.keys():
            if self.transition_probabilities_second_order[key] == 0:
                self.transition_probabilities_second_order[key] = min_tranprob

        for key in list(self.word_dict.keys()):
            parts = key.split("_")
            self.emission_probabilities[key] = float(self.word_dict[key] + 1) / float(self.part_of_speech[parts[1]] + self.total_words)
        pass

    ### Functions for each algorithm ###

    # The simplified function returns a list of the predicted POS is returned
    # for each word of the input sentence using emission probabilities
    #
    # It also returns a list of the predicted probabilities of each word
    def simplified(self, sentence):
        result = []
        result_prob = []
        for word in sentence:
            test_word_prob = {}
            for part_speech in list(self.sentence_dict.keys()):
                k = word.lower() + "_" + part_speech
                if k in self.word_dict:
                    prob = self.emission_probabilities[k] * self.initial_probabilities[part_speech]
                    test_word_prob[part_speech + "|" + word] = prob
                else:
                    test_word_prob[part_speech + "|" + word] = 0

            result_key = max(test_word_prob.iteritems(), key=operator.itemgetter(1))[0]
            result.append(result_key.split("|")[0])
            result_prob.append(test_word_prob[result_key])
        return [[result], [result_prob]]

    # The hmm function returns a list of the predicted POS is returned
    # for each word of the input sentence using the viterbi algorithm.
    #
    # For each word of the input sentence, emission probabilities and transmission
    # probabilities are used.
    #
    # Laplace smoothing is applied to the emission probabilities
    def hmm(self, words):
        result = []
        viterbi_dict = {}
        viterbi_dict_back = {}
        predicted_sentence = ''
        count = 1
        for index in range(len(words)):
            if index == 0:
                for key in list(self.sentence_dict.keys()):
                    emission_key = words[index]
                    if (emission_key + "_" + key) not in self.emission_probabilities:
                        self.emission_probabilities[emission_key + "_" + key] = float(1) / float(
                            self.part_of_speech[key] + self.total_words)

                    prob = self.initial_probabilities[key] * self.emission_probabilities[emission_key + "_" + key]

                    if count in viterbi_dict:
                        viterbi_dict[count]['v' + "_" + str(count) + "_" + key] = prob
                    else:
                        viterbi_dict[count] = {'v' + "_" + str(count) + "_" + key: prob}
                count = count + 1

            else:
                for key1 in list(self.sentence_dict.keys()):
                    temp_dict2 = {}
                    emission_key = words[index]
                    if (emission_key + "_" + key1) not in self.emission_probabilities:
                        self.emission_probabilities[emission_key + "_" + key1] = float(1) / float(self.part_of_speech[key1] + self.total_words)

                    for key2 in list(self.sentence_dict.keys()):
                        prob = viterbi_dict[count - 1]['v' + "_" + str(count - 1) + "_" + key2] * self.transition_probabilities[key1 + '|' + key2]
                        temp_dict2 = self.update_dictionary(temp_dict2, key2, prob)


                    if count in viterbi_dict:
                        viterbi_dict[count]['v' + "_" + str(count) + "_" + key1] = temp_dict2[temp_dict2.keys()[0]] * self.emission_probabilities[emission_key + "_" + key1]

                    else:
                        viterbi_dict[count] = {'v' + "_" + str(count) + "_" + key1: temp_dict2[temp_dict2.keys()[0]] * self.emission_probabilities[emission_key + "_" + key1]}
                    viterbi_dict_back['v' + "_" + str(count) + "_" + key1] = temp_dict2.keys()[0]

                count = count + 1
        count = count - 1
        predicted_key = max(viterbi_dict[count].iteritems(), key=operator.itemgetter(1))[0]
        predicted_sentence = predicted_sentence + predicted_key.split("_")[2] + " "
        for index in reversed(range(len(words) - 1)):
            count = count - 1
            predicted_key_new = 'v' + "_" + str(count) + "_" + viterbi_dict_back[predicted_key]
            predicted_sentence = predicted_sentence + viterbi_dict_back[predicted_key] + " "
            predicted_key = copy.deepcopy(predicted_key_new)
        for c in reversed(predicted_sentence.split()):
            result.append(c)

        return [ [result], [] ]

    # The update_dictionary function accepts a dictionary and returns it
    # by updating its key with a greater (if found) value.
    #
    # This function is used in the HMM algorithm
    def update_dictionary(self, dict, key, updated_value):

        if not dict:
            dict[key] = updated_value
        else:
            if (dict.itervalues().next()) < (updated_value):
                dict = {}
                dict[key] = updated_value
        return dict

    # The update_dictionary_complex function accepts a dictionary and returns it
    # by updating its key with a smaller (if found) value.
    #
    # This function is used in the complex algorithm
    def update_dictionary_complex(self, dict, key, updated_value):

        if not dict:
            dict[key] = updated_value
        else:
            if (dict.itervalues().next()) > (updated_value):
                dict = {}
                dict[key] = updated_value
        return dict

    # The complex function returns a list of the predicted POS is returned
    # for each word of the input sentence using the viterbi algorithm over HMM of second order.
    #
    # For each word of the input sentence, emission probabilities and the second order transmission
    # probabilities are used.
    #
    # Laplace smoothing is applied to the emission probabilities and -log of each viterbi probability is computed
    #
    # It also returns a list of normalized predicted probabilities of each word
    def complex(self, words):
        result = []
        result_prob = []
        viterbi_dict = {}
        viterbi_dict_back = {}
        predicted_sentence = ''
        marginal_prob = []
        count = 1
        for index in range(len(words)):
            if index == 0:
                for key in list(self.sentence_dict.keys()):
                    emission_key = words[index]
                    if (emission_key + "_" + key) not in self.emission_probabilities:
                        self.emission_probabilities[emission_key + "_" + key] = float(1) / float(
                            self.part_of_speech[key] + self.total_words)

                    prob = -math.log(self.initial_probabilities[key]) -math.log(self.emission_probabilities[emission_key + "_" + key])

                    if count in viterbi_dict:
                        viterbi_dict[count]['v' + "_" + str(count) + "_" + key] = prob
                    else:
                        viterbi_dict[count] = {'v' + "_" + str(count) + "_" + key: prob}
                count = count + 1

            elif index == 1:
                for key1 in list(self.sentence_dict.keys()):
                    temp_dict2 = {}
                    emission_key = words[index]
                    if (emission_key + "_" + key1) not in self.emission_probabilities:
                        self.emission_probabilities[emission_key + "_" + key1] = float(1) / float(
                            self.part_of_speech[key1] + self.total_words)

                    for key2 in list(self.sentence_dict.keys()):
                        prob = viterbi_dict[count - 1]['v' + "_" + str(count - 1) + "_" + key2] -math.log(self.transition_probabilities[key1 + '|' + key2])
                        temp_dict2 = self.update_dictionary_complex(temp_dict2, key2, prob)

                    if count in viterbi_dict:
                        viterbi_dict[count]['v' + "_" + str(count) + "_" + key1] = temp_dict2[temp_dict2.keys()[0]] -math.log(self.emission_probabilities[emission_key + "_" + key1])

                    else:
                        viterbi_dict[count] = {'v' + "_" + str(count) + "_" + key1: temp_dict2[temp_dict2.keys()[0]] -math.log(self.emission_probabilities[emission_key + "_" + key1])}

                    viterbi_dict_back['v' + "_" + str(count) + "_" + key1] = temp_dict2.keys()[0]
                count = count + 1

            else:
                for key1 in list(self.sentence_dict.keys()):
                    temp_dict2 = {}
                    emission_key = words[index]
                    if (emission_key + "_" + key1) not in self.emission_probabilities:
                        self.emission_probabilities[emission_key + "_" + key1] = float(1) / float(
                            self.part_of_speech[key1] + self.total_words)

                    for key2 in list(self.sentence_dict.keys()):
                        for key3 in list(self.sentence_dict.keys()):
                            prob = viterbi_dict[count - 2]['v' + "_" + str(count - 2) + "_" + key3] + viterbi_dict[count - 1]['v' + "_" + str(count - 1) + "_" + key2] -math.log(self.transition_probabilities_second_order[key1 + '|' + key3+"_"+key2])
                            temp_dict2 = self.update_dictionary_complex(temp_dict2, key2, prob)

                    if count in viterbi_dict:
                        viterbi_dict[count]['v' + "_" + str(count) + "_" + key1] = temp_dict2[temp_dict2.keys()[0]] -math.log(self.emission_probabilities[emission_key + "_" + key1])

                    else:
                        viterbi_dict[count] = {'v' + "_" + str(count) + "_" + key1: temp_dict2[temp_dict2.keys()[0]] -math.log(self.emission_probabilities[emission_key + "_" + key1])}

                    viterbi_dict_back['v' + "_" + str(count) + "_" + key1] = temp_dict2.keys()[0]
                count = count + 1

        count = count - 1
        predicted_key = min(viterbi_dict[count].iteritems(), key=operator.itemgetter(1))[0]
        predicted_sentence = predicted_sentence + predicted_key.split("_")[2] + " "
        marginal_prob.append(viterbi_dict[count]['v' + "_" + str(count) + "_" + predicted_key.split("_")[2]])
        for index in reversed(range(len(words) - 1)):
            count = count - 1
            predicted_key_new = 'v' + "_" + str(count) + "_" + viterbi_dict_back[predicted_key]
            predicted_sentence = predicted_sentence + viterbi_dict_back[predicted_key] + " "
            marginal_prob.append(viterbi_dict[count][predicted_key_new])
            predicted_key = copy.deepcopy(predicted_key_new)
        for c in reversed(predicted_sentence.split()):
            result.append(c)
        for v in reversed(marginal_prob):
            result_prob.append(v)
        sum_result_prob = sum(result_prob)
        result_prob = [i/float(sum_result_prob) for i in result_prob]
        return [[result], [result_prob]]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"

