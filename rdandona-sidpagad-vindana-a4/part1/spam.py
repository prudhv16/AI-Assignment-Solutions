###################################
# CS B551 Fall 2016, Assignment #4
#
# Your names and user ids:
# vindana (Venkata Prudhvi Raj Indana)
# sidpagad (Siddhartha Pagadala)
# rdandona (Rohit Dandona)
#
####
'''
Problem 1

Formulation of the problem:

Seperate dictionaries have been created with the likelihood probabilities of each unique word in the train repository for spam and notspam emails.

In the bag of words with binary features approach:
Train: The number of emails in which a word occurs is captured and is stored in a dictionary (seperately for spam and not spam) and is divided by 
the number of spam (or not spam) emails to get the likelihood probability of each word in the train set.
Test: The sum of the log equivalent of the probabilities calculated in train is calculated for each word of a test email for spam/notspam and is 
finally multiplied by the prior probability. The one with the higher is chosen as the label of the email.

In the bag of words with continuous features approach:
Train: Frequency of each individual word is captured accross the entire train corpus and is stored in a dictionary (seperately for spam and not spam) 
and is divided by the total number occurence of words in the spam (or not spam) corpus to get the likelihood probability of each word in the train set.
Test: The sum of the product of the log equivalent of the probabilities calculated in train and the frequency of each individual word obtained in train, 
is calculated for spam/notspam and is finally multiplied by the prior probability. The one with the higher is chosen as the label of the email.


The results obtained for the bayes approach is follows:

Results obtained for the bag of words model with binary features:
-----------------------------------------------------------------
Confusion Matrix:
987 198
129 1240

Accuracy:
87.1965544244

Top 10 words most associated with spam:
['to:', 'message-id:', 'received:', 'date:', '2002', 'esmtp', 'subject:', 'return-path:', 'id', 'from:']

Bottom 10 words most associated with spam:
['face="arial">-', 'karenc777@hotmail.com', '00556.098b57f5108ba34d21825f176e492786', '<0gu0006e91uuwz@chimmx04.algx.net>', '15:51:44', '15:51:47', 
'15:51:43', 'dqo8uefsqu0gtkfnrt0itw92awuiifzbtfvfpsjodhrwoi8vd3d3lm1sbxjlcg9ydgvylmnv', '<txoaplc@go-fishing.co.uk>', 'kalmadan']


Results obtained for the bag of words model with continuous features:
---------------------------------------------------------------------
Confusion Matrix:
1105 80
25 1344

Accuracy:
95.8888018794

Top 10 words most associated with spam:
['<td', 'received:', '<br>', '</tr>', '2002', 'esmtp', '+0100', ';', '=', 'id']

Bottom 10 words most associated with spam:
['karenc777@hotmail.com', '<0gu0006e91uuwz@chimmx04.algx.net>', '([202.93.254.118])', '"office@hotwebcenter.com">', '15:51:44', '15:51:47', '15:51:43', 
'dqo8uefsqu0gtkfnrt0itw92awuiifzbtfvfpsjodhrwoi8vd3d3lm1sbxjlcg9ydgvylmnv', '<nekb499a3@hotmail.com>', 'kalmadan']

Decision Tree:
Methodology : Entropy is being used to select best features at each level of learning Decision Tree
Since there are lot of features we got accuracy of 0.54 .
Accuracy for binary features: 54%
Accuracy for continuous features with binning: 61%

Comparing the accuracies of all the above learning models, it is being observed that the Naive Bayes classifier with frequencies as features 
performed better in classifying test data with accuracy of 96 %.

'''
####
import sys
import technique
import DecisionTree as dt;
import cPickle as pickle;
import os;


def cal_accuracy(test_data, tree_learn):
	predicted_result = []
	predicted = []
	FP = 0
	TP = 0
	FN = 0
	TN = 0
	for e, row in enumerate(test_data):
		result_maxVote = int(dt.classify(row[1:], tree_learn))
		predicted.append(result_maxVote)
		predicted_result.append(result_maxVote == int(row[-1]))
		if result_maxVote == int(row[-1]):
			if int(row[-1]) == 0:
				TN += 1
			else:
				TP += 1
		else:
			if int(row[-1]) == 0:
				FP += 1
			else:
				FN += 1

	accuracy = 0
	# Accuracy
	accuracy = float(predicted_result.count(True)) / float(len(predicted_result))
	# print("accuracy: %.4f" % accuracy)
	return accuracy, predicted, TP, TN, FP, FN


def create_vector(train_directory_path, category, distinct_words):
	count = 0
	file_dict = {}
	for fileName in os.listdir(os.path.join(train_directory_path, category)):
		with open(os.path.join(train_directory_path, category + "/" + fileName), 'r') as f:

			test_file_words = []

			for line in f:
				for word in line.split():
					lower_case_word = word.lower().strip()
					if lower_case_word not in test_file_words and lower_case_word in distinct_words:
						test_file_words.append(lower_case_word)
			file_dict[count] = test_file_words
			count += 1
	return file_dict

if len(sys.argv) != 5:
    print "Usage: spam.py mode technique dataset-directory model-file"
    sys.exit()

mode, techniq, train_directory_path, model_file_path = sys.argv[1:5]

if mode == "train" and techniq == "bayes":
    technique.bayes_train(train_directory_path,model_file_path)
elif mode == "train" and techniq == "dt":
    distinct_words = {}
    spam_dict, individualfile_dict_spam, no_spam_files, distinct_words = technique.get_word_dict_binary(train_directory_path,"spam",distinct_words)
    notspam_dict, individualfile_dict_nonspam, no_nonspam_files, distinct_words = technique.get_word_dict_binary(train_directory_path,"notspam",distinct_words)

    input_vector = technique.create_vector_binary(individualfile_dict_spam, individualfile_dict_nonspam, distinct_words)
	
    tree_learn = None
    tdepth = 9999
    tree_learn = dt.buildTree(input_vector, dt.TreeNode(None), tdepth)
	
    pickle_out = open(model_file_path, "wb")
    pickle.dump((distinct_words,tree_learn), pickle_out)
    pickle_out.close()

    print "Training complete! Model created"
    print"\t"

elif mode == "test" and techniq == "bayes":
    technique.bayes_test(train_directory_path,model_file_path)
elif mode == "test" and techniq == "dt":

    with open(model_file_path, "rb") as fh:
		tup = pickle.load(fh)

    distinct_words = tup[0]
    tree_learn = tup[1]

		
    test_words_spam = create_vector(train_directory_path,"spam", distinct_words)
    test_words_notspam = create_vector(train_directory_path,"notspam", distinct_words)
	
    test_vector = technique.create_vector_binary(test_words_spam, test_words_notspam, distinct_words)
	
    accuracy=cal_accuracy(test_vector, tree_learn)
    TP = accuracy[2]
    TN = accuracy[3]
    FP = accuracy[4]
    FN = accuracy[5]
    print "Accuracy:",accuracy[0]
    #print "Confusion Matrix:"
    #dt.print_confusion_matrix(TP, TN, FP, FN, ['0', '1'])


