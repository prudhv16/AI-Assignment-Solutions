###################################
# CS B551 Fall 2016, Assignment #4
#
# Your names and user ids:
# vindana (Venkata Prudhvi Raj Indana)
# sidpagad (Siddhartha Pagadala)
# rdandona (Rohit Dandona)
#
####
import os
import math
import pickle
import operator

# The confusion_matrix function determines and returns the confusion matrix 
# of the predicted and actual class labels
#
# This function returns a list of lists of the matrix.
def confusion_matrix(actual,predictions,uniquetopics):
    length = len(uniquetopics)
    confusion = [[0]*length for x in range(length)]
    length = len(actual)
    i = 0
    for i in range(length):
        index1 = uniquetopics.index(actual[i])
        index2 = uniquetopics.index(predictions[i])
        confusion[index1][index2] += 1
    return confusion

def getaccuracy(actual, predictions):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            correct += 1
    return (correct/float(len(actual))) * 100.0

# The get_word_dict_continuous function reads the train data and determines frequency of 
# each individual word accross the entire train corpus and stores in a dictionary seperately
# for spam and notspam. this function is specific to continuous features
#
# This function returns the frequency dictionary, number of documents and all the unique words 
def get_word_dict_continuous(train_directory_path,category,distinct_words):
    dict = {}
    file_dict = {}
    count = 0
    stopwords = set(
        [u'all', u'just', u'being', u'over', u'both', u'through', u'yourselves', u'its', u'before', u'o', u'hadn',
         u'herself', u'll', u'had', u'should', u'to', u'only', u'won', u'under', u'ours', u'has', u'do', u'them',
         u'his', u'very', u'they', u'not', u'during', u'now', u'him', u'nor', u'd', u'did', u'didn', u'this', u'she',
         u'each', u'further', u'where', u'few', u'because', u'doing', u'some', u'hasn', u'are', u'our', u'ourselves',
         u'out', u'what', u'for', u'while', u're', u'does', u'above', u'between', u'mustn', u't', u'be', u'we', u'who',
         u'were', u'here', u'shouldn', u'hers', u'by', u'on', u'about', u'couldn', u'of', u'against', u's', u'isn',
         u'or', u'own', u'into', u'yourself', u'down', u'mightn', u'wasn', u'your', u'from', u'her', u'their', u'aren',
         u'there', u'been', u'whom', u'too', u'wouldn', u'themselves', u'weren', u'was', u'until', u'more', u'himself',
         u'that', u'but', u'don', u'with', u'than', u'those', u'he', u'me', u'myself', u'ma', u'these', u'up', u'will',
         u'below', u'ain', u'can', u'theirs', u'my', u'and', u've', u'then', u'is', u'am', u'it', u'doesn', u'an',
         u'as', u'itself', u'at', u'have', u'in', u'any', u'if', u'again', u'no', u'when', u'same', u'how', u'other',
         u'which', u'you', u'shan', u'needn', u'haven', u'after', u'most', u'such', u'why', u'a', u'off', u'i', u'm',
         u'yours', u'so', u'y', u'the', u'having', u'once'])
    filedir = os.listdir(os.path.join(train_directory_path,category))
    number_docs = len(filedir)
    for fileName in filedir:
        with open(os.path.join(train_directory_path,category+"/"+fileName),'r') as f:
            file_words = []
            for line in f:
                for word in line.split():
                    lower_case_word = word.lower().strip()
                    if lower_case_word not in stopwords:

                        if lower_case_word not in distinct_words:
                            distinct_words[lower_case_word] = len(distinct_words)
                        if lower_case_word not in dict:
                            dict[lower_case_word] = 1
                        else:
                            dict[lower_case_word] = dict[lower_case_word] + 1
                        if lower_case_word not in file_words:
                            file_words.append(lower_case_word)

            file_dict[count] = file_words
            f.close()
            count = count + 1
    return dict, file_dict, number_docs, distinct_words

# The predict_continuous function performs predictions for continuous features
#
# This function returns a list of predicted and actual class labels
def predict_continuous(category, test_directory_path, actual, predicted, prior_spam, prior_nonspam, spam_prob_dict, notspam_prob_dict):
    for fileName in os.listdir(os.path.join(test_directory_path,category)):
        with open(os.path.join(test_directory_path,category+"/"+fileName),'r') as f:
            p_spam = 0
            p_nonspam = 0
            if category == "spam":
                actual.append(0)
            else:
                actual.append(1)
            test_file_words = {}

            for line in f:
                for word in line.split():
                    lower_case_word = word.lower().strip()
                    if lower_case_word not in test_file_words:
                        test_file_words[lower_case_word] = 1
                    else:
                        test_file_words[lower_case_word] = test_file_words[lower_case_word] + 1

            for key in test_file_words:
                if key in spam_prob_dict:
                    p_spam = p_spam + (math.log(spam_prob_dict[key]) * test_file_words[key])
                if key in notspam_prob_dict:
                    p_nonspam = p_nonspam + (math.log(notspam_prob_dict[key]) * test_file_words[key])

            p_spam = p_spam + math.log(prior_spam)
            p_nonspam = p_nonspam + math.log(prior_nonspam)

            if p_nonspam <= p_spam:
                predicted.append(1)
            else:
                predicted.append(0)

    return actual, predicted, test_file_words

# This function creates a list of lists of the train data of continuous features
def create_vector_continuous(individualfile_dict_spam, individualfile_dict_nonspam, distinct_words, spam_dict, notspam_dict):
    input_vector = []
    for file_no in individualfile_dict_spam:
        print(len(distinct_words))
        row = [0] * (len(distinct_words)+1)
        for word in individualfile_dict_spam[file_no]:
            row[distinct_words[word]] = spam_dict[word]
        row[len(distinct_words)] = 0
        input_vector.append(row)

    for file_no in individualfile_dict_nonspam:
        print(len(distinct_words))
        row = [0] * (len(distinct_words)+1)
        for word in individualfile_dict_nonspam[file_no]:
            row[distinct_words[word]] = notspam_dict[word]
        row[len(distinct_words)] = 1
        input_vector.append(row)

    return input_vector

# The get_word_dict_binary function reads the train data and determines the number of emails in which a word occurs 
# and stores in a dictionary seperately for spam and notspam. This function is specific to binary features
#
# This function returns the frequency dictionary, number of documents and all the unique words
def get_word_dict_binary(train_directory_path, category, distinct_words):
    dict = {}
    file_dict = {}
    count = 0
    stopwords = set(
        [u'all', u'just', u'being', u'over', u'both', u'through', u'yourselves', u'its', u'before', u'o', u'hadn',
         u'herself', u'll', u'had', u'should', u'to', u'only', u'won', u'under', u'ours', u'has', u'do', u'them',
         u'his', u'very', u'they', u'not', u'during', u'now', u'him', u'nor', u'd', u'did', u'didn', u'this', u'she',
         u'each', u'further', u'where', u'few', u'because', u'doing', u'some', u'hasn', u'are', u'our', u'ourselves',
         u'out', u'what', u'for', u'while', u're', u'does', u'above', u'between', u'mustn', u't', u'be', u'we', u'who',
         u'were', u'here', u'shouldn', u'hers', u'by', u'on', u'about', u'couldn', u'of', u'against', u's', u'isn',
         u'or', u'own', u'into', u'yourself', u'down', u'mightn', u'wasn', u'your', u'from', u'her', u'their', u'aren',
         u'there', u'been', u'whom', u'too', u'wouldn', u'themselves', u'weren', u'was', u'until', u'more', u'himself',
         u'that', u'but', u'don', u'with', u'than', u'those', u'he', u'me', u'myself', u'ma', u'these', u'up', u'will',
         u'below', u'ain', u'can', u'theirs', u'my', u'and', u've', u'then', u'is', u'am', u'it', u'doesn', u'an',
         u'as', u'itself', u'at', u'have', u'in', u'any', u'if', u'again', u'no', u'when', u'same', u'how', u'other',
         u'which', u'you', u'shan', u'needn', u'haven', u'after', u'most', u'such', u'why', u'a', u'off', u'i', u'm',
         u'yours', u'so', u'y', u'the', u'having', u'once'])
    filedir = os.listdir(os.path.join(train_directory_path, category))
    number_docs = len(filedir)
    for fileName in filedir:
        with open(os.path.join(train_directory_path,category+"/"+fileName),'r') as f:
            file_words = []
            for line in f:
                for word in line.split():
                    lower_case_word = word.lower().strip()
                    if lower_case_word not in stopwords:
                        if lower_case_word not in distinct_words:
                            distinct_words[lower_case_word] = len(distinct_words)
                        if lower_case_word not in dict:
                            dict[lower_case_word] = 1
                        else:
                            if lower_case_word not in file_words:
                                dict[lower_case_word] = dict[lower_case_word] + 1

                        if lower_case_word not in file_words:
                            file_words.append(lower_case_word)

            file_dict[count]=file_words
            f.close()
	    count = count+1

    return dict, file_dict, number_docs, distinct_words

# The predict_binary function performs predictions for binary features
#
# This function returns a list of predicted and actual class labels
def predict_binary(category, test_directory_path, actual, predicted, prior_spam, prior_nonspam, spam_prob_dict, notspam_prob_dict):
    for fileName in os.listdir(os.path.join(test_directory_path,category)):
        with open(os.path.join(test_directory_path,category+"/"+fileName),'r') as f:
            p_spam = 0
            p_nonspam = 0
            if category == "spam":
                actual.append(0)
            else:
                actual.append(1)
            test_file_words = []

            for line in f:
                for word in line.split():
                    lower_case_word = word.lower().strip()
                    if lower_case_word not in test_file_words:
                        test_file_words.append(lower_case_word)
            '''
            for key in spam_prob_dict:
                if key in test_file_words:
                    p_spam = p_spam + (math.log(spam_prob_dict[key]))
                #else:
                    #p_spam = p_spam + (1-math.log(spam_prob_dict[key]))

            for key in notspam_prob_dict:
                if key in test_file_words:
                    p_nonspam = p_nonspam + (math.log(notspam_prob_dict[key]))
                #else:
                    #p_nonspam = p_nonspam + (1-math.log(notspam_prob_dict[key]))
            '''
            for key in test_file_words:
                if key in spam_prob_dict:
                    p_spam = p_spam + (math.log(spam_prob_dict[key]))
                if key in notspam_prob_dict:
                    p_nonspam = p_nonspam + (math.log(notspam_prob_dict[key]))

            p_spam = p_spam + math.log(prior_spam)
            p_nonspam = p_nonspam + math.log(prior_nonspam)

            if p_nonspam <= p_spam:
                predicted.append(1)
            else:
                predicted.append(0)

    return actual, predicted, test_file_words

# This function creates a list of lists of the train data of binary features
def create_vector_binary(individualfile_dict_spam, individualfile_dict_nonspam, distinct_words):
    input_vector = []
    for file_no in individualfile_dict_spam:
        row = [0] * (len(distinct_words)+1)
        for word in individualfile_dict_spam[file_no]:
            row[distinct_words[word]] = 1
        row[len(distinct_words)] = 0
        input_vector.append(row)

    for file_no in individualfile_dict_nonspam:
        row = [0] * (len(distinct_words)+1)
        for word in individualfile_dict_nonspam[file_no]:
            row[distinct_words[word]] = 1
        row[len(distinct_words)] = 1
        input_vector.append(row)

    return input_vector

# Train model for both spam and notspam train data
def bayes_train(train_directory_path, model_file_path):

    distinct_words = {}
    spam_dict, individualfile_dict_spam, no_spam_files, distinct_words = get_word_dict_binary(train_directory_path,"spam",distinct_words)
    notspam_dict, individualfile_dict_nonspam, no_nonspam_files, distinct_words = get_word_dict_binary(train_directory_path,"notspam",distinct_words)

    spam_prob_dict_binary = {}
    notspam_prob_dict_binary = {}

    for key in spam_dict:
        spam_prob_dict_binary[key] = (float(spam_dict[key]+1) / float(no_spam_files+2))

    for key in notspam_dict:
        notspam_prob_dict_binary[key] = (float(notspam_dict[key]+1) / float(no_nonspam_files+2))

    distinct_words = {}
    spam_dict, individualfile_dict_spam, no_spam_files, distinct_words = get_word_dict_continuous(train_directory_path, "spam", distinct_words)
    notspam_dict, individualfile_dict_nonspam, no_nonspam_files, distinct_words = get_word_dict_continuous(train_directory_path, "notspam", distinct_words)

    # input_vector = create_vector_continuous(individualfile_dict_spam, individualfile_dict_nonspam, distinct_words, spam_dict, notspam_dict)

    spam_prob_dict_continuous = {}
    notspam_prob_dict_continuous = {}

    no_spam_words = sum(spam_dict.values())
    no_nonspam_words = sum(notspam_dict.values())

    for key in spam_dict:
        spam_prob_dict_continuous[key] = (float(spam_dict[key] + 1) / float(no_spam_words + 2))

    for key in notspam_dict:
        notspam_prob_dict_continuous[key] = (float(notspam_dict[key] + 1) / float(no_nonspam_words + 2))

    total_files = no_spam_files + no_nonspam_files

    prior_spam = float(no_spam_files) / float(total_files)
    prior_nonspam = float(no_nonspam_files) / float(total_files)

    top_ten_binary = dict(sorted(spam_prob_dict_binary.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])
    bottom_ten_binary = dict(sorted(spam_prob_dict_binary.iteritems(), key=operator.itemgetter(1), reverse=False)[:10])
    top_ten_continuous = dict(sorted(spam_prob_dict_continuous.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])
    bottom_ten_continuous = dict(sorted(spam_prob_dict_continuous.iteritems(), key=operator.itemgetter(1), reverse=False)[:10])

    pickle_out = open(model_file_path, "wb")
    pickle.dump((spam_prob_dict_binary, notspam_prob_dict_binary, spam_prob_dict_continuous,
                 notspam_prob_dict_continuous, prior_spam, prior_nonspam, distinct_words, top_ten_binary, bottom_ten_binary, top_ten_continuous, bottom_ten_continuous), pickle_out)
    pickle_out.close()

    print "Training complete! Model created"
    print"\t"

# Test the model for both spam and notspam test data
def bayes_test(test_directory_path, model_file_path):

    with open(model_file_path, "rb") as fh:
        tup = pickle.load(fh)

    spam_prob_dict_binary = tup[0]
    notspam_prob_dict_binary = tup[1]
    spam_prob_dict_continuous = tup[2]
    notspam_prob_dict_continuous = tup[3]
    prior_spam = tup[4]
    prior_nonspam = tup[5]
    distinct_words = tup[6]
    top_ten_binary = tup[7]
    bottom_ten_binary = tup[8]
    top_ten_continuous = tup[9]
    bottom_ten_continuous = tup[10]

    actual = []
    predicted = []

    actual, predicted, test_file_words_spam = predict_binary("spam", test_directory_path, actual, predicted, prior_spam, prior_nonspam, spam_prob_dict_binary, notspam_prob_dict_binary)
    actual, predicted, test_file_words_nonspam = predict_binary("notspam", test_directory_path, actual, predicted, prior_spam, prior_nonspam, spam_prob_dict_binary, notspam_prob_dict_binary)
    
    print "\t"
    print "Results obtained for the bag of words model with binary features:"
    print "-----------------------------------------------------------------"
    print "Confusion Matrix:"
    confusion = confusion_matrix(actual, predicted, [0,1])
    print "\n".join([" ".join(map(str, x)) for x in confusion])
    print "\t"
    print "Accuracy:"
    acc = getaccuracy(actual, predicted)
    print (str(acc))
    print "\t"
    print "Top 10 words most associated with spam:"
    print top_ten_binary.keys()
    print "\t"
    print "Bottom 10 words most associated with spam:"
    print bottom_ten_binary.keys()
    print "\t"
    print "\t"

    actual = []
    predicted = []

    actual, predicted, test_file_words_spam = predict_continuous("spam", test_directory_path, actual, predicted, prior_spam, prior_nonspam, spam_prob_dict_continuous, notspam_prob_dict_continuous)
    actual, predicted, test_file_words_nonspam = predict_continuous("notspam", test_directory_path, actual, predicted, prior_spam, prior_nonspam, spam_prob_dict_continuous, notspam_prob_dict_continuous)

    print "Results obtained for the bag of words model with continuous features:"
    print "---------------------------------------------------------------------"
    print "Confusion Matrix:"
    confusion = confusion_matrix(actual, predicted, [0, 1])
    print "\n".join([" ".join(map(str, x)) for x in confusion])
    print "\t"
    print "Accuracy:"
    acc = getaccuracy(actual, predicted)
    print (str(acc))
    print "\t"
    print "Top 10 words most associated with spam:"
    print top_ten_continuous.keys()
    print "\t"
    print "Bottom 10 words most associated with spam:"
    print bottom_ten_continuous.keys()
    print "\t" 
