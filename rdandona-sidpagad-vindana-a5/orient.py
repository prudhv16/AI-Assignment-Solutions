import sys
import neuralnetwork
import adaBoost
import knn


input_args = sys.argv[1:]

if input_args[3] == "best":
    train_file, test_file, technique = input_args[:3]

    adaBoost.adaboost_classifier(train_file, test_file, technique, str(11))
else:
    train_file, test_file, technique = input_args[:3]

if technique == "nnet":
    model_parameter = input_args[3]
    neuralnetwork.nnet(train_file, test_file, model_parameter)
elif technique == "adaboost":
    adaBoost.adaboost_classifier(train_file, test_file, technique, input_args[3])
elif technique == "nearest":
    knn.knn_classifier(train_file, test_file, technique)
else:
    pass


