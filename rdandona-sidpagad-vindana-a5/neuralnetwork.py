import numpy as np
import math
import sys
import copy
import os.path
import time

def get_data_subset(dataset, percent):
    fraction = float(percent)/float(100)
    dataset_new = dataset[(np.random.choice(len(dataset), (int(len(dataset) * fraction)), replace=False))]
    return dataset_new

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

def activation_function(z):
    sigmoid_value = 1.0 / (1.0 + np.exp(np.negative(z)))
    #sigmoid_value = np.tanh(z)
    return sigmoid_value

def getaccuracy(actual, predictions):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            correct += 1
    return (correct/float(len(actual))) * 100.0

def activation_function_derivation(z):
    sigmoid_value = activation_function(z)
    #return 1 - np.square(np.tanh(z))
    return sigmoid_value

def predict(output_val):
    output_val_list = output_val.tolist()
    y_dash = output_val_list.index(max(output_val_list))
    if y_dash == 0:
        return 0.0
    elif y_dash == 1:
        return 90.0
    elif y_dash == 2:
        return 180.0
    else:
        return 270.0

def get_vector(y):
    if y == 0.0:
        return np.asarray([1,0,0,0])
    elif y == 90.0:
        return np.asarray([0,1,0,0])
    elif y == 180.0:
        return np.asarray([0,0,1,0])
    else:
        return np.asarray([0,0,0,1])


def data_preprocessing(dataset):
    x_train = dataset[:, 2:]
    # Normalizing
    x_train = np.true_divide(x_train, 255)
    x_train -= np.mean(x_train)
    x_train /= np.std(x_train)
    y_train = dataset[:, 1:2]
    return x_train, y_train

def neuron(input_row, hidden_weights):
    dot_product_out_temp = []
    post_activation_t = []
    for row in hidden_weights:
        temp = np.dot(input_row, row.T)
        post_activation_t.append(temp)
        dot_product_out_temp.append(activation_function(temp))
    dot_product_out = np.asarray(dot_product_out_temp)
    post_activation = np.asarray(post_activation_t)
    return dot_product_out, post_activation

def write_to_output(test_file, y_predict):
    dataset = np.loadtxt(test_file, dtype=str)
    image_names = dataset[:, 0:1]
    file_name = "nnet_output.txt"
    if os.path.isfile(file_name):
        open(file_name, 'w').close()
    file = open(file_name, "a")
    for index in range(len(image_names)):
        file.write(str(image_names[index][0])+" "+str(int(y_predict[index]))+"\n")
    file.close()

def nnet(train_file, test_file, model_parameter):
    start_time = time.time()
    print "Pre-processing data..."
    dataset = np.genfromtxt(train_file)

    '''
    Note: The following two lines of code needs to be uncommented if a
    percentage of the train set is to be used for training the model. The
    desired percentage needs to be hard coded.
    '''
    #percent = 20
    #dataset = get_data_subset(dataset, percent)

    x_train, y_train= data_preprocessing(dataset)
    hidden_nodes = int(model_parameter)
    step_size = 0.01
    hidden_weights = np.random.uniform(low=-1, high=1, size=(hidden_nodes,x_train.shape[1]))
    output_nodes = 4
    output_weights = np.random.uniform(low=-1, high=1, size=(output_nodes,hidden_nodes))

    print "Training..."
    # Training
    epochs = 1
    for rep in range(epochs):
        print "Epoch: ", rep + 1
        index = 0
        np.random.shuffle(dataset)
        x_train, y_train = data_preprocessing(dataset)
        for input_row in x_train:

            # Hidden layer
            h_out, ni = neuron(input_row, hidden_weights)
            # Output layer
            o_out, nj = neuron(h_out, output_weights)

            # Back propogation
            y_vector = get_vector(y_train[index])
            delta_j = np.multiply(np.subtract(y_vector, o_out), activation_function_derivation(nj))
            delta_i = np.multiply(np.dot(output_weights.T,delta_j), activation_function_derivation(ni))

            h_out_new = np.asarray([h_out.tolist()])
            delta_j_new = np.asarray([delta_j.tolist()])
            output_weights = np.add((step_size * np.dot(h_out_new.T, delta_j_new)).T, output_weights)

            input_row_new = np.asarray([input_row.tolist()])
            delta_i_new = np.asarray([delta_i.tolist()])
            hidden_weights = np.add((step_size * np.dot(input_row_new.T, delta_i_new)).T, hidden_weights)

            index = index + 1
            sys.stdout.write("\r" +"Rows processed: "+str(index))
            sys.stdout.flush()
        print "\n"

    print "Testing..."
    # Testing
    dataset = np.genfromtxt(test_file)
    x_test, y_test= data_preprocessing(dataset)
    y_predict = []
    for input_row in x_test:
        h_out, ni = neuron(input_row, hidden_weights)
        o_out, nj = neuron(h_out, output_weights)
        y_predict.append(predict(o_out))

    print "Accuracy with "+model_parameter+" hidden layer neurons and "+str(step_size)+" step size : ",getaccuracy(y_test.T.tolist()[0],y_predict)
    write_to_output(test_file, y_predict)
    print "Output file created!"
    print("--- %s seconds ---" % (time.time() - start_time))
    print

    uniquetopics = [0,90,180,270]
    uniquetopics_str = ['0', '90', '180', '270']
    confusion = confusion_matrix(y_test.T.tolist()[0], y_predict, uniquetopics)
    print "confusion matrix \n"
    print "Predicted".center(30, '*')
    print '      '.join(uniquetopics_str)
    print ''.join(['*'] * 30)
    for item in confusion:
        x = map(str, item)
        for j in range(len(x)):
            print x[j], " " * (5 - len(x[j])),
        print
        
