import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

from sklearn.model_selection import train_test_split

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('basecode/mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.

    print(mat.keys())
    train_set, labels_set = None, None
    test_data, test_label = None, None

    for i in range(10):
        train_key, test_key = "train"+str(i), "test"+str(i)
        train_value, test_value = mat[train_key], mat[test_key]
        train_labels = np.full((len(train_value),), i)
        test_labels = np.full((len(test_value),), i)
        if train_set is None:
            train_set, labels_set = train_value, train_labels
            test_data, test_label = test_value, test_labels
        else:
            train_set = np.concatenate((train_set, train_value), axis=0)
            labels_set = np.concatenate((labels_set, train_labels), axis=0)
            test_data = np.concatenate((test_data, test_value), axis=0)
            test_label = np.concatenate((test_label, test_labels), axis=0)

    # Split the training data into 50,000 for training and 10,000 for validation
    train_data, validation_data, train_label, validation_label = train_test_split(
        train_set, labels_set, test_size=(10000/60000), random_state=42)
    

    # Feature selection
    # Your code here.

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    N = training_data.shape[0]
    training_data_with_bias = np.column_stack((np.ones((N,)), training_data))
    hidden_input = np.dot(training_data_with_bias, w1.T)
    hidden_output = sigmoid(hidden_input)

    hidden_output_with_bias = np.column_stack((np.ones((N,)), hidden_output))
    output_input = np.dot(hidden_output_with_bias, w2.T)
    output = sigmoid(output_input)

    y = np.zeros((N, n_class))
    y[np.arange(N), training_label] = 1 # one-hot encoding

    error = y * np.log(output) + (1 - y) * np.log(1 - output)
    data_loss = -np.sum(error) / N

    regularization = (lambdaval / (2 * N)) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))
    obj_val = data_loss + regularization

    # Backpropagation
    delta_output = output - y
    grad_w2 = np.dot(delta_output.T, hidden_output_with_bias) / N
    
    delta_hidden = np.dot(delta_output, w2[:, 1:]) * hidden_output * (1 - hidden_output)
    grad_w1 = np.dot(delta_hidden.T, training_data_with_bias) / N

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    # obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here

    N = data.shape[0]
    data_with_bias = np.column_stack((np.ones((N,)), data))
    hidden_input = np.dot(data_with_bias, w1.T)
    hidden_output = sigmoid(hidden_input)

    hidden_output_with_bias = np.column_stack((np.ones((N,)), hidden_output))
    output_input = np.dot(hidden_output_with_bias, w2.T)
    output = sigmoid(output_input)

    labels = np.argmax(output, axis=1)
    return labels

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_sample_images(data, true_labels, pred_labels, num_samples=16):
    """
    Randomly select num_samples from the data and display them with their true and predicted labels.
    Assumes that each row in data is a flattened image (28*28 = 784 features).
    """
    indices = np.random.choice(range(data.shape[0]), num_samples, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(8,8))
    axes = axes.ravel()
    for i, idx in enumerate(indices):
        img = data[idx].reshape(28,28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {true_labels[idx]}\nPred: {pred_labels[idx]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion(true_labels, pred_labels):
    """
    Plot a confusion matrix using the true and predicted labels.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8,8))
    plt.matshow(cm, cmap='Blues', fignum=1)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix", pad=20)
    plt.show()


"""**************Neural Network Script Starts here********************************"""
if __name__ == "__main__":
    
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 50

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset

    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
    plot_sample_images(test_data, test_label, predicted_label)
    plot_confusion(test_label, predicted_label)


