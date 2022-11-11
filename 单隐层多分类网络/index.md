# 单隐层多分类网络


# 单隐层多分类神经网络（numpy实现）
使用Numpy实现，并且使用命令行的形式设定参数。

是一个作业里面的，实现的时候踩了一些坑，主要是训练里面的，最主要的就是没有实现batch，所以对于batch数据就要加上一层循环了。还有就是shuffle函数，返回值不要和参数一样，意思就是要对一个固定的数据进行打乱，而不是对上一epoch的数据进行打乱，这个坑很难察觉到。

下面是代码：

```python

import numpy as np

import matplotlib.pyplot as plt

import argparse

import logging

from typing import Callable

  

# This takes care of command line argument parsing for you!

# To access a specific argument, simply access args.<argument name>.

parser = argparse.ArgumentParser()

parser.add_argument('train_input', type=str,

                    help='path to training input .csv file')

parser.add_argument('validation_input', type=str,

                    help='path to validation input .csv file')

parser.add_argument('train_out', type=str,

                    help='path to store prediction on training data')

parser.add_argument('validation_out', type=str,

                    help='path to store prediction on validation data')

parser.add_argument('metrics_out', type=str,

                    help='path to store training and testing metrics')

parser.add_argument('num_epoch', type=int,

                    help='number of training epochs')

parser.add_argument('hidden_units', type=int,

                    help='number of hidden units')

parser.add_argument('init_flag', type=int, choices=[1, 2],

                    help='weight initialization functions, 1: random')

parser.add_argument('learning_rate', type=float,

                    help='learning rate')

parser.add_argument('--debug', type=bool, default=False,

                    help='set to True to show logging')

  
  

def args2data(args) :

    '''

    No need to modify this function!

  

    Parse command line arguments, create train/test data and labels.

    :return:

    X_tr: train data *without label column and without bias folded in* (numpy array)

    y_tr: train label (numpy array)

    X_te: test data *without label column and without bias folded in* (numpy array)

    y_te: test label (numpy array)

    out_tr: predicted output for train data (file)

    out_te: predicted output for test data (file)

    out_metrics: output for train and test error (file)

    n_epochs: number of train epochs

    n_hid: number of hidden units

    init_flag: weight initialize flag -- 1 means random, 2 means zero

    lr: learning rate

    '''

    # Get data from arguments

    out_tr = args.train_out

    out_te = args.validation_out

    out_metrics = args.metrics_out

    n_epochs = args.num_epoch

    n_hid = args.hidden_units

    init_flag = args.init_flag

    lr = args.learning_rate

  

    X_tr = np.loadtxt(args.train_input, delimiter=',')

    y_tr = X_tr[:, 0].astype(int)

    X_tr = X_tr[:, 1:] # cut off label column

  

    X_te = np.loadtxt(args.validation_input, delimiter=',')

    y_te = X_te[:, 0].astype(int)

    X_te = X_te[:, 1:] # cut off label column

  

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,

            n_epochs, n_hid, init_flag, lr)

  
  

def shuffle(X, y, epoch):

    '''

    DO NOT modify this function.

  

    Permute the training data for SGD.

    :param X: The original input data in the order of the file.

    :param y: The original labels in the order of the file.

    :param epoch: The epoch number (0-indexed).

    :return: Permuted X and y training data for the epoch.

    '''

    np.random.seed(epoch)

    N = len(y)

    ordering = np.random.permutation(N)

    return X[ordering], y[ordering]

  
  

def random_init(shape):

    '''

    Randomly initialize a numpy array of the specified shape

    :param shape: list or tuple of shapes

    :return: initialized weights

    '''

    M, D = shape

    np.random.seed(M*D) # Don't change this line!

    # TODO: create the random matrix here!

    # Hint: numpy might have some useful function for this

    W = np.random.uniform(-0.1, 0.1, size=(M, D))

  

    return W

  
  

def zero_init(shape):

    '''

    Do not modify this function.

  

    Initialize a numpy array of the specified shape with zero

    :param shape: list or tuple of shapes

    :return: initialized weights

    '''

    return np.zeros(shape = shape)

  
  

def softmax(z: np.ndarray) -> np.ndarray:

    '''

    Implement softmax function.

    :param z: input logits of shape (num_classes,)

    :return: softmax output of shape (num_classes,)

    '''

    shiftz = z - np.max(z)

    expz = np.exp(shiftz)

    return expz / np.sum(expz)

  

def cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:

    '''

    Compute cross entropy loss.

    :param y: label (a number or an array containing a single element)

    :param y_hat: prediction with shape (num_classes,)

    :return: cross entropy loss

    '''

    if isinstance(y, np.ndarray):

        y = y[0]

    return -np.log(y_hat[y])

def d_softmax_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:

    '''

    Compute gradient of loss w.r.t. ** softmax input **.

    Note that here instead of calculating the gradient w.r.t. the softmax

    probabilities, we are directly computing gradient w.r.t. the softmax input.

  

    Try deriving the gradient yourself (see Question 1.2(b) on the written),

    and you'll see why we want to calculate this in a single step.

  

    :param y: label (a number or an array containing a single element)

    :param y_hat: predicted softmax probability with shape (num_classes,)

    :return: gradient with shape (num_classes,)

    '''

    if isinstance(y, np.ndarray):

        y = y[0]

    y_hat[y] -= 1

    return y_hat

  

class Sigmoid(object):

    def __init__(self):

        '''

        Initialize state for sigmoid activation layer

        '''

        # Create cache to hold values for backward pass

        self.cache: dict[str, np.ndarray] = dict()

  

    def forward(self, x: np.ndarray) -> np.ndarray:

        '''

        Implement sigmoid activation function.

        :param x: input of shape (num_classes,)

        :return: sigmoid output of shape (num_classes,)

        '''

        sigmoid = 1 / (1 + np.exp(-x))

        self.cache['sigmoid'] = sigmoid

        return sigmoid

    def backward(self, dz: np.ndarray) -> np.ndarray:

        '''

        :param dz: partial derivative of loss with respect to output of sigmoid activation

        :return: partial derivative of loss with respect to input of sigmoid activation

        '''

        sigmoid = self.cache['sigmoid']

        return dz * sigmoid * (1 - sigmoid)

  
  

# This refers to a function type that takes in a tuple of 2 integers (row, col)

# and returns a numpy array (which should have the specified dimensions).

  
  

class Linear(object):

    def __init__(self, input_size: int, output_size: int,

                 weight_init_fn, learning_rate: float):

        '''

        :param input_size: number of units in the input of the layer

                           *not including* the folded bias

        :param output_size: number of units in the output of the layer

        :param weight_init_fn: function that creates and initializes weight

                               matrices for layer. This function takes in a

                               tuple (row, col) and returns a matrix with

                               shape row x col.

        :param learning_rate: learning rate for SGD training updates

        '''

        # TODO: Initialize weight matrix for this layer - since we are

        # folding the bias into the weight matrix, be careful about the

        # shape you pass in.

        self.w = weight_init_fn((output_size, input_size + 1))

  

        # TODO: Initialize matrix to store gradient with respect to weights

        self.dw = np.zeros_like(self.w, dtype=np.float64)

  

        # TODO: Initialize learning rate for SGD

        self.lr = learning_rate

  

        # Create cache to hold certain values for backward pass

        self.cache: dict[str, np.ndarray] = dict()

  

    def forward(self, x: np.ndarray) -> np.ndarray:

        '''

        :param x: Input to linear layer with shape (input_size,)

                  where input_size *does not include* the folded bias.

                  In other words, the input does not contain the bias column

                  and you will need to add it in yourself in this method.

        :return: output z of linear layer with shape (output_size,)

  

        HINT: You may want to cache some of the values you compute in this

        function. Inspect your expressions for backprop to see which values

        should be cached.

        '''

        if x.ndim == 1:

            x = np.append(1, x)

        else:

            x = np.append(np.ones((x.shape[0], 1)), x, axis=1)

        z = np.dot(x, self.w.T)

        self.cache['x'] = x

        return z        

  

    def backward(self, dz: np.ndarray) -> np.ndarray:

        '''

        :param dz: partial derivative of loss with respect to output z of linear

        :return: dx, partial derivative of loss with respect to input x of linear

        Note that this function should set self.dw (gradient of weights with respect to loss)

        but not directly modify self.w; NN.step() is responsible for updating the weights.

  

        HINT: You may want to use some of the values you previously cached in

        your forward() method.

        '''

        # TODO: implement this based on your written answers!

        # Hint: when calculating dx, be careful to use the right "version" of

        # the weight matrix!

        dx = np.dot(self.w.T, dz)[1:]

        self.dw = np.dot(dz[:, np.newaxis], self.cache['x'][:, np.newaxis].T)

        return dx

    def step(self) -> None:

        '''

        Apply SGD update to weights using self.dw, which should have been

        set in NN.backward().

        '''

        # TODO: implement this!

        self.w -= self.lr * self.dw

        self.dw = np.zeros_like(self.w, dtype=np.float64)

  

class NN(object):

    def __init__(self, input_size: int, hidden_size: int, output_size: int,

                 weight_init_fn, learning_rate: float):

        '''

        Initalize neural network (NN) class. Note that this class is composed

        of the layer objects (Linear, Sigmoid) defined above.

  

        :param input_size: number of units in input to network

        :param hidden_size: number of units in the hidden layer of the network

        :param output_size: number of units in output of the network - this

                            should be equal to the number of classes

        :param weight_init_fn: function that creates and initializes weight

                               matrices for layer. This function takes in a

                               tuple (row, col) and returns a matrix with

                               shape row x col.

        :param learning_rate: learning rate for SGD training updates

        '''

        self.weight_init_fn = weight_init_fn

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.output_size = output_size

  

        # TODO: initialize modules (see section 9.1.2 of the writeup)

        # Hint: use the classes you've implemented above!

        self.linear1 = Linear(input_size, hidden_size, weight_init_fn, learning_rate)

        self.activation = Sigmoid()

        self.linear2 = Linear(hidden_size, output_size, weight_init_fn, learning_rate)

  

    def forward(self, x: np.ndarray) -> np.ndarray:

        '''

        Neural network forward computation.

        Follow the pseudocode!

        :param X: input data point *without the bias folded in*

        :param nn: neural network class

        :return: output prediction with shape (num_classes,). This should be

                 a valid probability distribution over the classes.

        '''

        z1 = self.linear1.forward(x)

        a1 = self.activation.forward(z1)

        z2 = self.linear2.forward(a1)

        return softmax(z2)

    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> None:

        '''

        Neural network backward computation.

        Follow the pseudocode!

        :param y: label (a number or an array containing a single element)

        :param y_hat: prediction with shape (num_classes,)

        :param nn: neural network class

        '''

        dz2 = d_softmax_cross_entropy(y, y_hat)

        da1 = self.linear2.backward(dz2)

        dz1 = self.activation.backward(da1)

        dx = self.linear1.backward(dz1)

    def step(self) -> None:

        '''

        Apply SGD update to weights.

        '''

        self.linear1.step()

        self.linear2.step()

    def print_weights(self) -> None:

        '''

        An example of how to use logging to print out debugging infos.

  

        Note that we use the debug logging le    vel -- if we use a higher logging

        level, we will log things with the default logging configuration,

        causing potential slowdowns.

  

        Note that we log NumPy matrices on separate lines -- if we do not do this,

        the arrays will be turned into strings even when our logging is set to

        ignore debug, causing potential massive slowdowns.

        '''

        logging.debug(f"shape of w1: {self.linear1.w.shape}")

        logging.debug(self.linear1.w)

        logging.debug(f"shape of w2: {self.linear2.w.shape}")

        logging.debug(self.linear2.w)

  
  

def test(X: np.ndarray, y: np.ndarray, nn: NN):

    '''

    Compute the label and error rate.

    :param X: input data

    :param y: label

    :param nn: neural network class

    :return:

    labels: predicted labels

    error_rate: prediction error rate

    '''

    labels = []

    for x in X:

        y_hat = nn.forward(x)

        labels.append(np.argmax(y_hat))

    labels = np.array(labels)

    error_rate = np.mean(labels != y)

    return labels, error_rate

  
  

def train(X_tr: np.ndarray, y_tr: np.ndarray,

          X_test: np.ndarray, y_test: np.ndarray,

          nn: NN, n_epochs: int):

    '''

    Train the network using SGD for some epochs.

    :param X_tr: train data

    :param y_tr: train label

    :param X_te: train data

    :param y_te: train label

    :param nn: neural network class

    :param n_epochs: number of epochs to train for

    :return:

    train_losses: Training losses *after* each training epoch

    test_losses: Test losses *after* each training epoch

    '''

    # TODO: implement this!

    # Hint: Be sure to shuffle the train data at the start of each epoch

    # using *our provided* shuffle() function.

    train_losses = []

    test_losses = []

    for epoch in range(n_epochs):

        X_train, y_train = shuffle(X_tr, y_tr, epoch)

        for x, y in zip(X_train, y_train):

            y_hat = nn.forward(x)

            nn.backward(y, y_hat)

            nn.step()

        print('alpha:', nn.linear1.w)

        print('beta:', nn.linear2.w)

        train_temp = []

        test_temp = []

        for x, y in zip(X_train, y_train):

            y_hat = nn.forward(x)

            train_temp.append(cross_entropy(y, y_hat))

        for x, y in zip(X_test, y_test):

            y_hat = nn.forward(x)

            test_temp.append(cross_entropy(y, y_hat))

        train_losses.append(np.mean(train_temp))

        test_losses.append(np.mean(test_temp))

    return train_losses, test_losses
```

记录下来主要是学习这种思想，神经网络就是不同的组件一点一点拼接起来的，这点在复现transformer的时候深有感受，因此以后复现论文的话要按照这样的思路来，比如linear、sigmoid，都实现了forward和backword，这样的思想就不会显得特别复杂。特此记录。


