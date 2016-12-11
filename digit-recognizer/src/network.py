
# coding: utf-8

# # Recognizing Handwritten Digits using Neural Networks
# 
# http://neuralnetworksanddeeplearning.com/chap1.html
# 
# This is a program that will learn how to **_recognize handwritten digits_** using **_stochastic gradient descent_** and the **_MNIST training data_** found [here](https://github.com/mnielsen/neural-networks-and-deep-learning/archive/master.zip).

# ### The Neural Network
# 
# The centerpiece is a 'Network' class, which we use to represent a neural network.

# In[1]:

import numpy as np

class Network(object):
    def __init__ (self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


# **`sizes`** contains the number of neurons in the respective layers. 
# 
# **`biases`** and **`weights`** are randomly initialized. This gives our stochastic gradient descent algorithm a starting place.
# 
# To a create **`Network`** object with 784 neurons in the first layer, 15 neurons in the second layer and 10 neurons in the output layer, we'd do this code: 

# In[2]:

net = Network([784, 15, 10])


# ![img](http://neuralnetworksanddeeplearning.com/images/tikz12.png)

# ### Computing the output of our Network
# 
# With this in mind, it's easy to write code to begin computing the output of the **`Network`** class. Let **`σ`** be the **`sigmoid function`**:
# 
# $$σ(z)≡\frac{1}{1+e^{-z}}\tag{1}\\$$
# 

# In[3]:

def sigmoid(z):
    1.0/(1.0+np.exp(-z))


# In[4]:

net.weights[1][1]


# `net.weights[1]` denotes the weights connecting the second and third layer of the network. Let's call that matrix **`w`** for now. Let **`a`** be the vector of activations of the second layer of neurons and let **`b`** be the vector of biases. Let **`a′`** be the vector of activations of the third layer of neurons:
# 
# $$a′ =σ(wa+b)\tag{2}\\$$
# 
# We then add a `feedforward` method to the `Network` class, which, given an input a for the network, returns the corresponding output. Essentially applying equation **(2)** for each layer.

# In[5]:

class Network(Network):
    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


# We then want a way to apply the `Network`'s `feedforward` method. Let's do this using **[stochastic gradient descent](http://alexminnaar.com/deep-learning-basics-neural-networks-backpropagation-and-stochastic-gradient-descent.html) (SGD)**.
# 
# The idea is to use gradient descent to find the weights w<sub>k</sub> and biases b<sub>l</sub> which minize the cost function such that the output from the network approximates y(x) for all training inputs x. In other words, our "position" now has components w<sub>k</sub> and b<sub>l</sub>, and the gradient vector ∇C has corresponding components ∂C/∂w<sub>k</sub> and ∂C/∂b<sub>l</sub>. Writing out the gradient descent update rule in terms of components, we have
# 
# $$w_k \rightarrow w_k' = w_k-\frac{\eta}{m}
#   \sum_j \frac{\partial C_{X_j}}{\partial w_k}\\$$
# 
# $$ b_l  \rightarrow  b_l' = b_l-\frac{\eta}{m}
#   \sum_j \frac{\partial C_{X_j}}{\partial b_l}\\$$
#   
# By repeatedly applying this update rule we can "roll down the hill", and hopefully find a minimum of the cost function. In other words, this is a rule which can be used to learn in a neural network.

# In[6]:

class Network(Network):
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
            gradient descent.  The "training_data" is a list of tuples
            "(x, y)" representing the training inputs and the desired
            outputs.  The other non-optional parameters are
            self-explanatory.  If "test_data" is provided then the
            network will be evaluated against the test data after each
            epoch, and partial progress printed out.  This is useful for
            tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) ## Will be defined below
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                        j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete.".format(j))


# The above code works as follows. In each epoch, it randomly shuffles the training data then partitions it into mini-batches. For each `mini_batch`, we apply a single step of gradient descent using `self.update_mini_batch(mini_batch, eta)` - which updates the network weights and biases according to a single iteration of gradient descent.

# In[16]:

class Network(Network):
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # Will be defined below
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)) * nw
                       for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


# Most of the work is done by the line: 
# ```python
# delta_nabla_b, delta_nabla_w = self.backprop(x,y)
# ```
# Which invokes a *backpropagation* algorithm - a fast way of computing the gradient of the cost function. So `update_mini_batch` computes these gradients for every training example in `mini_batch` and then updates `self.weights` and `self.biases` appropriately.

# In[17]:

class Network(Network):
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) *             sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

