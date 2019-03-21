import sys
import csv
import numpy as np

# Load training/validation and testing dataset
def load_csv(filename, typeOfData):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        dataset = list(reader)
    
    for row in dataset:
        if typeOfData == "train":
            y_train.append(int(row[0]))
            val = row[1:]
            x_train.append(val)
        else:
            y_test.append(int(row[0]))
            val = row[1:]
            x_test.append(val)
    
    return dataset

# SGD for NN implementation
def sgd(trainingData, testData, epochs, alpha, beta, lRate):
    for e in range(epochs):
        print "Epoch " + str(e)
        for row in trainingData:
            y = int(row[0])
            x = np.array(row)
            x = x.astype(np.float)
            x[0] = 1
            x = np.reshape(x, (len(x), 1))
            o = nn_forward(x, y, alpha, beta)
            gab= nn_backward(x, y, alpha, beta, o)
            galpha = gab['galpha']
            gbeta = gab['gbeta']
            alpha = alpha - (lRate * galpha)
            beta = beta - (lRate * gbeta)
            
        # Training data mean cross entropy calculation
        yhatVals = 0
        for row in trainingData:
            y = int(row[0])
            x = np.array(row)
            x = x.astype(np.float)
            x[0] = 1
            x = np.reshape(x, (len(x), 1))
            o = nn_forward(x, y, alpha, beta)
            yhatVals = yhatVals + np.log(o['yhat'][y])
        mcetrain = -1 * (yhatVals / len(trainingData))
        metricsFile.write("epoch=" + str(e + 1) + " crossentropy(train): " + str(mcetrain[0]) + "\n")
        
        # Test data mean cross entropy calculation
        yhatVals  = 0
        for row in testData:
            y = int(row[0])
            x = np.array(row)
            x = x.astype(np.float)
            x[0] = 1
            x = np.reshape(x, (len(x), 1))
            o = nn_forward(x, y, alpha, beta)
            yhatVals = yhatVals + np.log(o['yhat'][y])
        mcetest = -1 * (yhatVals / len(testData))
        metricsFile.write("epoch=" + str(e + 1) + " crossentropy(test): " + str(mcetest[0]) + "\n")
    return {'alpha': alpha, 'beta': beta}
        
# Feed Forward Implemenation
def nn_forward(x, y, alpha, beta):
    a = linear_forward(x, alpha)
    z = sigmoid_forward(a)
    b = linear_forward(z, beta)
    yhat = softmax_forward(b)
    j = cross_entropy_forward(y, yhat)
    return {'a': a, 'z': z, 'b': b, 'yhat': yhat, 'j': j}

# Linear Combination at the first (hidden) layer
def linear_forward(val1, val2):
    return val2.dot(val1)
    
# Sigmoid activation at the first (hidden) layer
def sigmoid_forward(val):
    z = np.empty([len(val) + 1, 1], dtype=float)
    z[0] = 1
    i = 1
    for v in val:
        sigVal = 1 / (1 + np.exp(-1 * v))
        z[i] = sigVal
        i = i + 1
    return z
    
# Softmax activation at the second (output) layer
def softmax_forward(val):
    return np.exp(val) / np.sum(np.exp(val))

# Cross Entropy Loss
def cross_entropy_forward(y, yhat):
    return -1 * np.log(yhat[y])
    
# Backpropogation Implementation
def nn_backward(x, y, alpha, beta, o):
    gyhat = cross_entropy_backward(y, o['yhat'], o['j'])
    gb = softmax_backward(y, o['yhat'], gyhat)
    gvals = linear_backward(o['z'], beta, o['b'], gb)
    gbeta = gvals['galphabeta']
    gz = gvals['gz']
    ga = sigmoid_backward(o['a'], o['z'], gz)
    ga = np.reshape(ga, (len(ga), 1))
    gvals = linear_backward(x, alpha, o['a'], ga)
    galpha = gvals['galphabeta']
    return {'galpha': galpha, 'gbeta': gbeta}
    
# Derivative of cross entropy loss
def cross_entropy_backward(y, yhat, j):
    gyhat = np.zeros([len(yhat), 1], dtype=float)
    gyhat[y] = -1 / yhat[y]
    return gyhat

# Derivative of softmax function
def softmax_backward(y, yhat, gyhat):
    gb = np.empty([len(yhat), 1])
    i = 0
    for v in yhat:
        if i == y:
            gb[i] = v - 1
        else:
            gb[i] = v
        i = i + 1
    return gb
    
# Derivative of linear combination
def linear_backward(z, beta, b, gb):
    gbeta = gb.dot(z.transpose())
    gz = beta.transpose().dot(gb)
    return { 'galphabeta': gbeta, 'gz': gz}

# Derivative of sigmoid function
def sigmoid_backward(a, z, gz):
    z = np.delete(z, 0)
    gz = np.delete(gz, 0)
    ga = np.multiply(gz, z * (1 - z))
    return ga
    
# Predicting class for test data
def predict(alpha, beta, testData, datatype):
    errorSum = 0
    for row in testData:
        y = int(row[0])
        x = np.array(row)
        x = x.astype(np.float)
        x[0] = 1
        x = np.reshape(x, (len(x), 1))
        o = nn_forward(x, y, alpha, beta)
        yhatPrediction = np.argmax(o['yhat'])
        if datatype == "train":
            train_out.write(str(yhatPrediction) + "\n")
        else: 
            test_out.write(str(yhatPrediction) + "\n")
        if y != yhatPrediction:
            errorSum = errorSum + 1
    errorRate = errorSum / float(len(testData))
    return errorRate

# Load datasets
x_train = []
y_train = []
x_test = []
y_test = []
trainData = load_csv(sys.argv[1], "train")
testData = load_csv(sys.argv[2], "test")

train_out = open(sys.argv[3], 'w')
test_out = open(sys.argv[4], 'w')
metricsFile = open(sys.argv[5], 'w')

typeOfVal = int(sys.argv[8])  # 1 for RANDOM and 2 for ZERO
numHiddenUnits = int(sys.argv[7]) # number of hidden units

# RANDOM will initialize weights randomly from a distribution from -0.1 to 0.1
# ZERO all weights are initialized to 0
if typeOfVal == 1:
    alpha = (0.2) * np.random.random_sample([numHiddenUnits, len(x_train[0])]) - 0.1
    beta = (0.2) * np.random.random_sample([10, numHiddenUnits]) - 0.1
elif typeOfVal == 2:
    alpha = np.zeros([numHiddenUnits, len(x_train[0])], dtype=float)
    beta = np.zeros([10, numHiddenUnits])

# Adding a bias column of zeros to the weights matrix
bias = np.zeros((numHiddenUnits, 1))
alpha = np.hstack((bias, alpha))

bias = np.zeros((10, 1))
beta = np.hstack((bias, beta))
learningRate = float(sys.argv[9])

epochs = int(sys.argv[6]) # number of epochs

# Get updated weights after running feed forward and backpropagation
ab = sgd(trainData, testData, epochs, alpha, beta, learningRate)

# Error rate of prediction in training data
errorVal = predict(ab['alpha'], ab['beta'], trainData, "train")
metricsFile.write("error(train): " + str(errorVal) + "\n")

# Error rate of prediction in test data
errorVal = predict(ab['alpha'], ab['beta'], testData, "test")
metricsFile.write("error(test): " + str(errorVal))