import sys
import numpy as np
    
# Forward-Backward Algorithm
def forwardbackward(data, pi, b, a):
    totalWords = 0
    totalCorrects = 0
    logLikelihoods = 0
    for row in data:
        if row.endswith("\n"):
            row = row[:-1]
        words = row.split(" ")
        alpha = np.zeros([len(words), len(tags)])
        beta = np.zeros([len(words), len(tags)])
        alpha = forward_recursion(alpha, pi, a, b, words, len(words) - 1)
        beta = backward_recursion(beta, pi, a, b, words, 0)
        totalWords = totalWords + len(words)
        totalCorrects = totalCorrects + min_bayes_risk_prediction(alpha, beta, words)
        logLikelihoods = logLikelihoods + log_likelihood(alpha, words)
    metricsTxt.write("Average Log-Likelihood: " + str(logLikelihoods / float(len(data))) + "\n")
    metricsTxt.write("Accuracy: " + str(totalCorrects / float(totalWords)))
        
# Recursive DP implementation of Forward Algorithm
def forward_recursion(alpha, pi, a, b, seq, t):
    if t == 0:
        for i in range(len(tags)):
            alpha[t][i] = log_multiplication(pi[i], b[i][wordsList.index(seq[t].split("_")[0])])
    else:
        alpha = forward_recursion(alpha, pi, a, b, seq, t-1)
        for i in range(len(tags)):
            sum = np.Infinity
            for j in range(len(tags)):
                sum = log_addition(sum, log_multiplication(alpha[t - 1][j], a[j][i]))
            alpha[t][i] = log_multiplication(b[i][wordsList.index(seq[t].split("_")[0])], sum)
    return alpha

# Recursive DP implementation of Backward Algorithm
def backward_recursion(beta, pi, a, b, seq, t):
    if t == len(seq) - 1:
        for i in range(len(tags)):
            beta[t][i] = np.log(1)
    else:
        beta = backward_recursion(beta, pi, a, b, seq, t + 1)
        for i in range(len(tags)):
            sum = np.Infinity
            for k in range(len(tags)):
                sum = log_addition(sum, log_multiplication(b[k][wordsList.index(seq[t+1].split("_")[0])], log_multiplication(beta[t + 1][k], a[i][k])))
            beta[t][i] = sum
    return beta
    
# Minimum Bayes Risk Prediction Implementation
def min_bayes_risk_prediction(alpha, beta, seq):
    corrects = 0
    for i in range(len(alpha)):
        yhat = np.zeros([len(alpha[0]), 1])
        for j in range(len(alpha[i])):
            yhat[j] = log_multiplication(alpha[i][j], beta[i][j])
        ind = np.argmax(yhat)
        if tags[ind] == seq[i].split("_")[1]:
            corrects = corrects + 1
        if i == len(alpha) - 1:
            predictedTxt.write(seq[i].split("_")[0] + "_" + tags[ind])
        else:
            predictedTxt.write(seq[i].split("_")[0] + "_" + tags[ind] + " ")
    predictedTxt.write("\n")
    return corrects
    
# Calculating Log-likelihood
def log_likelihood(alpha, seq):
    sum = np.Infinity
    for i in range(len(alpha[len(seq) - 1])):
        sum = log_addition(sum, alpha[len(seq) - 1][i])
    return sum

# Functions supporting compution in log-space
# Log Multiplication
def log_multiplication(a, b):
    return a + b

# Log Addition
def log_addition(a, b):
    if np.isinf(a):
        return b
    if np.isinf(b):
        return a
    m = np.maximum(a, b)
    return m + np.log(1 + np.exp(-np.abs(a - b)))

# Loading initial probabilities learned in learnhmm.py
def load_prior(filename):
    with open(filename, 'rb') as f:
        reader = f.readlines()
        data = list(reader)
    return np.log(np.array(data, dtype = float))
    
def load_trans(filename):
    with open(filename, 'rb') as f:
        reader = f.readlines()
        data = list(reader)
    
    trans = np.zeros([len(tags), len(tags)], dtype = float)
    i = 0
    for row in data:
        info = row.split()
        info = np.array(info, dtype = float)
        trans[i] = np.log(info)
        i = i + 1
    return trans
    
def load_emit(filename):
    with open(filename, 'rb') as f:
        reader = f.readlines()
        data = list(reader)
        
    emit = np.zeros([len(tags), len(wordsList)], dtype = float)
    i = 0
    for row in data:
        info = row.split()
        info = np.array(info, dtype = float)
        emit[i] = np.log(info)
        i = i + 1
    return emit
    
# Loading initial data
tags = []
wordsList = []
def load_tags(filename):
    with open(filename, 'rb') as f:
        reader = f.readlines()
        data = list(reader) 
    
    for row in data:
        info = row.split()
        tags.append(info[0])
        
def load_words(filename):
    with open(filename, 'rb') as f:
        reader = f.readlines()
        data = list(reader) 
    
    for row in data:
        info = row.split()
        wordsList.append(info[0])

def load_dataset(filename):
    with open(filename, 'rb') as f:
        reader = f.readlines()
        data = list(reader) 
    return data

data = load_dataset(sys.argv[1])
load_words(sys.argv[2])
load_tags(sys.argv[3])
prior = load_prior(sys.argv[4])
emit = load_emit(sys.argv[5])
trans = load_trans(sys.argv[6])
predictedTxt = open(sys.argv[7], 'w')
metricsTxt = open(sys.argv[8], 'w')
forwardbackward(data, prior, emit, trans)