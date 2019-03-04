import sys
import csv
import math

dictList = {}
def load_dict(filename):
    with open(filename, 'rb') as f:
        reader = f.readlines()
        data = list(reader) 
    
    for row in data:
        info = row.split()
        dictList[info[0]] = info[1]

def load_formattedData(filename):
    with open(filename, 'rb') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        dataset = list(reader)
    return dataset

def getThetaTransposeX(theta, x):
    sum = 0
    for i in range(len(x)):
        if i < (len(x)-1):
            sum = sum + theta[x[i]]
        if i == len(x) - 1:
            sum = sum + theta[len(dictList) - 1]
    value = math.exp(sum) / (1 + math.exp(sum))   
    return value

def learn(dataset, epochs):
    theta = [0] * (len(dictList) + 1)
    for k in range(epochs):
        for row in dataset:
            y = row[0]
            x = []
            for i in range(len(row)):
                if i > 0:
                    values = row[i].split(":")
                    x.append(int(values[0]))
            
            x.append(1)
            val = getThetaTransposeX(theta, x)
            for j in range(len(x)):
                if j < (len(x) - 1):
                    theta[x[j]] = theta[x[j]] + (0.1 * (int(y) - val))
                if j == (len(x) - 1):
                    theta[len(dictList) - 1] = theta[len(dictList) - 1] + (0.1 * x[j] * (int(y) - val))
    return theta
    
def predict(data, theta, dataType):
    for row in data:
        x = []
        
        for i in range(len(row)):
            if i > 0:
                values = row[i].split(":")
                x.append(int(values[0]))
        
        x.append(1)
        
        sum = 0
        for i in range(len(x)):
            if i < len(x) - 1:
                sum = sum + theta[x[i]]
            if i == len(x) - 1:
                sum = sum + theta[len(dictList) - 1]
            
        sigmoid_sum = calc_sigmoid(sum)
        if sigmoid_sum > 0.5:
            if dataType == "train":
                trainLabelsList.append(1)
                trainLabels.write("1\n")
            else:
                testLabelsList.append(1)
                testLabels.write("1\n")
        else:
            if dataType == "train":
                trainLabelsList.append(0)
                trainLabels.write("0\n")
            else:
                testLabelsList.append(0)
                testLabels.write("0\n")
                
def calc_sigmoid(value):
    val = 1 / float((1 + math.exp(-value)))
    return val
                
def calc_metrics(dataset, labels, dataType):
    sum = 0
    i = 0
    for row in dataset:
        y = row[0]
        if int(y) != labels[i]:
            sum = sum + 1
        i = i + 1
        
    error = sum / float(len(labels))
    if dataType == "train":
        metricsFile.write("error(train):" + str(error) + "\n")
    else:
        metricsFile.write("error(test):" + str(error) + "\n")
    
trainDataset = load_formattedData(sys.argv[1])
validDataset = load_formattedData(sys.argv[2])
testDataset = load_formattedData(sys.argv[3])
load_dict(sys.argv[4])
trainLabelsList = []
testLabelsList = []
trainLabels = open(sys.argv[5], 'w')
testLabels = open(sys.argv[6], 'w')
metricsFile = open(sys.argv[7], 'w')
epoch_nums = int(sys.argv[8])
thetaVal = learn(trainDataset, epoch_nums)
predict(trainDataset, thetaVal, "train")
predict(testDataset, thetaVal, "test")
calc_metrics(trainDataset, trainLabelsList, "train")
calc_metrics(testDataset, testLabelsList, "test")