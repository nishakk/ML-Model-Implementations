import sys
import csv
import math

attributes = []
outputVals = []

# Method to load csv file.
def load_csv(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        dataset = list(reader)
    attributes.append(dataset.pop(0))
    return dataset
    
# Method to calculate entropy for the given dataset.
def entropy_calc_dataset(data):
    outputs = {}
    for row in data:
        classVal = row[len(row) - 1]
        count = 0
        if classVal in outputs:
            count = outputs[classVal]
        count = count + 1
        outputs[classVal] = count
        
    entropy = 0
    for key, value in outputs.iteritems():
        prob = (value / float(len(data)))
        if prob == 0:
            entropy = entropy
        else:
            entropy = entropy - (prob * math.log(prob, 2))
    
    entropy_dataset = entropy
    return entropy

# Method to calculate entropy for the attribute at the given index in the dataset.
def entropy_calc_attribute(data, index, classVal):
    attributeMap = {}
    totalVals = 0
    for row in data:
        if row[index] == classVal:
            count = 0
            totalVals = totalVals + 1
            if row[len(row) - 1] in attributeMap:
                count = attributeMap[row[len(row) - 1]]
            count = count + 1
            attributeMap[row[len(row) - 1]] = count
    entropy = 0
    for key, value in attributeMap.iteritems():
        prob = (value / float(totalVals))
        if prob == 0:
            entropy = entropy
        else:
            entropy = entropy - (prob * math.log(prob, 2))
    
    return entropy

# Method to calculate mutual information for the attribute at the given index.
def mutual_info_calc(data, index):
    mutual_info = entropy_calc_dataset(data)
    attributeVals = {}
    totalVals = 0
    for row in data:
        count = 0
        if row[index] in attributeVals:
            count = attributeVals[row[index]]
        count = count + 1
        attributeVals[row[index]] = count

    for key, value in attributeVals.iteritems():
        prob = value / float(len(data))
        if prob == 0:
            mutual_info = mutual_info
        else:
            mutual_info = mutual_info - (prob * entropy_calc_attribute(data, index, key))
        
    return mutual_info
    
# Method to find the best attribute for splitting the given data.
# Attribute with the highest mutual information is selected for splitting.
# If the highest mutual information is less than 0, data will not be split.
def bestSplitNodeIndex(data):
    mutualInfoMap = {}
    maxMutualInfo = 0
    bestNodeForSplit = 0
    for i in range(len(attributes[0]) - 1):
        mutualInfoVal = mutual_info_calc(data, i)
        if i == 0: 
            maxMutualInfo = mutualInfoVal
            bestNodeForSplit = i
        elif maxMutualInfo < mutualInfoVal:
            maxMutualInfo = mutualInfoVal
            bestNodeForSplit = i
    
    if maxMutualInfo < 0:
        return -1
    return bestNodeForSplit
    
# Method to split the data at the attribute into left and right child nodes.
def split_data(data, index):
    values = list(set(row[index] for row in data))
    left, right = list(), list()
    for row in data:
        if row[index] == values[0]:
            left.append(row)
        else:
            right.append(row)
    leftVal = values[0]
    if len(values) == 2:
        rightVal = values[1]
    else:
        if results[0] == leftVal:
            rightVal = results[1]
        else:
            rightVal = results[0]
    groups = []
    groups.append(left)
    groups.append(right)
    return {'groups': groups, 'leftValue': leftVal, 'rightValue': rightVal}

# Method to create the leaf node.
def terminalNode(group):
    outcomes = [row[-1] for row in group]
    outcomeCount = {}
    for val in outcomes:
        count = 0
        if val in outcomeCount:
            count = outcomeCount[val]
        count = count + 1
        outcomeCount[val] = count
        
    return outcomeCount
    
# Get split at the attribute with highest mutual information value.
def get_split(data):
    index = bestSplitNodeIndex(data)
    info = split_data(data, index)
    return {'index': index, 'leftValue': info['leftValue'], 'rightValue': info['rightValue'],'groups': info['groups']}
   
# Method to recurse and split and build the tree. 
def split(node, max_depth, depth):
    left, right = node['groups'];
    if not left or not right:
        node['left'] = node['right'] = terminalNode(left + right)
        return
    
    if depth >= max_depth:
        node['left'], node['right'] = terminalNode(left), terminalNode(right)
        frontLines = ""
        for i in range(depth):
            frontLines = frontLines + "| "
        if len(node['left']) < 2:
            if results[0] in node['left']:
                pass
            else:
                node['left'][results[0]] = 0
            if results[1] in node['left']:
                pass
            else:
                node['left'][results[1]] = 0
                
        if len(node['right']) < 2:
            if results[0] in node['right']:
                pass
            else:
                node['right'][results[0]] = 0
            if results[1] in node['right']:
                pass
            else:
                node['right'][results[1]] = 0
        
        print("%s%s = %s: [%s %s /%s %s]" % (frontLines, attributes[0][node['index']], node['leftValue'], node['left'][results[1]], results[1], node['left'][results[0]], results[0]))
        
        print("%s%s = %s: [%s %s /%s %s]" % (frontLines, attributes[0][node['index']], node['rightValue'], node['right'][results[1]], results[1], node['right'][results[0]], results[0]))
        return
    
    frontLines = ""
    for i in range(depth):
        frontLines = frontLines + "| "
    info = split_data(left, len(left[0]) - 1)
    print("%s%s = %s: [%s %s /%s %s]" % (frontLines, attributes[0][node['index']], node['leftValue'], len(info['groups'][1]), info['rightValue'], len(info['groups'][0]), info['leftValue']))

    leftIndex = bestSplitNodeIndex(left)
    if leftIndex == -1:
        node['left'] = terminalNode(left)
        return
    
    node['left'] = get_split(left)
    split(node['left'], max_depth, depth + 1)
    
    info = split_data(right, len(right[0]) - 1)
    print("%s%s = %s: [%s %s /%s %s]" % (frontLines, attributes[0][node['index']], node['rightValue'], len(info['groups'][1]), info['rightValue'], len(info['groups'][0]), info['leftValue']))
    
    rightIndex = bestSplitNodeIndex(right)
    if rightIndex == -1:
        node['right'] = terminalNode(right)
        return
    
    node['right'] = get_split(right)
    split(node['right'], max_depth, depth + 1)
    
    
# Method to build the decision tree.
def build_tree(data, max_depth):
    info =  split_data(data, len(data[0]) - 1)
    outputs = info['groups']
    o1len = len(outputs[0])
    o1 = outputs[0][0][len(outputs[0][0]) - 1]
    o2len = len(outputs[1])
    o2 = outputs[1][0][len(outputs[1][0]) - 1]
    print("[%s %s /%s %s]" % (o2len, o2, o1len, o1))
    
    root = get_split(data)
    if root['index'] == -1:
        root = terminalNode(data)
        return root
    
    if max_depth > 0:
        split(root, max_depth, 1)
    return root
    
# Method to get prediction for the given dataset.
def prediction(testdataset, node, datatype):
    testPredictions = []
    for row in testdataset:
        testPredictions.append(predict(node, row))
        
    errors = 0
    index = 0
    for row in testdataset:
        if datatype == "test":
            # testPredictionFile.write(testPredictions[index])
            if index != (len(testdataset) - 1):
                testPredictionFile.write("\n")
        if datatype == "train":
            trainPredictionFile.write(testPredictions[index])
            if index != (len(testdataset) - 1):
                trainPredictionFile.write("\n")
        if row[len(row) - 1] != testPredictions[index]:
            errors = errors + 1
        index = index + 1
    
    if datatype == "train":
        print "Train error"
        print (errors / float(len(testdataset)))
        metricsFile.write("error(train): " + str(errors / float(len(testdataset))) + "\n")
    else:
        print "Test error"
        print (errors / float(len(testdataset)))
        metricsFile.write("error(test): " + str(errors / float(len(testdataset))))

# Helper method to predict recursively.
def predict(node, row):
    if row[node['index']] == node['leftValue']:
        if 'leftValue' in node['left']:
            return predict(node['left'], row)
        else:
            pred = max(node['left'], key=node['left'].get)
            # testPredictionFile.write(pred + "\n")
            return pred
    elif row[node['index']] == node['rightValue']:
        if 'rightValue' in node['right']:
            return predict(node['right'], row)
        else:
            pred = max(node['right'], key=node['right'].get)
            # testPredictionFile.write(pred + "\n")
            return pred
    else:
        print node
    
# Load training data.
dataset = load_csv(sys.argv[1])

# Get the output labels.
outputVals = split_data(dataset, len(dataset[0])-1)

# Save the two labels
results = []
results.append(outputVals['leftValue'])
results.append(outputVals['rightValue'])

# Write predictions for training data to a file.
trainPredictionFile = open(sys.argv[4], 'w')

# Write predictions for testing data to a file.
testPredictionFile = open(sys.argv[5], 'w')

# Write train error and test errors to a file.
metricsFile = open(sys.argv[6], 'w')

# Method to build the tree and get the root node.
tree = build_tree(dataset, int(sys.argv[3]))

# Load test data.
testdata = load_csv(sys.argv[2])

# Get predictions for training data.
prediction(dataset, tree, "train")

# Get predictions for test data
prediction(testdata, tree, "test")