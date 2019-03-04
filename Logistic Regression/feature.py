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
       
def load_tsv(filename):
    with open(filename, 'rb') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        dataset = list(reader)
    return dataset

def formatInfoModel1(data, datasetType):
    for row in data:
        info = row[1].split()
        tsvLine = row[0]
        lineData = []
        for index in range(len(info)):
            if info[index] in dictList:
                if dictList[info[index]] not in lineData:
                    lineData.append(dictList[info[index]])
                    
        for i in range(len(lineData)):
            tsvLine = tsvLine + "\t" + lineData[i] + ":1"
        
        if datasetType == "train":
            formattedTrainDataFile.write(tsvLine)
            formattedTrainDataFile.write('\n')
        elif datasetType == "validation":
            formattedValidDataFile.write(tsvLine)
            formattedValidDataFile.write('\n')
        else:
            formattedTestDataFile.write(tsvLine)
            formattedTestDataFile.write('\n')
        
def formatInfoModel2(data, datasetType):
    for row in data:
        info = row[1].split()
        tsvLine = row[0]
        lineData = {}
        for index in range(len(info)):
            if info[index] in dictList:
                count = 0
                if dictList[info[index]] in lineData:
                    count = lineData[dictList[info[index]]]
                count = count + 1
                lineData[dictList[info[index]]] = count
        
        for key, value in lineData.iteritems():
            if value < 4:
                tsvLine = tsvLine + "\t" + str(key) + ":1"
        
        if datasetType == "train":
            formattedTrainDataFile.write(tsvLine)
            formattedTrainDataFile.write('\n')
        elif datasetType == "validation":
            formattedValidDataFile.write(tsvLine)
            formattedValidDataFile.write('\n')
        else:
            formattedTestDataFile.write(tsvLine)
            formattedTestDataFile.write('\n')
        

formattedTrainDataFile = open(sys.argv[5], 'w')
formattedValidDataFile = open(sys.argv[6], 'w')
formattedTestDataFile = open(sys.argv[7], 'w')
load_dict(sys.argv[4])
traindataset = load_tsv(sys.argv[1])
validationdataset = load_tsv(sys.argv[2])
testdataset = load_tsv(sys.argv[3])
featureFlag = int(sys.argv[8])

if featureFlag == 1:
    formatInfoModel1(traindataset, "train")
    formatInfoModel1(validationdataset, "validation")
    formatInfoModel1(testdataset, "test")
else:
    formatInfoModel2(traindataset, "train")
    formatInfoModel2(validationdataset, "validation")
    formatInfoModel2(testdataset, "test")