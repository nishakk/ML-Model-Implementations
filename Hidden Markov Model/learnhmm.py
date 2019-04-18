import sys
import numpy as np

tags = []
tagCount = {}
wordsList = []
def load_tags(filename):
    with open(filename, 'rb') as f:
        reader = f.readlines()
        data = list(reader) 
    
    for row in data:
        info = row.split()
        tags.append(info[0])
        tagCount[info[0]] = 0
        
def load_words(filename):
    with open(filename, 'rb') as f:
        reader = f.readlines()
        data = list(reader) 
    
    for row in data:
        info = row.split()
        wordsList.append(info[0])

def load_train(filename):
    with open(filename, 'rb') as f:
        reader = f.readlines()
        data = list(reader) 
    return data
    
def learn_prior(data):
    x = 0
    for row in data:
        if x == 10000:
            break
        x = x + 1
        if row.endswith("\n"):
            row = row[:-1]
        words = row.split(" ")
        w = words[0]
        wtag = w.split("_")
        count = tagCount[wtag[1]]
        count = count + 1
        tagCount[wtag[1]] = count
    
    total = 0
    for tag in tags:
        total = total + tagCount[tag] + 1
    
    pi = []
    for tag in tags:
        val = (tagCount[tag] + 1) / float(total)
        pi.append(val)
        priorTxt.write(str("{:.20E}".format(val)) + "\n")
    return pi

def learn_trans(data):
    x = 0
    trans = np.zeros([len(tags), len(tags)])
    for row in data:
        if x == 10000:
            break
        x = x + 1
        if row.endswith("\n"):
            row = row[:-1]
        words = row.split(" ")
        prev = ""
        i = 0
        for w in words:
            warr = w.split("_")
            if i == 0:
                prev = warr[1]
            if i > 0:
                prevIndex = tags.index(prev)
                currIndex = tags.index(warr[1])
                trans[prevIndex][currIndex] = trans[prevIndex][currIndex] + 1
                prev = warr[1]
            i = i + 1
    
    transTotal = []
    for row in trans:
        total = 0
        for val in row:
            total = total + val + 1
        transTotal.append(total)
    for i in range(len(tags)):
        for j in range(len(tags)):
            trans[i][j] = (trans[i][j] + 1) / transTotal[i]
            
    for i in range(len(trans)):
        for j in range(len(trans[0])):
            transTxt.write(str("{:.20E}".format(trans[i][j])) + " ")
        transTxt.write("\n")
    return trans
    
def learn_emit(data):
    x = 0
    emit = np.zeros([len(tags), len(wordsList)], dtype = float)
    for row in data:
        if x == 10000:
            break
        x = x + 1
        if row.endswith("\n"):
            row = row[:-1]
        wrds = row.split(" ")
        for w in wrds:
            warr = w.split("_")
            wordIndex = wordsList.index(warr[0])
            tagIndex = tags.index(warr[1])
            emit[tagIndex][wordIndex] = emit[tagIndex][wordIndex] + 1
    
    emitTotal = []
    for row in emit:
        total = 0
        for val in row:
            total = total + val + 1
        emitTotal.append(total)
    for i in range(len(emit)):
        for j in range(len(emit[0])):
            emit[i][j] = (emit[i][j] + 1) / emitTotal[i]
            
    for i in range(len(emit)):
        for j in range(len(emit[0])):
            emitTxt.write(str("{:.20E}".format(emit[i][j])) + " ")
        emitTxt.write("\n")
    return emit

load_tags(sys.argv[3])
data = load_train(sys.argv[1])
load_words(sys.argv[2])
priorTxt = open(sys.argv[4], 'w')
emitTxt = open(sys.argv[5], 'w')
transTxt = open(sys.argv[6], 'w')
learn_prior(data)
print tagCount
learn_trans(data)
learn_emit(data)