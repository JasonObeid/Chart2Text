import os
import pandas as pd
import re
from scripts.tokenizer import word_tokenize, detokenize
from math import ceil
import nltk
nltk.download('punkt')
#todo SHUFFLE SETS
dataFiles = os.listdir('dataset/data')
dataFiles.sort()

captionFiles = os.listdir('dataset/captions')
captionFiles.sort()

dataArr = []
dataLabelArr = []
summaryArr = []
summaryLabelArr = []
labelList = []
assert len(captionFiles) == len(dataFiles)

def getChartType(x):
    if x.lower() == 'year':
        return 'line_chart'
    else:
        return 'bar_chart'


for i in range(len(dataFiles)):
    dataPath = 'dataset/data/'+ dataFiles[i]
    captionPath = 'dataset/captions/' + captionFiles[i]
    with open(captionPath, 'r', encoding='utf-8') as captionFile:
        caption = captionFile.read()
    df = pd.read_csv(dataPath)
    cols = df.columns
    size = df.shape[0]
    xAxis = cols[0]
    yAxis = cols[1]
    chartType = getChartType(xAxis)
    dataLabelLine = ""
    dataLine = ""
    summaryLabelLine = ""
    xValueArr = []
    yValueArr = []
    for i in range(0,size):
        #cleanXAxis = re.sub('[^0-9a-zA-Z] +', "", xAxis)
        cleanXAxis = re.sub('\s', '_', xAxis)
        #cleanYAxis = re.sub('[^0-9a-zA-Z] +', "", yAxis)
        cleanYAxis = re.sub('\s', '_', yAxis)
        xDataType = "x"
        yDataType = "y"
        xValue = str(df.at[i,xAxis])
        yValue = str(df.at[i,yAxis])
        #cleanXValue = re.sub('[^0-9a-zA-Z] +', "", xValue)
        cleanXValue = re.sub('\s', '_', xValue)
        #cleanYValue = re.sub('[^0-9a-zA-Z] +', "", yValue)
        cleanYValue = re.sub('\s', '_', yValue)
        #cleanXAxis = xAxis
        #cleanYAxis = yAxis
        #cleanXValue = xValue
        #cleanYValue = yValue
        xValueArr.append(cleanXValue)
        yValueArr.append(cleanYValue)
        dataMatchCount = 0
        if(cleanXValue in caption):
            xbool = 1
            dataMatchCount += 1
        else:
            xbool = 0
        if(cleanYValue in caption):
            ybool = 1
            dataMatchCount += 1
        else:
            ybool = 0
        dataLabelLine = (dataLabelLine + str(xbool) + ' ' + str(ybool) + ' ')
        xRecord = (cleanXAxis + '|' + cleanXValue + '|' + xDataType + '|' + chartType)
        yRecord = (cleanYAxis + '|' + cleanYValue + '|' + yDataType + '|' + chartType)
        dataLine = dataLine + xRecord + ' ' + yRecord + ' '
    captionTokens = caption.rstrip().split()
    labelMap = []
    captionMatchCount = 0
    for token in captionTokens:
        if token in (xValueArr or yValueArr):
            tokenBool = 1
            captionMatchCount += 1
        else:
            tokenBool = 0
        labelMap.append(str(tokenBool))
    if captionMatchCount > 0 and dataMatchCount > 0:
        print(captionMatchCount, dataMatchCount)
        summaryLabelLine = detokenize(labelMap)
        assert len(captionTokens) == len(summaryLabelLine.rstrip().split())
        labelList.append(labelMap)
        dataArr.append(dataLine)
        dataLabelArr.append(dataLabelLine)
        summaryArr.append(caption)
        summaryLabelArr.append(summaryLabelLine)

assert len(dataArr) == len(dataLabelArr)
assert len(summaryArr) == len(summaryLabelArr)

trainSize = round(len(dataArr) * 0.7)
testSize = round(len(dataArr) * 0.15)
validSize = len(dataArr) - trainSize - testSize

trainData = dataArr[0:trainSize]
testData = dataArr[trainSize:trainSize+testSize]
validData = dataArr[trainSize+testSize:]

trainDataLabel = dataLabelArr[0:trainSize]
testDataLabel = dataLabelArr[trainSize:trainSize+testSize]
validDataLabel = dataLabelArr[trainSize+testSize:]

trainSummary = summaryArr[0:trainSize]
testSummary = summaryArr[trainSize:trainSize+testSize]
validSummary = summaryArr[trainSize+testSize:]

trainSummaryLabel = summaryLabelArr[0:trainSize]
testSummaryLabel = summaryLabelArr[trainSize:trainSize+testSize]
validSummaryLabel = summaryLabelArr[trainSize+testSize:]


with open('data/train/trainData.txt', mode='wt', encoding='utf8') as myfile0:
     myfile0.writelines("%s\n" % line for line in trainData)
with open('data/train/trainDataLabel.txt', mode='wt', encoding='utf8') as myfile1:
     myfile1.writelines("%s\n" % line for line in trainDataLabel)
     
with open('data/test/testData.txt', mode='wt', encoding='utf8') as myfile2:
    myfile2.writelines("%s\n" % line for line in testData)
with open('data/test/testDataLabel.txt', mode='wt', encoding='utf8') as myfile3:
    myfile3.writelines("%s\n" % line for line in testDataLabel)
                            
with open('data/valid/validData.txt', mode='wt', encoding='utf8') as myfile4:
    myfile4.writelines("%s\n" % line for line in validData)
with open('data/valid/validDataLabel.txt', mode='wt', encoding='utf8') as myfile5:
    myfile5.writelines("%s\n" % line for line in validDataLabel)


with open('data/train/trainSummary.txt', mode='wt', encoding='utf8') as myfile6:
    myfile6.writelines("%s\n" % line for line in trainSummary)
with open('data/train/trainSummaryLabel.txt', mode='wt', encoding='utf8') as myfile7:
    myfile7.writelines("%s\n" % line for line in trainSummaryLabel)

with open('data/test/testSummary.txt', mode='wt', encoding='utf8') as myfile8:
    myfile8.writelines("%s\n" % line for line in testSummary)
with open('data/test/testSummaryLabel.txt', mode='wt', encoding='utf8') as myfile9:
    myfile9.writelines("%s\n" % line for line in testSummaryLabel)

with open('data/valid/validSummary.txt', mode='wt', encoding='utf8') as myfile10:
    myfile10.writelines("%s\n" % line for line in validSummary)
with open('data/valid/validSummaryLabel.txt', mode='wt', encoding='utf8') as myfile11:
    myfile11.writelines("%s\n" % line for line in validSummaryLabel)