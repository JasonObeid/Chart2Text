import os
import re

# from random import shuffle
# from math import ceil
# import spacy
import nltk
import pandas as pd

import scripts.tokenizer as tkn

nltk.download('punkt')
# todo SHUFFLE SETS
dataFiles = os.listdir('dataset/data')
dataFiles.sort()

captionFiles = os.listdir('dataset/captions')
captionFiles.sort()

titleFiles = os.listdir('dataset/titles')
titleFiles.sort()

dataArr = []
dataLabelArr = []
summaryArr = []
summaryLabelArr = []
labelList = []

dataRatioArr = []
captionRatioArr = []
assert len(captionFiles) == len(dataFiles) == len(titleFiles)


def getChartType(x):
    if x.lower() == 'year':
        return 'line_chart'
    else:
        return 'bar_chart'


def openCaption(captionPath):
    with open(captionPath, 'r', encoding='utf-8') as captionFile:
        caption = captionFile.read()
    return caption


def openData(dataPath):
    df = pd.read_csv(dataPath)
    cols = df.columns
    size = df.shape[0]
    xAxis = cols[0]
    yAxis = cols[1]
    chartType = getChartType(xAxis)
    return df, cols, size, xAxis, yAxis, chartType


def cleanAxisLabel(label):
    cleanLabel = re.sub('\s', '_', label)
    cleanLabel = re.sub('\*', '', cleanLabel)
    cleanLabel = re.sub('%', '', cleanLabel)
    return cleanLabel


def cleanAxisValue(value):
    cleanValue = re.sub('\s', '_', value)
    # tokenizedValue = tkn.word_tokenize(value)
    # cleanValue = tkn.detokenize(tokenizedValue)
    cleanValue = re.sub('\*', '', cleanValue)
    cleanValue = re.sub('%', '', cleanValue)
    return cleanValue


def checkToken(value, caption):
    bool = 0
    for word in caption.split():
        if str(value).lower() in word.lower():
            bool = 1
            break
            return bool
    return bool



def compareToken(token, titleWords, xValueArr, yValueArr, cleanXAxis, cleanYAxis):
    # check if token in title
    for word in titleWords:
        if token.lower() in word.replace('_', ' ').lower():
            # print(f'token:{token},  TitleValue:{word}')
            return 1
    # check if token in chart values
    for xWord, yWord in zip(xValueArr, yValueArr):
        if token.lower() in xWord.replace('_', ' ').lower():
            # print(f'token:{token},  xValue:{xWord}')
            return 1
        elif token.lower() in yWord.replace('_', ' ').lower():
            # print(f'token:{token},  yValue:{yWord}')
            return 1
    # check if token in axis names
    if token.lower() in cleanXAxis.replace('_', ' ').lower():
        # print(f'token:{token},  xLabel:{cleanXAxis}')
        return 1
    elif token.lower() in cleanYAxis.replace('_', ' ').lower():
        # print(f'token:{token},  yLabel:{cleanYAxis}')
        return 1
    return 0


# nlp = spacy.load('en_core_web_md')

for i in range(len(dataFiles)):
    dataPath = 'dataset/data/' + dataFiles[i]
    captionPath = 'dataset/captions/' + captionFiles[i]
    titlePath = 'dataset/titles/' + titleFiles[i]
    caption = openCaption(captionPath)
    title = openCaption(titlePath)
    cleanTitle = cleanAxisValue(title)
    df, cols, size, xAxis, yAxis, chartType = openData(dataPath)
    # append chart title to start of data file, set data label for it to always be 1
    dataLabelLine = ""  # "1 "
    cleanXAxis = cleanAxisLabel(xAxis)
    cleanYAxis = cleanAxisLabel(yAxis)
    dataLine = ''  # 'Title|' + cleanTitle + '|' + cleanXAxis + '|' + cleanYAxis + ' '
    summaryLabelLine = ""
    xValueArr = []
    yValueArr = []
    # iterate through each table row
    for i in range(0, size):
        xDataType = "x"
        yDataType = "y"

        xValue = str(df.at[i, xAxis])
        yValue = str(df.at[i, yAxis])

        cleanXValue = cleanAxisValue(xValue)
        cleanYValue = cleanAxisValue(yValue)

        xValueArr.append(cleanXValue)
        yValueArr.append(cleanYValue)

        dataMatchCount = 0
        xbool = checkToken(cleanXValue, caption)
        ybool = checkToken(cleanYValue, caption)
        if (xbool == 1):
            dataMatchCount += 1
        if (ybool == 1):
            dataMatchCount += 1

        dataLabelLine = (dataLabelLine + str(xbool) + ' ' + str(ybool) + ' ')
        xRecord = (cleanXAxis + '|' + cleanXValue + '|' + xDataType + '|' + chartType)
        yRecord = (cleanYAxis + '|' + cleanYValue + '|' + yDataType + '|' + chartType)
        dataLine = dataLine + xRecord + ' ' + yRecord + ' '
    # REGEX split punctuation away from word
    captionTokens = caption.split()
    labelMap = []
    captionMatchCount = 0
    print(' ')
    for token in captionTokens:
        if (token.lower() not in ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', 'to', 'but', 'is', 'of', 'it',
                                  'on', '.', 'at', '(', ')', ',']):
            tokenBool = compareToken(token, title.split(), xValueArr, yValueArr, cleanXAxis, cleanYAxis)
            if tokenBool == 1:
                captionMatchCount += 1
        else:
            tokenBool = 0
        labelMap.append(str(tokenBool))
    if captionMatchCount > 0 and dataMatchCount > 0:
        assert len(xValueArr) == len(yValueArr)
        dataRatio = round(dataMatchCount / len(xValueArr), 2)
        captionRatio = round(captionMatchCount / len(captionTokens), 2)
        dataRatioArr.append(dataRatio)
        captionRatioArr.append(captionRatio)
        print(f' data: {dataRatio}, caption: {captionRatio}')
        print(dataMatchCount, captionMatchCount)
        summaryLabelLine = tkn.detokenize(labelMap)
        assert len(captionTokens) == len(summaryLabelLine.rstrip().split())
        print(title)
        print(xValueArr)
        print(yValueArr)
        print(caption)
        print(summaryLabelLine)
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
testData = dataArr[trainSize:trainSize + testSize]
validData = dataArr[trainSize + testSize:]

trainDataLabel = dataLabelArr[0:trainSize]
testDataLabel = dataLabelArr[trainSize:trainSize + testSize]
validDataLabel = dataLabelArr[trainSize + testSize:]

trainSummary = summaryArr[0:trainSize]
testSummary = summaryArr[trainSize:trainSize + testSize]
validSummary = summaryArr[trainSize + testSize:]

trainSummaryLabel = summaryLabelArr[0:trainSize]
testSummaryLabel = summaryLabelArr[trainSize:trainSize + testSize]
validSummaryLabel = summaryLabelArr[trainSize + testSize:]

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
    myfile6.writelines("%s" % line for line in trainSummary)
with open('data/train/trainSummaryLabel.txt', mode='wt', encoding='utf8') as myfile7:
    myfile7.writelines("%s\n" % line for line in trainSummaryLabel)

with open('data/test/testSummary.txt', mode='wt', encoding='utf8') as myfile8:
    myfile8.writelines("%s" % line for line in testSummary)
with open('data/test/testSummaryLabel.txt', mode='wt', encoding='utf8') as myfile9:
    myfile9.writelines("%s\n" % line for line in testSummaryLabel)

with open('data/valid/validSummary.txt', mode='wt', encoding='utf8') as myfile10:
    myfile10.writelines("%s" % line for line in validSummary)
with open('data/valid/validSummaryLabel.txt', mode='wt', encoding='utf8') as myfile11:
    myfile11.writelines("%s\n" % line for line in validSummaryLabel)

with open('data/dataRatio.txt', mode='wt', encoding='utf8') as myfile12:
    myfile12.write(str(dataRatioArr))
with open('data/captionRatio.txt', mode='wt', encoding='utf8') as myfile13:
    myfile13.write(str(captionRatioArr))

import matplotlib.pyplot as plt
plt.hist(dataRatioArr, 6)
plt.savefig('data/data.png')
plt.close('all')
plt.hist(captionRatioArr, 6)
plt.savefig('data/caption.png')
plt.close('all')

with open('data/fineTune/data.txt', mode='wt', encoding='utf8') as myfile14, \
        open('data/fineTune/dataLabel.txt', mode='wt', encoding='utf8') as myfile15, \
        open('data/fineTune/summary.txt', mode='wt', encoding='utf8') as myfile16, \
        open('data/fineTune/summaryLabel.txt', mode='wt', encoding='utf8') as myfile17:
    for i in range(0, len(captionRatioArr)):
        if captionRatioArr[i] > 0.35:
            myfile14.writelines(dataArr[i] + "\n")
            myfile15.writelines(dataLabelArr[i] + "\n")
            myfile16.writelines(summaryArr[i])
            myfile17.writelines(summaryLabelArr[i] + "\n")

    # tokenVector = nlp(token)
    # xVector =
    # xSimilarity = tokenVector.similarity()
    # ySimilarity = tokenVector.similarity(nlp(cleanYValue))
    # print(token, cleanXValue, xSimilarity)
    # print(token, cleanYValue, ySimilarity)
    # print(' ')
