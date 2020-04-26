import os
import re

# from random import shuffle
# from math import ceil
# import spacy
import nltk
import pandas as pd
from text_to_num import text2num

nltk.download('punkt')
# todo SHUFFLE SETS
dataFiles = os.listdir('dataset/data')
dataFiles.sort()
# dataFiles = dataFiles[500:550]

captionFiles = os.listdir('dataset/captions')
captionFiles.sort()
# captionFiles = captionFiles[500:550]

titleFiles = os.listdir('dataset/titles')
titleFiles.sort()
# titleFiles = titleFiles[500:550]


dataArr = []
dataLabelArr = []
summaryArr = []
summaryLabelArr = []
labelList = []
titleArr = []

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
    cleanValue = cleanValue.replace(',', '')
    return cleanValue


def checkToken(value, caption):
    bool = 0
    for word in caption.split():
        if str(value).lower() in word.lower():
            bool = 1
            return bool
    return bool


def templateAssigner(token, valueArr, words, i, axis):
    if is_number(token) and are_numbers(valueArr):
        if float(words) == max([float(i) for i in valueArr]):
            return [1, f'template{axis}Value[max]']
        elif float(words) == min([float(i) for i in valueArr]):
            return [1, f'template{axis}Value[min]']
    elif words == valueArr[len(valueArr) - 1]:
        return [1, f'template{axis}Value[last]']
    return [1, f'template{axis}Value[{i}]']

def compareToken(captionTokens, index, titleWords, xValueArr, yValueArr, cleanXAxis, cleanYAxis):
    # check if numbers are in thousands, millions, billions, trillions
    # check if token in chart values
    token = captionTokens[index].replace(',', '').lower()
    if is_word_number(token):
        token = str(text2num(token, 'en'))
    #iterate through x and y values
    for xWords, yWords, i in zip(xValueArr, yValueArr, range(0, len(xValueArr))):
        #iterate through values with multiple tokens in them, delimited by '_'
        for xWord in xWords.split('_'):
            xWord = xWord.replace(',', '').lower()
            if is_word_number(xWord):
                xWord = str(text2num(xWord, 'en'))
            if token == xWord:
                return templateAssigner(token, xValueArr, xWords, i, 'X')
            elif is_number(token) and are_numbers(xValueArr):
                if numberComparison(float(token), captionTokens, index, float(xWord), xWords):
                    #print('here')
                    return templateAssigner(token, xValueArr, xWords, i, 'X')
        for yWord in yWords.split('_'):
            yWord = yWord.replace(',', '').lower()
            if is_word_number(yWord):
                yWord = str(text2num(yWord, 'en'))
            if token == yWord:
                return templateAssigner(token, yValueArr, yWords, i, 'Y')
            elif is_number(token) and are_numbers(yValueArr):
                if numberComparison(float(token), captionTokens, index, float(yWord), yWords):
                    #print('here')
                    return templateAssigner(token, yValueArr, yWords, i, 'Y')
    # check if token in axis names
    cleanXArr = cleanXAxis.split('_')
    cleanYArr = cleanYAxis.split('_')
    for xLabelword, i in zip(cleanXArr, range(0, len(cleanXArr))):
        if str(token).lower() in xLabelword.replace('_', ' ').lower():
            # print(f'token:{token},  xLabel:{cleanXAxis}')
            return [1, f'templateXLabel[{i}]']
    for yLabelword, i in zip(cleanYArr, range(0, len(cleanYArr))):
        if str(token).lower() in yLabelword.replace('_', ' ').lower():
            return [1, f'templateYLabel[{i}]']
    # check if token in title
    for word, i in zip(titleWords, range(0, len(titleWords))):
        if str(token).lower() in word.replace('_', ' ').lower():
            # print(f'token:{token},  TitleValue:{word}')
            return [1, f'templateTitle[{i}]']
    #if is_number(token):
        #print(f'no match for number: {token}')
    return [0, token]


def numberComparison(token, captionTokens, index, word, words):
    #try checking for simple round errors first
    #if round(token) == round(word):
    #    print(f'found one: {token}, {word}')
    #    return True
    # if round(token,1) == round(word,1):
        #print(f'found one: {token}, {word}')
    #     return True
    # elif round(token,2) == round(word,2):
        #print(f'found one: {token}, {word}')
    #     return True
    token = float(token)
    tokenSignificantDigits = len(str(token).replace('.',''))
    wordSignificantDigits = len(str(word).replace('.', ''))
    digitsToRound = wordSignificantDigits - tokenSignificantDigits
    roundWords = ['about', 'around', 'roughly']
    # data usually more specific, therefore divide data to match significant digits of token
    if 0 < index < len(captionTokens)-1 :
        priorToken = captionTokens[index - 1]
        nextToken = captionTokens[index + 1]
        multiplier = checkForMultiplier(words, nextToken)
        if (priorToken in roundWords) or (nextToken in roundWords):
            newWord = round(word * multiplier, digitsToRound)
            #print(f'rounded: {token}, {word}, {multiplier}, {newToken}')
        else:
            newWord = round(word * multiplier, 1)
            #print(f'normal: {token}, {word}, {multiplier}, {newToken}')
        if token == newWord:
            #print(token, newWord)
            return True
    return False


def are_numbers(stringList):
    try:
        for value in stringList:
            float(value)
        return True
    except ValueError:
        return False


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def is_word_number(string):
    try:
        text2num(string, 'en')
        return True
    except Exception:
        return False


# nlp = spacy.load('en_core_web_md')

def checkForMultiplier(axisLabel, nextToken):
    axisMultiplier = 1
    tokenMultiplier = 1
    if 'thousand' in axisLabel:
        axisMultiplier = 1000
    elif 'million' in axisLabel:
        axisMultiplier = 1000000
    elif 'billion' in axisLabel:
        axisMultiplier = 1000000000
    elif 'trillion' in axisLabel:
        axisMultiplier = 1000000000000
    if 'thousand' in nextToken:
        tokenMultiplier = 1000
    elif 'million' in nextToken:
        tokenMultiplier = 1000000
    elif 'billion' in nextToken:
        tokenMultiplier = 1000000000
    elif 'trillion' in nextToken:
        tokenMultiplier = 1000000000000
    conversionRatio = axisMultiplier / tokenMultiplier
    return conversionRatio


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

    # xMultiplier = checkForMultiplier(cleanXAxis)
    # yMultiplier = checkForMultiplier(cleanYAxis)
    # print(xMultiplier, yMultiplier)
    # REGEX split punctuation away from word
    captionTokens = caption.split()
    labelMap = []
    captionMatchCount = 0
    # print(' ')
    for token, i in zip(captionTokens, range(0, len(captionTokens))):
        if (token.lower() not in ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', 'to', 'but', 'is', 'of', 'it',
                                  'on', '.', 'at', '(', ')', ',']):
            # MODIFY COMPARETOKEN, CREATE NEW FUNCTIONS TO FIGURE OUT WHAT TO ENCODE IN TEMPLATE PLACED IN TOKEN
            tokenBool, newToken = compareToken(captionTokens, i, title.split(), xValueArr, yValueArr, cleanXAxis,
                                               cleanYAxis)
            if tokenBool == 1:
                captionTokens[i] = newToken
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
        # print(f' data: {dataRatio}, caption: {captionRatio}')
        # print(dataMatchCount, captionMatchCount)
        summaryLabelLine = (' ').join(labelMap)
        assert len(captionTokens) == len(summaryLabelLine.rstrip().split())
        # HERE TOO
        newCaption = (' ').join(captionTokens)
        # print(title)
        # print(xValueArr)
        # print(yValueArr)
        # print(caption)
        # print(summaryLabelLine)
        labelList.append(labelMap)
        dataArr.append(dataLine)
        dataLabelArr.append(dataLabelLine)
        summaryArr.append(newCaption)
        summaryLabelArr.append(summaryLabelLine)
        titleArr.append(title)

assert len(dataArr) == len(dataLabelArr)
assert len(summaryArr) == len(summaryLabelArr)
assert len(titleArr) == len(dataArr)

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

trainTitle = titleArr[0:trainSize]
testTitle = titleArr[trainSize:trainSize + testSize]
validTitle = titleArr[trainSize + testSize:]

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

with open('data/dataRatio.txt', mode='wt', encoding='utf8') as myfile12:
    myfile12.write(str(dataRatioArr))
with open('data/captionRatio.txt', mode='wt', encoding='utf8') as myfile13:
    myfile13.write(str(captionRatioArr))

with open('data/train/trainTitle.txt', mode='wt', encoding='utf8') as myfile14:
    myfile14.writelines("%s" % line for line in trainTitle)
with open('data/test/testTitle.txt', mode='wt', encoding='utf8') as myfile15:
    myfile15.writelines("%s" % line for line in testTitle)
with open('data/valid/validTitle.txt', mode='wt', encoding='utf8') as myfile16:
    myfile16.writelines("%s" % line for line in validTitle)

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
