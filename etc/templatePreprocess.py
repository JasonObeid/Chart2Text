import math
import os
import re
# from random import shuffle
# from math import ceil
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_md

import nltk
import pandas as pd
from text_to_num import text2num
# import random
nlp = spacy.load('en_core_web_md')
nltk.download('punkt')

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
    cleanLabel = cleanLabel.replace('%', '').replace('*','')
    return cleanLabel


def cleanAxisValue(value):
    cleanValue = re.sub('\s', '_', value)
    cleanValue = cleanValue.replace('|', '').replace(',', '').replace('%', '').replace('*','')
    return cleanValue


def adjustDataLabel(bool, axis, index):
    if axis == 'x':
        xDataLabels[index] = bool
    elif axis == 'y':
        yDataLabels[index] = bool

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


def checkForMultiplier(axisLabel, nextToken):
    axisMultiplier = 1
    tokenMultiplier = 1
    axisLabels = ' '.join(labelWord.lower() for labelWord in axisLabel.split())
    if 'thousand' in axisLabels or 'thousands' in axisLabels:
        axisMultiplier = 1000
    elif 'million' in axisLabels or 'millions' in axisLabels:
        axisMultiplier = 1000000
    elif 'billion' in axisLabels or 'billions' in axisLabels:
        axisMultiplier = 1000000000
    elif 'trillion' in axisLabels or 'trillions' in axisLabels:
        axisMultiplier = 1000000000000
    if 'thousand' in nextToken or 'thousands' in nextToken:
        tokenMultiplier = 1000
    elif 'million' in nextToken or 'millions' in nextToken:
        tokenMultiplier = 1000000
    elif 'billion' in nextToken or 'billions' in nextToken:
        tokenMultiplier = 1000000000
    elif 'trillion' in nextToken or 'trillions' in nextToken:
        tokenMultiplier = 1000000000000
    conversionRatio = axisMultiplier / tokenMultiplier
    return conversionRatio


def normal_round(n, decimals=0):
    expoN = n * 10 ** decimals
    if abs(expoN) - abs(math.floor(expoN)) < 0.5:
        return math.floor(expoN) / 10 ** decimals
    return math.ceil(expoN) / 10 ** decimals


def numberComparison(token, captionTokens, index, word, axisLabel):
    tokenSignificantDigits = len(str(token).replace('.', ''))
    wordSignificantDigits = len(str(word).replace('.', ''))
    # data usually more specific, therefore divide data to match significant digits of token
    if index < len(captionTokens) - 1:
        nextToken = captionTokens[index + 1]
        multiplier = checkForMultiplier(axisLabel, nextToken.lower())
        # floor100 = int(math.floor(word / 100.0)) * 100
        # ceil100 = int(math.ceil(word / 100.0)) * 100
        newWord = normal_round(word * multiplier)
        newWord1 = normal_round(word * multiplier, 1)
        newWord2 = normal_round(word * multiplier, 2)
        newWord3 = normal_round(word)
        newWord4 = normal_round(word, 1)
        newWord5 = normal_round(word, 2)
        if token == newWord or token == newWord1 or token == newWord2 or \
                token == newWord3 or token == newWord4 or token == newWord5:
            return True
        elif wordSignificantDigits > 3:
            newWord = normal_round(word)
            newWord1 = normal_round(word, 1)
            newWord2 = normal_round(word, 2)
            if token == newWord or token == newWord1 or token == newWord2:
                return True
        else:
            newWord = normal_round(word * multiplier)
            newWord1 = normal_round(word * multiplier, 1)
            newWord2 = normal_round(word * multiplier, 2)
            # print(f'normal: {token}, {word}, {multiplier}, {newToken}')
            if token == newWord or token == newWord1 or token == newWord2:
                return True
    return False


def checkForParallelInSentence(axis, type, arrayIndex):
    if axis.lower() == 'y':
        inverseAxis = 'X'
        parallel = xValueArr[arrayIndex]
    elif axis.lower() == 'x':
        inverseAxis =  'Y'
        parallel = yValueArr[arrayIndex]
    cleanCaption = [token.replace(',','') for token in captionTokens if token not in fillers]
    for token in cleanCaption:
        if token.lower() == parallel:
            tokensNoCommas = [token.replace(',','') if token != ',' else token for token in captionTokens ]
            tokenIndex = tokensNoCommas.index(token)
            #print(f'match in {caption}\n{xValueArr[arrayIndex]} == {token}')
            template = f'template{inverseAxis}Value[idx{type}({axis.upper()})]'
            parallelData.append([template, axis, tokenIndex])


def templateAssigner(token, valueArr, words, arrayIndex, axis):
    if axis.lower() == 'x':
        if xDataType.lower() == 'ordinal':
            if is_number(token) and are_numbers(valueArr):
                if float(words) == max([float(i) for i in valueArr]):
                    checkForParallelInSentence(axis, 'max', arrayIndex)
                    return [1, f'template{axis}Value[max]']
                elif float(words) == min([float(i) for i in valueArr]):
                    checkForParallelInSentence(axis, 'min', arrayIndex)
                    return [1, f'template{axis}Value[min]']
    else:
        if yDataType.lower() == 'numerical':
            if is_number(token) and are_numbers(valueArr):
                if float(words) == max([float(i) for i in valueArr]):
                    checkForParallelInSentence(axis, 'max', arrayIndex)
                    return [1, f'template{axis}Value[max]']
                elif float(words) == min([float(i) for i in valueArr]):
                    checkForParallelInSentence(axis, 'min', arrayIndex)
                    return [1, f'template{axis}Value[min]']
    if words == valueArr[len(valueArr) - 1]:
        return [1, f'template{axis}Value[last]']
    return [1, f'template{axis}Value[{arrayIndex}]']


def compareToken(captionTokens, index, cleanTitle, xValueArr,
                 yValueArr, cleanXAxis, cleanYAxis, entities):
    #check if last token was an un-templated month
    if index > 0:
        lastToken = captionTokens[index - 1]
        if lastToken.lower() in months or lastToken == 'May':
            captionTokens.pop(index - 1)
    token = captionTokens[index].replace(',', '').lower()
    if is_word_number(token):
        token = str(text2num(token, 'en'))
    # iterate through x and y values
    for xWords, yWords, i in zip(xValueArr, yValueArr, range(0, len(xValueArr))):
        # iterate through values with multiple tokens in them, delimited by '_'
        for xWord in xWords.split('_'):
            xWord = xWord.replace(',', '').lower()
            if is_word_number(xWord):
                xWord = str(text2num(xWord, 'en'))
            if token == xWord:
                adjustDataLabel(1, 'x', i)
                return templateAssigner(token, xValueArr, xWords, i, 'X')
            elif is_number(token) and are_numbers(xValueArr):
                if numberComparison(float(token), captionTokens, index, float(xWord), cleanXAxis):
                    adjustDataLabel(1, 'x', i)
                    return templateAssigner(token, xValueArr, xWords, i, 'X')
        for yWord in yWords.split('_'):
            yWord = yWord.replace(',', '').lower()
            if is_word_number(yWord):
                yWord = str(text2num(yWord, 'en'))
            if token == yWord:
                adjustDataLabel(1, 'y', i)
                return templateAssigner(token, yValueArr, yWords, i, 'Y')
            elif is_number(token) and are_numbers(yValueArr):
                if numberComparison(float(token), captionTokens, index, float(yWord), cleanYAxis):
                    adjustDataLabel(1, 'y', i)
                    return templateAssigner(token, yValueArr, yWords, i, 'Y')
    # check if token in axis names
    # remove filler words from labels
    cleanXArr = [xWord for xWord in cleanXAxis.split('_') if xWord.lower() not in fillers]
    cleanYArr = [yWord for yWord in cleanYAxis.split('_') if yWord.lower() not in fillers]
    numbers = ['percent', 'percentage', '%', 'hundred', 'thousand', 'million', 'billion', 'trillion',
               'hundreds', 'thousands', 'millions', 'billions', 'trillions']
    for xLabelToken, i in zip(cleanXArr, range(0, len(cleanXArr))):
        xLabelWord = xLabelToken.replace('_', ' ').lower()
        if str(token).lower() == xLabelWord:
            return [1, f'templateXLabel[{i}]']
        elif str(token).lower() in numbers and xLabelWord.lower() in numbers:
            return [1, f'templateScale']
    for yLabelToken, i in zip(cleanYArr, range(0, len(cleanYArr))):
        yLabelWord = yLabelToken.replace('_', ' ').lower()
        if str(token).lower() == yLabelWord:
            return [1, f'templateYLabel[{i}]']
        elif str(token).lower() in numbers and yLabelWord.lower() in numbers:
            return [1, f'templateScale']
    # check if token in title
    for titleToken, i in zip(cleanTitle, range(0, len(cleanTitle))):
        titleWord = titleToken.lower()
        if str(token).lower() == titleWord:
            for subject, n in zip(entities['Subject'], range(0, len(entities['Subject']))):
                if titleWord == subject.lower():
                    return [1, f'templateTitleSubject[{n}]']
            for date, m in zip(entities['Date'], range(0, len(entities['Date']))):
                if titleWord == str(date).lower():
                    if len(entities['Date']) > 1:
                        #cant check for parallels in title
                        if date == max(entities['Date']):
                            return [1, f'templateTitleDate[max]']
                        elif date == min(entities['Date']):
                            return [1, f'templateTitleDate[min]']
                    return [1, f'templateTitleDate[{m}]']
            return [1, f'templateTitle[{i}]']
    #replace unmatched united states tokens with country to reduce bias
    if index < len(captionTokens) - 1:
        nextToken = captionTokens[index+1]
        if token.lower() == 'united' and nextToken.lower() == 'states':
            if 'U.S.' in cleanTitle:
                usIndex = cleanTitle.index('U.S.')
                captionTokens[index] = f'templateTitle[{usIndex}]'
                captionTokens.pop(index + 1)
                return [1, f'templateTitle[{usIndex}]']
            elif 'American' in cleanTitle:
                usIndex = cleanTitle.index('American')
                captionTokens[index] = f'templateTitle[{usIndex}]'
                captionTokens.pop(index + 1)
                return [1, f'templateTitle[{usIndex}]']
            else:
                captionTokens.pop(index + 1)
                captionTokens[index] = 'country'
                return [0, 'country']
        elif token.lower() == 'u.s.' or token.lower() == 'u.s':
            if 'U.S.' in cleanTitle:
                usIndex = cleanTitle.index('U.S.')
                captionTokens[index] = f'templateTitle[{usIndex}]'
                return [1, f'templateTitle[{usIndex}]']
            elif 'United' in cleanTitle and 'States' in cleanTitle:
                usIndex = cleanTitle.index('States')
                captionTokens[index] = f'templateTitle[{usIndex}]'
                return [1, f'templateTitle[{usIndex}]']
    return [0, token]


dataFiles = os.listdir('../dataset/data')
dataFiles.sort()
# dataFiles = dataFiles[3800:4800]

captionFiles = os.listdir('../dataset/captions')
captionFiles.sort()
# captionFiles = captionFiles[3800:4800]

titleFiles = os.listdir('../dataset/titles')
titleFiles.sort()
# titleFiles = titleFiles[3800:4800]

# shuffle data
# random.seed(10)
# zipped = list(zip(dataFiles, captionFiles, titleFiles))
# random.shuffle(zipped)
# dataFiles, captionFiles, titleFiles = zip(*zipped)

dataArr = []
dataLabelArr = []
summaryArr = []
summaryLabelArr = []
labelList = []
titleArr = []
oldSummaryArr = []

dataRatioArr = []
captionRatioArr = []

assert len(captionFiles) == len(dataFiles) == len(titleFiles)

#may implemented seperately to avoid accidentally ignoring the word rather than month
months = ['january', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'november', 'december']

fillers = ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', ':', '-',
           'to', 'but', 'is', 'of', 'it', 'on', '.', 'at', '(', ')', ',']

for i in range(len(dataFiles)):
    dataPath = '../dataset/data/' + dataFiles[i]
    captionPath = '../dataset/captions/' + captionFiles[i]
    titlePath = '../dataset/titles/' + titleFiles[i]
    caption = openCaption(captionPath)
    title = openCaption(titlePath)
    df, cols, size, xAxis, yAxis, chartType = openData(dataPath)
    cleanXAxis = cleanAxisLabel(xAxis)
    cleanYAxis = cleanAxisLabel(yAxis)
    # if cleanYAxis.split('_') == ['Current', 'year', '(as', 'of', 'January', '25,', '2020)']:
    #    print(dataPath)
    dataLine = ''
    summaryLabelLine = ""
    xValueArr = []
    yValueArr = []
    # iterate through each table row
    for i in range(0, size):
        if chartType == 'line_chart':
            xDataType = "Ordinal"
            yDataType = "Numerical"
        else:
            xDataType = "Nominal"
            yDataType = "Numerical"

        xValue = str(df.at[i, xAxis])
        yValue = str(df.at[i, yAxis])

        cleanXValue = cleanAxisValue(xValue)
        cleanYValue = cleanAxisValue(yValue)

        xValueArr.append(cleanXValue)
        yValueArr.append(cleanYValue)

        xRecord = cleanXAxis + '|' + cleanXValue + '|' + 'x' + '|' + 'line'
        yRecord = cleanYAxis + '|' + cleanYValue + '|' + 'y' + '|' + 'bar'
        dataLine = dataLine + xRecord + ' ' + yRecord + ' '

    captionSentences = caption.split(' . ')
    if len(captionSentences) >= 4:
        trimmedCaption = (' . ').join(captionSentences[0:3]) + ' .\n'
    else:
        trimmedCaption = (' . ').join(captionSentences)
    captionTokens = trimmedCaption.split()

    xDataLabels = [0 for item in range(0, len(xValueArr))]
    yDataLabels = [0 for item in range(0, len(yValueArr))]
    labelMap = []

    captionMatchCount = 0
    doc = nlp(title)
    entities = {}
    entities['Subject'] = []
    entities['Date'] = []
    for X in doc.ents:
        if X.label_ == 'GPE' or X.label_ == 'ORG' or X.label_ == 'NORP' or X.label_ == 'LOC':
            cleanSubject = [word for word in X.text.split() if word.isalpha() and word not in fillers]
            if len(cleanSubject) > 0:
                entities['Subject'].append(' '.join(cleanSubject))
        elif X.label_ == 'DATE':
            if X.text.isnumeric():
                entities['Date'].append(X.text)
    if len(entities['Subject']) == 0:
        uppercaseWords = [word for word in title.split() if word[0].isupper()]
        if len(uppercaseWords) > 1:
            guessedSubject = ' '.join(uppercaseWords[1:])
        else:
            guessedSubject = uppercaseWords[0]
        entities['Subject'].append(guessedSubject)
    cleanTitle = [titleWord for titleWord in title.split() if titleWord.lower() not in fillers]
    parallelData = []
    for token, i in zip(captionTokens, range(0, len(captionTokens))):
        # check for duplicates before token replacement
        if i < len(captionTokens) - 1:
            if captionTokens[i] == captionTokens[i + 1]:
                captionTokens.pop(i + 1)
        if token.lower() not in fillers:
            # find labels for summary tokens, call function to replace token with template
            tokenBool, newToken = compareToken(captionTokens, i, cleanTitle, xValueArr,
                                               yValueArr, cleanXAxis, cleanYAxis, entities)
            if tokenBool == 1:
                captionTokens[i] = newToken
                captionMatchCount += 1
        # check for duplicates after token replacement
        if i > 0:
            if captionTokens[i - 1] == captionTokens[i]:
                captionTokens.pop(i)
        else:
            tokenBool = 0
        labelMap.append(str(tokenBool))
    #replace tokens with their parallel templates if they exist
    # ex: in 2019 sales was 300 million -> in templateXValue[max] sales was templateYValue[idxmax(x)] million
    if len(parallelData) > 0:
        for item in parallelData:
            template = item[0]
            axis = item[1]
            tokenIndex = item[2]
            try:
                labelMap[tokenIndex] = '1'
                captionTokens[tokenIndex] = template
            except IndexError:
                # TODO find out if this means that any time a pop occurs the replacement is misaligned,
                # maybe track the # of pops and subtract that from tokenIndex
                # this happens twice due to popping values and changing length of list
                print('index error')
                tokenIndex = len(labelMap)-1
                labelMap[tokenIndex] = '1'
                captionTokens[tokenIndex] = template

    dataRowPairs = [f'{xLabel} {yLabel}' for xLabel, yLabel in zip(xDataLabels, yDataLabels)]
    dataLabelLine = (' ').join(dataRowPairs)
    assert len(dataLabelLine.split()) == (len(xValueArr) + len(yValueArr))
    dataMatchCount = sum(xDataLabels) + sum(yDataLabels)
    dataRatio = round(dataMatchCount / (len(xValueArr) + len(yValueArr)), 2)
    captionRatio = round(captionMatchCount / len(captionTokens), 2)
    if captionMatchCount >= 1 and dataMatchCount >= 1:
        assert len(xValueArr) == len(yValueArr)
        dataRatioArr.append(dataRatio)
        captionRatioArr.append(captionRatio)
        summaryLabelLine = (' ').join(labelMap)
        assert len(captionTokens) == len(summaryLabelLine.split())
        newCaption = (' ').join(captionTokens)
        oldSummaryArr.append(trimmedCaption)
        labelList.append(labelMap)
        dataArr.append(dataLine)
        dataLabelArr.append(dataLabelLine)
        summaryArr.append(newCaption)
        summaryLabelArr.append(summaryLabelLine)
        titleArr.append(title)

assert len(dataArr) == len(dataLabelArr)
assert len(summaryArr) == len(summaryLabelArr)
assert len(summaryArr) == len(oldSummaryArr)
assert len(titleArr) == len(dataArr)

# TODO REVERT SET SIZES
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

oldTrainSummary = oldSummaryArr[0:trainSize]
oldTestSummary = oldSummaryArr[trainSize:trainSize + testSize]
oldValidSummary = oldSummaryArr[trainSize + testSize:]

with open('../data/train/trainData.txt', mode='wt', encoding='utf8') as myfile0:
    myfile0.writelines("%s\n" % line for line in trainData)
with open('../data/train/trainDataLabel.txt', mode='wt', encoding='utf8') as myfile1:
    myfile1.writelines("%s\n" % line for line in trainDataLabel)

with open('../data/test/testData.txt', mode='wt', encoding='utf8') as myfile2:
    myfile2.writelines("%s\n" % line for line in testData)
with open('../data/test/testDataLabel.txt', mode='wt', encoding='utf8') as myfile3:
    myfile3.writelines("%s\n" % line for line in testDataLabel)

with open('../data/valid/validData.txt', mode='wt', encoding='utf8') as myfile4:
    myfile4.writelines("%s\n" % line for line in validData)
with open('../data/valid/validDataLabel.txt', mode='wt', encoding='utf8') as myfile5:
    myfile5.writelines("%s\n" % line for line in validDataLabel)

with open('../data/train/trainSummary.txt', mode='wt', encoding='utf8') as myfile6:
    myfile6.writelines("%s\n" % line for line in trainSummary)
with open('../data/train/trainSummaryLabel.txt', mode='wt', encoding='utf8') as myfile7:
    myfile7.writelines("%s\n" % line for line in trainSummaryLabel)

with open('../data/test/testSummary.txt', mode='wt', encoding='utf8') as myfile8:
    myfile8.writelines("%s\n" % line for line in testSummary)
with open('../data/test/testSummaryLabel.txt', mode='wt', encoding='utf8') as myfile9:
    myfile9.writelines("%s\n" % line for line in testSummaryLabel)

with open('../data/valid/validSummary.txt', mode='wt', encoding='utf8') as myfile10:
    myfile10.writelines("%s\n" % line for line in validSummary)
with open('../data/valid/validSummaryLabel.txt', mode='wt', encoding='utf8') as myfile11:
    myfile11.writelines("%s\n" % line for line in validSummaryLabel)

with open('../data/dataRatio.txt', mode='wt', encoding='utf8') as myfile12:
    myfile12.write(str(dataRatioArr))
with open('../data/captionRatio.txt', mode='wt', encoding='utf8') as myfile13:
    myfile13.write(str(captionRatioArr))

with open('../data/train/trainTitle.txt', mode='wt', encoding='utf8') as myfile14:
    myfile14.writelines("%s" % line for line in trainTitle)
with open('../data/test/testTitle.txt', mode='wt', encoding='utf8') as myfile15:
    myfile15.writelines("%s" % line for line in testTitle)
with open('../data/valid/validTitle.txt', mode='wt', encoding='utf8') as myfile16:
    myfile16.writelines("%s" % line for line in validTitle)

with open('../data/train/trainOriginalSummary.txt', mode='wt', encoding='utf8') as myfile17:
    myfile17.writelines("%s" % line for line in oldTrainSummary)
with open('../data/test/testOriginalSummary.txt', mode='wt', encoding='utf8') as myfile18:
    myfile18.writelines("%s" % line for line in oldTestSummary)
with open('../data/valid/validOriginalSummary.txt', mode='wt', encoding='utf8') as myfile19:
    myfile19.writelines("%s" % line for line in oldValidSummary)

import matplotlib.pyplot as plt

plt.hist(dataRatioArr, 6)
plt.savefig('../data/data.png')
plt.close('all')
plt.hist(captionRatioArr, 6)
plt.savefig('../data/caption.png')
plt.close('all')
"""
with open('../data/fineTune/data.txt', mode='wt', encoding='utf8') as myfile14, \
        open('../data/fineTune/dataLabel.txt', mode='wt', encoding='utf8') as myfile15, \
        open('../data/fineTune/summary.txt', mode='wt', encoding='utf8') as myfile16, \
        open('../data/fineTune/summaryLabel.txt', mode='wt', encoding='utf8') as myfile17:
    for i in range(0, len(captionRatioArr)):
        if captionRatioArr[i] > 0.35:
            myfile14.writelines(dataArr[i] + "\n")
            myfile15.writelines(dataLabelArr[i] + "\n")
            myfile16.writelines(summaryArr[i])
            myfile17.writelines(summaryLabelArr[i] + "\n")
"""
    # tokenVector = nlp(token)
    # xVector =
    # xSimilarity = tokenVector.similarity()
    # ySimilarity = tokenVector.similarity(nlp(cleanYValue))
    # print(token, cleanXValue, xSimilarity)
    # print(token, cleanYValue, ySimilarity)
    # print(' ')
