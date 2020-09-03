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
dataFiles = os.listdir('../data_testing/dataset/data')
dataFiles.sort()
# dataFiles = dataFiles[500:550]

captionFiles = os.listdir('../data_testing/dataset/captions')
captionFiles.sort()
# captionFiles = captionFiles[500:550]

titleFiles = os.listdir('../data_testing/dataset/titles')
titleFiles.sort()
# titleFiles = titleFiles[500:550]


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
    dataPath = '../data_testing/dataset/data/' + dataFiles[i]
    captionPath = '../data_testing/dataset/captions/' + captionFiles[i]
    titlePath = '../data_testing/dataset/titles/' + titleFiles[i]
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
    oldSummaryArr.append(caption)
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

trainData = dataArr

trainDataLabel = dataLabelArr
trainSummary = summaryArr

trainSummaryLabel = summaryLabelArr

trainTitle = titleArr

oldTrainSummary = oldSummaryArr

with open('../data_testing/trainData.txt', mode='wt', encoding='utf8') as myfile0:
    myfile0.writelines("%s\n" % line for line in trainData)
with open('../data_testing/trainDataLabel.txt', mode='wt', encoding='utf8') as myfile1:
    myfile1.writelines("%s\n" % line for line in trainDataLabel)

with open('../data_testing/trainSummary.txt', mode='wt', encoding='utf8') as myfile6:
    myfile6.writelines("%s\n" % line for line in trainSummary)
with open('../data_testing/trainSummaryLabel.txt', mode='wt', encoding='utf8') as myfile7:
    myfile7.writelines("%s\n" % line for line in trainSummaryLabel)

with open('../data_testing/dataRatio.txt', mode='wt', encoding='utf8') as myfile12:
    myfile12.write(str(dataRatioArr))
with open('../data_testing/captionRatio.txt', mode='wt', encoding='utf8') as myfile13:
    myfile13.write(str(captionRatioArr))

with open('../data_testing/trainTitle.txt', mode='wt', encoding='utf8') as myfile14:
    myfile14.writelines("%s\n" % line for line in trainTitle)

with open('../data_testing/trainOriginalSummary.txt', mode='wt', encoding='utf8') as myfile17:
    myfile17.writelines("%s\n" % line for line in oldTrainSummary)

    # tokenVector = nlp(token)
    # xVector =
    # xSimilarity = tokenVector.similarity()
    # ySimilarity = tokenVector.similarity(nlp(cleanYValue))
    # print(token, cleanXValue, xSimilarity)
    # print(token, cleanYValue, ySimilarity)
    # print(' ')
