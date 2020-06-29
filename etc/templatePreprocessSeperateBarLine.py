import math
import os
import re
# from random import shuffle
# from math import ceil
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_md

import matplotlib.pyplot as plt

import nltk
import pandas as pd
from text_to_num import text2num
# import random
nlp = spacy.load('en_core_web_md')
nltk.download('punkt')

def mapParallelIndex(valueArr, type):
    if are_numbers(valueArr):
        try:
            array = [float(i) for i in valueArr]
            if type == 'max':
                index = array.index(max(array))
                return int(index)
            elif type == 'min':
                index = array.index(min(array))
                return int(index)
        except:
            print('Parallel num err')
            print(valueArr, type)
            return 0


def mapIndex(index, array):
    if are_numbers(array):
        try:
            array = [float(i) for i in array]
            if str(index) == 'max':
                index = array.index(max(array))
                return int(index)
            elif str(index) == 'min':
                index = array.index(min(array))
                return int(index)
        except:
            print('numbers num err')
            return 0
    elif are_numbers(array[0:len(array) - 1]):
        try:
            array = [float(i) for i in array[0:len(array) - 1]]
            if str(index) == 'max':
                index = array.index(max(array))
                return int(index)
            elif str(index) == 'min':
                index = array.index(min(array))
                return int(index)
        except:
            print('n-1 num err')
            return 0
    if index == 'last':
        index = len(array) - 1
        return int(index)
    try:
        # this exception occurs with min/max on data which isn't purely numeric: ex. ['10_miles_or_less', '11_-_50_miles', '51_-_100_miles']
        cleanArr = [float("".join(filter(str.isdigit, item))) for item in array if
                    "".join(filter(str.isdigit, item)) != '']
        if str(index) == 'max':
            index = cleanArr.index(max(cleanArr))
            return int(index)
        elif str(index) == 'min':
            index = cleanArr.index(min(cleanArr))
            return int(index)
        return int(index)
    except:
        if not are_numbers(array) and (index == 'min' or index == 'max'):
            return 0
        return int(index)


def getTemplateValues(xCount, yCount, xValueArr, yValueArr):
    values = []
    indices = []
    for template in xCount:
        if 'idxmin' in template or 'idxmax' in template:
            idxType = template[-7:-4]
            if 'templateYValue' in template:
                index = mapParallelIndex(xValueArr, idxType)
                try:
                    values.append(yValueArr[index].replace('_', ' '))
                    indices.append(index)
                except:
                    print(f'{idxType} error at {index} in {title}')
                    values.append(yValueArr[len(yValueArr) - 1].replace('_', ' '))
                    indices.append(len(yValueArr) - 1)
            elif 'templateXValue' in template:
                index = mapParallelIndex(yValueArr, idxType)
                try:
                    values.append(yValueArr[index].replace('_', ' '))
                    indices.append(index)
                except:
                    print(f'{type} error at {index} in {title}')
                    values.append(yValueArr[len(yValueArr) - 1].replace('_', ' '))
                    indices.append(len(yValueArr) - 1)
        else:
            index = str(re.search(r"\[(\w+)\]", template).group(0)).replace('[', '').replace(']', '')
            if 'templateXValue' in template:
                index = mapIndex(index, xValueArr)
                if index < len(xValueArr):
                    values.append(yValueArr[index].replace('_', ' '))
                    indices.append(index)
                else:
                    print(f'xvalue index error at {index} in {title}')
                    values.append(yValueArr[len(yValueArr) - 1].replace('_', ' '))
                    indices.append(len(yValueArr) - 1)
            elif 'templateYValue' in token:
                index = mapIndex(index, yValueArr)
                if index < len(yValueArr):
                    values.append(yValueArr[index].replace('_', ' '))
                    indices.append(index)
                else:
                    print(f'yvalue index error at {index} in {title}')
                    values.append(yValueArr[len(yValueArr) - 1].replace('_', ' '))
                    indices.append(len(yValueArr) - 1)
    return values, indices

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
    for xLabelToken, i in zip(cleanXArr, range(0, len(cleanXArr))):
        xLabelWord = xLabelToken.replace('_', ' ').lower()
        if str(token).lower() == xLabelWord:
            return [1, f'templateXLabel[{i}]']
        elif str(token).lower() in numbers: #and xLabelWord.lower() in numbers:
            return [1, f'templateScale']
    for yLabelToken, i in zip(cleanYArr, range(0, len(cleanYArr))):
        yLabelWord = yLabelToken.replace('_', ' ').lower()
        if str(token).lower() == yLabelWord:
            return [1, f'templateYLabel[{i}]']
        elif str(token).lower() in numbers: #and yLabelWord.lower() in numbers:
            return [1, f'templateScale']
    # check if token in title
    for titleToken, i in zip(cleanTitle, range(0, len(cleanTitle))):
        titleWord = titleToken.lower()
        if str(token).lower() == titleWord:
            for subject, n in zip(entities['Subject'], range(0, len(entities['Subject']))):
                if titleWord in subject.lower():
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
barDataRatioArr = []
barCaptionRatioArr = []
barOldSummaryArr = []
barLabelList = []
barDataArr = []
barDataLabelArr = []
barSummaryArr = []
barSummaryLabelArr = []
barTitleArr = []

lineDataRatioArr = []
lineCaptionRatioArr = []
lineOldSummaryArr = []
lineLabelList = []
lineDataArr = []
lineDataLabelArr = []
lineSummaryArr = []
lineSummaryLabelArr = []
lineTitleArr = []

barDataRatioArr = []
barCaptionRatioArr = []

lineDataRatioArr = []
lineCaptionRatioArr = []

assert len(captionFiles) == len(dataFiles) == len(titleFiles)

#may implemented seperately to avoid accidentally ignoring the word rather than month
months = ['january', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'november', 'december']

years = [str(i) for i in range(1850, 2050)]

fillers = ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', ':', '-',
           'to', 'but', 'is', 'of', 'it', 'on', '.', 'at', '(', ')', ',']

numbers = ['percent', 'percentage', '%', 'hundred', 'thousand', 'million', 'billion', 'trillion',
           'hundreds', 'thousands', 'millions', 'billions', 'trillions']

positiveTrends = ['increased', 'increase', 'increasing', 'grew', 'growing', 'rose', 'rising', 'gained', 'gaining']
negativeTrends = ['decreased', 'decrease', 'decreasing', 'shrank', 'shrinking', 'fell', 'falling', 'dropped', 'dropping']

#2 column data
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

        xRecord = cleanXAxis + '|' + cleanXValue + '|' + 'x' + '|' + chartType
        yRecord = cleanYAxis + '|' + cleanYValue + '|' + 'y' + '|' + chartType
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
    #manually find dates, it performs better than using NER
    for word in title.split():
        if word.isnumeric():
            if len(word) > 3:
                entities['Date'].append(word)
        elif word.replace('/','').isnumeric():
            word = word.split('/')[0]
            if len(word) > 3:
                entities['Date'].append(word)
        elif word.replace('-','').isnumeric():
            word = word.split('-')[0]
            if len(word) > 3:
                entities['Date'].append(word)
    #get named entites from title
    for X in doc.ents:
        if X.label_ == 'GPE' or X.label_ == 'ORG' or X.label_ == 'NORP' or X.label_ == 'LOC':
            cleanSubject = [word for word in X.text.split() if word.isalpha() and word not in fillers]
            if len(cleanSubject) > 0:
                entities['Subject'].append(' '.join(cleanSubject))
        if len(entities['Date']) < 1:
            if X.label_ == 'DATE':
                if X.text.isnumeric():
                    entities['Date'].append(X.text)
    #guess subject if NER doesn't find one
    if len(entities['Subject']) == 0:
        uppercaseWords = [word for word in title.split() if word[0].isupper()]
        if len(uppercaseWords) > 1:
            guessedSubject = ' '.join(uppercaseWords[1:])
        else:
            guessedSubject = uppercaseWords[0]
        entities['Subject'].append(guessedSubject)
    #print(entities['Date'])
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
            # check if last token was an un-templated month
            elif captionTokens[i].lower() in months or captionTokens[i] == 'May':
                captionTokens.pop(i)
        else:
            tokenBool = 0
        labelMap.append(str(tokenBool))
    assert len(captionTokens) == len(labelMap)
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
    #check for sentences containing a delta value
    newSentences = []
    cleanSentences = ' '.join(captionTokens).split(' . ')
    for sentence, sentIdx in zip(cleanSentences, range(len(cleanSentences))):
        scaleIndicator = False
        trendIndicator = False
        newSentence = []
        for token, tokenIdx in zip(sentence.split(), range(len(sentence))):
            if token == 'templateScale':
                try:
                    scale = captionSentences[sentIdx].split()[tokenIdx]
                    if scale in numbers:
                        scaleIndicator = True
                except:
                    print('scale err')
            if token.lower() in positiveTrends:
                token = 'templatePositiveTrend'
                trendIndicator = True
            elif token.lower() in negativeTrends:
                token = 'templateNegativeTrend'
                trendIndicator = True
            # if there is an unlabelled numeric token in a sentence containing a trend word, assume that token is a delta between two values
            if trendIndicator:
                if token not in years:
                    if is_number(token):
                        sentenceTemplates = [token for token in sentence.split() if 'template' in token]
                        xCount = {token for token in sentenceTemplates if 'templateXValue' in token}
                        yCount = {token for token in sentenceTemplates if 'templateYValue' in token}
                        #also compare 1 x and 1 y s
                        if len(xCount) == 2 or len(yCount) == 2 or (len(xCount) == 1 and len(yCount) == 1):
                            values, indices = getTemplateValues(xCount, yCount, xValueArr, yValueArr)
                            if len(values) > 1:
                                print(token, tokenIdx)
                                print(sentence)
                                print(xValueArr)
                                print(yValueArr)
                                print(xCount, values)
                                print(scale)
                                if scaleIndicator and (scale == 'percent' or scale == 'percentage'):
                                    valueDiff = abs((float(values[1]) - float(values[0]) / float(values[0])) * 100)
                                    rounded1 = abs(normal_round(valueDiff))
                                    rounded2 = abs(normal_round(valueDiff, 1))
                                    print(f'original: {token}, diff:{valueDiff} rounded:{rounded1, rounded2}')
                                else:
                                    valueDiff = abs(float(values[0]) - float(values[1]))
                                    rounded1 = abs(normal_round(valueDiff))
                                    rounded2 = abs(normal_round(valueDiff, 1))
                                    print(f'original: {token}, diff:{valueDiff} rounded:{rounded1, rounded2}')
                                if rounded1 == float(token) or rounded2 == float(token) or valueDiff == float(token):
                                    token = f'templateDelta[{indices[0]},{indices[1]}]'
                                    print('DELTA')
            newSentence.append(token)
        newSentences.append(' '.join(newSentence))
    assert len(captionTokens) == len(labelMap)
    dataRowPairs = [f'{xLabel} {yLabel}' for xLabel, yLabel in zip(xDataLabels, yDataLabels)]
    dataLabelLine = (' ').join(dataRowPairs)
    assert len(dataLabelLine.split()) == (len(xValueArr) + len(yValueArr))
    dataMatchCount = sum(xDataLabels) + sum(yDataLabels)
    dataRatio = round(dataMatchCount / (len(xValueArr) + len(yValueArr)), 2)
    captionRatio = round(captionMatchCount / len(captionTokens), 2)
    if captionMatchCount >= 1 and dataMatchCount >= 1:
        assert len(xValueArr) == len(yValueArr)
        if chartType == 'bar_chart':
            barDataRatioArr.append(dataRatio)
            barCaptionRatioArr.append(captionRatio)
            summaryLabelLine = (' ').join(labelMap)
            assert len(captionTokens) == len(summaryLabelLine.split())
            newCaption = (' . ').join(newSentences)
            barOldSummaryArr.append(trimmedCaption)
            barLabelList.append(labelMap)
            barDataArr.append(dataLine)
            barDataLabelArr.append(dataLabelLine)
            barSummaryArr.append(newCaption)
            barSummaryLabelArr.append(summaryLabelLine)
            barTitleArr.append(title)
        elif chartType == 'line_chart':
            lineDataRatioArr.append(dataRatio)
            lineCaptionRatioArr.append(captionRatio)
            summaryLabelLine = (' ').join(labelMap)
            assert len(captionTokens) == len(summaryLabelLine.split())
            newCaption = (' . ').join(newSentences)
            lineOldSummaryArr.append(trimmedCaption)
            lineLabelList.append(labelMap)
            lineDataArr.append(dataLine)
            lineDataLabelArr.append(dataLabelLine)
            lineSummaryArr.append(newCaption)
            lineSummaryLabelArr.append(summaryLabelLine)
            lineTitleArr.append(title)

assert len(barDataArr) == len(barDataLabelArr)
assert len(barSummaryArr) == len(barSummaryLabelArr)
assert len(barSummaryArr) == len(barOldSummaryArr)
assert len(barTitleArr) == len(barDataArr)

assert len(lineDataArr) == len(lineDataLabelArr)
assert len(lineSummaryArr) == len(lineSummaryLabelArr)
assert len(lineSummaryArr) == len(lineOldSummaryArr)
assert len(lineTitleArr) == len(lineDataArr)

# TODO REVERT SET SIZES
barTrainSize = round(len(barDataArr) * 0.7)
barTestSize = round(len(barDataArr) * 0.15)
barValidSize = len(barDataArr) - barTrainSize - barTestSize

barTrainData = barDataArr[0:barTrainSize]
barTestData = barDataArr[barTrainSize:barTrainSize + barTestSize]
barValidData = barDataArr[barTrainSize + barTestSize:]

barTrainDataLabel = barDataLabelArr[0:barTrainSize]
barTestDataLabel = barDataLabelArr[barTrainSize:barTrainSize + barTestSize]
barValidDataLabel = barDataLabelArr[barTrainSize + barTestSize:]

barTrainSummary = barSummaryArr[0:barTrainSize]
barTestSummary = barSummaryArr[barTrainSize:barTrainSize + barTestSize]
barValidSummary = barSummaryArr[barTrainSize + barTestSize:]

barTrainSummaryLabel = barSummaryLabelArr[0:barTrainSize]
barTestSummaryLabel = barSummaryLabelArr[barTrainSize:barTrainSize + barTestSize]
barValidSummaryLabel = barSummaryLabelArr[barTrainSize + barTestSize:]

barTrainTitle = barTitleArr[0:barTrainSize]
barTestTitle = barTitleArr[barTrainSize:barTrainSize + barTestSize]
barValidTitle = barTitleArr[barTrainSize + barTestSize:]

barOldTrainSummary = barOldSummaryArr[0:barTrainSize]
barOldTestSummary = barOldSummaryArr[barTrainSize:barTrainSize + barTestSize]
barOldValidSummary = barOldSummaryArr[barTrainSize + barTestSize:]

with open('../data/bar/train/trainData.txt', mode='wt', encoding='utf8') as myfile0:
    myfile0.writelines("%s\n" % line for line in barTrainData)
with open('../data/bar/train/trainDataLabel.txt', mode='wt', encoding='utf8') as myfile1:
    myfile1.writelines("%s\n" % line for line in barTrainDataLabel)

with open('../data/bar/test/testData.txt', mode='wt', encoding='utf8') as myfile2:
    myfile2.writelines("%s\n" % line for line in barTestData)
with open('../data/bar/test/testDataLabel.txt', mode='wt', encoding='utf8') as myfile3:
    myfile3.writelines("%s\n" % line for line in barTestDataLabel)

with open('../data/bar/valid/validData.txt', mode='wt', encoding='utf8') as myfile4:
    myfile4.writelines("%s\n" % line for line in barValidData)
with open('../data/bar/valid/validDataLabel.txt', mode='wt', encoding='utf8') as myfile5:
    myfile5.writelines("%s\n" % line for line in barValidDataLabel)

with open('../data/bar/train/trainSummary.txt', mode='wt', encoding='utf8') as myfile6:
    myfile6.writelines("%s\n" % line for line in barTrainSummary)
with open('../data/bar/train/trainSummaryLabel.txt', mode='wt', encoding='utf8') as myfile7:
    myfile7.writelines("%s\n" % line for line in barTrainSummaryLabel)

with open('../data/bar/test/testSummary.txt', mode='wt', encoding='utf8') as myfile8:
    myfile8.writelines("%s\n" % line for line in barTestSummary)
with open('../data/bar/test/testSummaryLabel.txt', mode='wt', encoding='utf8') as myfile9:
    myfile9.writelines("%s\n" % line for line in barTestSummaryLabel)

with open('../data/bar/valid/validSummary.txt', mode='wt', encoding='utf8') as myfile10:
    myfile10.writelines("%s\n" % line for line in barValidSummary)
with open('../data/bar/valid/validSummaryLabel.txt', mode='wt', encoding='utf8') as myfile11:
    myfile11.writelines("%s\n" % line for line in barValidSummaryLabel)

with open('../data/bar/dataRatio.txt', mode='wt', encoding='utf8') as myfile12:
    myfile12.write(str(barDataRatioArr))
with open('../data/bar/captionRatio.txt', mode='wt', encoding='utf8') as myfile13:
    myfile13.write(str(barCaptionRatioArr))

with open('../data/bar/train/trainTitle.txt', mode='wt', encoding='utf8') as myfile14:
    myfile14.writelines("%s" % line for line in barTrainTitle)
with open('../data/bar/test/testTitle.txt', mode='wt', encoding='utf8') as myfile15:
    myfile15.writelines("%s" % line for line in barTestTitle)
with open('../data/bar/valid/validTitle.txt', mode='wt', encoding='utf8') as myfile16:
    myfile16.writelines("%s" % line for line in barValidTitle)

with open('../data/bar/train/trainOriginalSummary.txt', mode='wt', encoding='utf8') as myfile17:
    myfile17.writelines("%s" % line for line in barOldTrainSummary)
with open('../data/bar/test/testOriginalSummary.txt', mode='wt', encoding='utf8') as myfile18:
    myfile18.writelines("%s" % line for line in barOldTestSummary)
with open('../data/bar/valid/validOriginalSummary.txt', mode='wt', encoding='utf8') as myfile19:
    myfile19.writelines("%s" % line for line in barOldValidSummary)

plt.hist(barDataRatioArr, 6)
plt.savefig('../data/bar/data.png')
plt.close('all')
plt.hist(barCaptionRatioArr, 6)
plt.savefig('../data/bar/caption.png')
plt.close('all')

lineTrainSize = round(len(lineDataArr) * 0.7)
lineTestSize = round(len(lineDataArr) * 0.15)
lineValidSize = len(lineDataArr) - lineTrainSize - lineTestSize

lineTrainData = lineDataArr[0:lineTrainSize]
lineTestData = lineDataArr[lineTrainSize:lineTrainSize + lineTestSize]
lineValidData = lineDataArr[lineTrainSize + lineTestSize:]

lineTrainDataLabel = lineDataLabelArr[0:lineTrainSize]
lineTestDataLabel = lineDataLabelArr[lineTrainSize:lineTrainSize + lineTestSize]
lineValidDataLabel = lineDataLabelArr[lineTrainSize + lineTestSize:]

lineTrainSummary = lineSummaryArr[0:lineTrainSize]
lineTestSummary = lineSummaryArr[lineTrainSize:lineTrainSize + lineTestSize]
lineValidSummary = lineSummaryArr[lineTrainSize + lineTestSize:]

lineTrainSummaryLabel = lineSummaryLabelArr[0:lineTrainSize]
lineTestSummaryLabel = lineSummaryLabelArr[lineTrainSize:lineTrainSize + lineTestSize]
lineValidSummaryLabel = lineSummaryLabelArr[lineTrainSize + lineTestSize:]

lineTrainTitle = lineTitleArr[0:lineTrainSize]
lineTestTitle = lineTitleArr[lineTrainSize:lineTrainSize + lineTestSize]
lineValidTitle = lineTitleArr[lineTrainSize + lineTestSize:]

lineOldTrainSummary = lineOldSummaryArr[0:lineTrainSize]
lineOldTestSummary = lineOldSummaryArr[lineTrainSize:lineTrainSize + lineTestSize]
lineOldValidSummary = lineOldSummaryArr[lineTrainSize + lineTestSize:]

with open('../data/line/train/trainData.txt', mode='wt', encoding='utf8') as myfile0:
    myfile0.writelines("%s\n" % line for line in lineTrainData)
with open('../data/line/train/trainDataLabel.txt', mode='wt', encoding='utf8') as myfile1:
    myfile1.writelines("%s\n" % line for line in lineTrainDataLabel)

with open('../data/line/test/testData.txt', mode='wt', encoding='utf8') as myfile2:
    myfile2.writelines("%s\n" % line for line in lineTestData)
with open('../data/line/test/testDataLabel.txt', mode='wt', encoding='utf8') as myfile3:
    myfile3.writelines("%s\n" % line for line in lineTestDataLabel)

with open('../data/line/valid/validData.txt', mode='wt', encoding='utf8') as myfile4:
    myfile4.writelines("%s\n" % line for line in lineValidData)
with open('../data/line/valid/validDataLabel.txt', mode='wt', encoding='utf8') as myfile5:
    myfile5.writelines("%s\n" % line for line in lineValidDataLabel)

with open('../data/line/train/trainSummary.txt', mode='wt', encoding='utf8') as myfile6:
    myfile6.writelines("%s\n" % line for line in lineTrainSummary)
with open('../data/line/train/trainSummaryLabel.txt', mode='wt', encoding='utf8') as myfile7:
    myfile7.writelines("%s\n" % line for line in lineTrainSummaryLabel)

with open('../data/line/test/testSummary.txt', mode='wt', encoding='utf8') as myfile8:
    myfile8.writelines("%s\n" % line for line in lineTestSummary)
with open('../data/line/test/testSummaryLabel.txt', mode='wt', encoding='utf8') as myfile9:
    myfile9.writelines("%s\n" % line for line in lineTestSummaryLabel)

with open('../data/line/valid/validSummary.txt', mode='wt', encoding='utf8') as myfile10:
    myfile10.writelines("%s\n" % line for line in lineValidSummary)
with open('../data/line/valid/validSummaryLabel.txt', mode='wt', encoding='utf8') as myfile11:
    myfile11.writelines("%s\n" % line for line in lineValidSummaryLabel)

with open('../data/line/dataRatio.txt', mode='wt', encoding='utf8') as myfile12:
    myfile12.write(str(lineDataRatioArr))
with open('../data/line/captionRatio.txt', mode='wt', encoding='utf8') as myfile13:
    myfile13.write(str(lineCaptionRatioArr))

with open('../data/line/train/trainTitle.txt', mode='wt', encoding='utf8') as myfile14:
    myfile14.writelines("%s" % line for line in lineTrainTitle)
with open('../data/line/test/testTitle.txt', mode='wt', encoding='utf8') as myfile15:
    myfile15.writelines("%s" % line for line in lineTestTitle)
with open('../data/line/valid/validTitle.txt', mode='wt', encoding='utf8') as myfile16:
    myfile16.writelines("%s" % line for line in lineValidTitle)

with open('../data/line/train/trainOriginalSummary.txt', mode='wt', encoding='utf8') as myfile17:
    myfile17.writelines("%s" % line for line in lineOldTrainSummary)
with open('../data/line/test/testOriginalSummary.txt', mode='wt', encoding='utf8') as myfile18:
    myfile18.writelines("%s" % line for line in lineOldTestSummary)
with open('../data/line/valid/validOriginalSummary.txt', mode='wt', encoding='utf8') as myfile19:
    myfile19.writelines("%s" % line for line in lineOldValidSummary)

plt.hist(lineDataRatioArr, 6)
plt.savefig('../data/line/data.png')
plt.close('all')
plt.hist(lineCaptionRatioArr, 6)
plt.savefig('../data/line/caption.png')
plt.close('all')