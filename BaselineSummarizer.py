import json
import os
from statistics import mean

import numpy as np
import pandas as pd
import math
from random import randint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import traceback
from scipy.stats import linregress
import re

goldPath = './data/test/testOriginalSummary.txt'
dataPath = './data/test/testData.txt'
titlePath = './data/test/testTitle.txt'

websitePath = './results/july30/generated'

dataArr = []
dataLabelArr = []
summaryArr = []
summaryLabelArr = []
labelList = []
titleArr = []
oldSummaryArr = []

dataRatioArr = []
captionRatioArr = []


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
    cleanLabel = cleanLabel.replace('%', '').replace('*', '')
    return cleanLabel


def cleanAxisValue(value):
    #print(value)
    if value == '-' or value == 'nan':
        return '0'
    cleanValue = re.sub('\s', '_', value)
    cleanValue = cleanValue.replace('|', '').replace(',', '').replace('%', '').replace('*', '')
    return cleanValue

count = 0
with open(goldPath, 'r', encoding='utf-8') as goldFile, open(dataPath, 'r', encoding='utf-8') as dataFile, \
        open(titlePath, 'r', encoding='utf-8') as titleFile:
    #assert len(goldFile.readlines()) == len(dataFile.readlines()) == len(titleFile.readlines())
    fileIterators = zip(goldFile.readlines(), dataFile.readlines(), titleFile.readlines())
    for gold, data, title in fileIterators:
        count += 1
        datum = data.split()
        # check if data is multi column
        columnType = datum[0].split('|')[2].isnumeric()
        if columnType:
            labelArr = []
            chartType = datum[0].split('|')[3].split('_')[0]
            values = [value.split('|')[1] for value in datum]
            # find number of columns:
            columnCount = max([int(data.split('|')[2]) for data in datum]) + 1
            # Get labels
            for i in range(columnCount):
                label = datum[i].split('|')[0].split('_')
                labelArr.append(label)
            stringLabels = [' '.join(label) for label in labelArr]
            # Get values
            valueArr = [[] for i in range(columnCount)]
            cleanValArr = [[] for i in range(columnCount)]
            rowCount = round(len(datum) / columnCount)
            categoricalValueArr = [[] for i in range(rowCount)]
            i = 0
            for n in range(rowCount):
                for m in range(columnCount):
                    value = values[i]
                    cleanVal = datum[i].split('|')[1].replace('_', ' ')
                    valueArr[m].append(value)
                    cleanValArr[m].append(cleanVal)
                    if m == 0:
                        categoricalValueArr[n].append(cleanVal)
                    else:
                        categoricalValueArr[n].append(float(re.sub("[^\d\.]", "", cleanVal)))
                    i += 1
            titleArr = title.split()
            #calculate top two largest categories

            summaryArray = []
            dataJson = []
            # iterate over index of a value
            for i in range(len(cleanValArr[0])):
                # iterate over each value
                dico = {}
                for value, label in zip(cleanValArr, labelArr):
                    cleanLabel = ' '.join(label)
                    dico[cleanLabel] = value[i]
                dataJson.append(dico)
            if (chartType == "bar"):
                meanCategoricalDict = {}
                stringLabels.insert(len(stringLabels)-1, 'and')
                categories = ', '.join(stringLabels[1:-1]) + f' {stringLabels[-1]}'
                if rowCount > 1:
                    for category in categoricalValueArr:
                        meanCategoricalDict[category[0]] = mean(category[1:])
                    sortedCategories = sorted(meanCategoricalDict.items(), key=lambda x: x[1])
                    numerator = abs(sortedCategories[-1][1] - sortedCategories[-2][1])
                    denominator = (sortedCategories[-1][1] + sortedCategories[-2][1]) / 2
                    topTwoDelta = round((numerator / denominator) * 100, 1)

                    summary1 = f"This grouped bar chart has {rowCount} categories for the x axis of {stringLabels[0]} representing {categories}."
                    summary2 = f" The highest category is found at {sortedCategories[-1][0]} with a mean value of {sortedCategories[-1][1]}."
                    summaryArray.append(summary1)
                    summaryArray.append(summary2)
                    maxValueIndex = cleanValArr[0].index(sortedCategories[-1][0])
                    secondValueIndex = cleanValArr[0].index(sortedCategories[-1][0])
                    #minValueIndex = valueArr[0].index(sortedCategories[-1][0])
                    trendsArray = {"maxIndex": maxValueIndex, "secondIndex": secondValueIndex}#, "minIndex": minValueIndex}
                else:
                    summary1 = f"This grouped bar chart has 1 category for the x axis of {stringLabels[0]} representing {categories}."
                    summaryArray.append(summary1)
                    trendsArray = {"maxIndex": 0, "secondIndex": 0}
                summary3 = f" The second highest category is found at {sortedCategories[-2][0]} with a mean value of {sortedCategories[-2][1]}."
                summary4 = f" This represents a difference of {topTwoDelta}%."
                summaryArray.append(summary3)
                summaryArray.append(summary4)
                websiteInput = {"title": title.strip(), "labels": [' '.join(label) for label in labelArr],
                                "columnType": "multi", \
                                "graphType": chartType, "summary": summaryArray, "trends": trendsArray,
                                "data": dataJson}
                with open(f'{websitePath}/{count}.json', 'w', encoding='utf-8') as websiteFile:
                    json.dump(websiteInput, websiteFile, indent=3)
            elif (chartType == "line"):
                #clean data
                intData = []
                for line in valueArr[1:]:
                    cleanLine = []
                    for data in line:
                        if data.isnumeric():
                            cleanLine.append(float(data))
                        else:
                            cleanData = re.sub("[^\d\.]", "", data)
                            if len(cleanData) > 0:
                                cleanLine.append(float(cleanData[:4]))
                            else:
                                cleanLine.append(float(cleanData))
                        intData.append(cleanLine)
                #calculate mean for each line
                meanLineVals = {}
                for label, data in zip(stringLabels[1:], intData[1:]):
                    meanLineVals[label] = round(mean(data), 1)
                sortedLines = sorted(meanLineVals.items(), key=lambda x: x[1])
                #if more than 2 lines
                lineCount = len(labelArr) - 1
                maxLine = sortedLines[-1]
                index = stringLabels.index(maxLine[0])
                maxLineData = max(intData[index])
                maxXValue = valueArr[0][intData[index].index(maxLineData)]
                secondLine = sortedLines[-2]
                index = stringLabels.index(secondLine[0])
                secondLineData = max(intData[index])
                secondXValue = valueArr[0][intData[index].index(secondLineData)]
                if len(labelArr[1:]) > 2:
                    summaryArr = f'This line chart has {lineCount} lines. The line representing' \
                                 f' {maxLine[0]} had the highest values across {stringLabels[0]} with a mean value of {maxLine[1]}.' \
                                 f' This peaked at {maxXValue} with a value of {maxLineData}. ' \
                                 f' The line with the second highest values was {secondLine[0]} with a mean value of {secondLine[1]}.' \
                                 f' This line peaked at {secondXValue} with a value of {secondLineData}'
                    trendsArray = {"maxIndex": 0, "secondIndex": 0}
                #if 2 lines
                else:
                    summaryArr = f'This line chart has {lineCount} lines. The line representing' \
                                 f' {maxLine[0]} had the highest values across {stringLabels[0]} with a mean value of {maxLine[1]}.' \
                                 f' This peaked at {maxXValue} with a value of {maxLineData}. ' \
                                 f' The line with the second highest values was {secondLine[0]} with a mean value of {secondLine[1]}.' \
                                 f' This line peaked at {secondXValue} with a value of {secondLineData}'
                    trendsArray = {"maxIndex": 0, "secondIndex": 0}
                websiteInput = {"title": title.strip(), "labels": [' '.join(label) for label in labelArr],
                                "columnType": "multi", \
                                "graphType": chartType, "summary": summaryArray, "trends": trendsArray,
                                "data": dataJson}
                #with open(f'{websitePath}/{count}.json', 'w', encoding='utf-8') as websiteFile:
                #    json.dump(websiteInput, websiteFile, indent=3)
        else:
            xValueArr = []
            yValueArr = []
            cleanXArr = []
            cleanYArr = []
            xLabel = ' '.join(datum[0].split('|')[0].split('_'))
            yLabel = ' '.join(datum[1].split('|')[0].split('_'))
            chartType = datum[0].split('|')[3].split('_')[0]
            for i in range(0, len(datum)):
                if i % 2 == 0:
                    xValueArr.append((datum[i].split('|')[1]))
                    cleanXArr.append((datum[i].split('|')[1].replace('_', ' ')))
                else:
                    yValueArr.append(float(re.sub("[^\d\.]", "", datum[i].split('|')[1])))
                    cleanYArr.append(float(re.sub("[^\d\.]", "", datum[i].split('|')[1])))
            titleArr = title.split()
            xValueArr = pd.Series(xValueArr)
            yValueArr = pd.Series(yValueArr)
            maxValue = str(max(yValueArr))
            minValue = str(min(yValueArr))
            maxValueIndex = yValueArr.idxmax()
            minValueIndex = yValueArr.idxmin()
            # run bar
            summaryArray = []
            if (chartType == "bar"):
                summary1 = "This bar chart has " + str(xValueArr.size) + " categories for the x axis representing " + xLabel + ", and a y axis representing " + yLabel + "."
                summary2 = " The highest category is found at " + str(
                    xValueArr[maxValueIndex]) + " where " + yLabel + " is " + str(yValueArr[maxValueIndex]) + "."
                summary3 = " The lowest category is found at " + str(
                    xValueArr[minValueIndex]) + " where " + yLabel + " is " + str(yValueArr[minValueIndex]) + "."
                summaryArray.append(summary1)
                summaryArray.append(summary2)
                summaryArray.append(summary3)
                trendsArray = {"maxIndex": maxValueIndex, "minIndex": minValueIndex}
                dataJson = [{xLabel: xVal, yLabel: yVal} for xVal, yVal in zip(cleanXArr, cleanYArr)]
                websiteInput = {"title": title, "xAxis": xLabel, "yAxis": yLabel,
                                "columnType": "two", \
                                "graphType": chartType, "summary": summaryArray, "trends": trendsArray,
                                "data": dataJson}
                with open(f'{websitePath}/{count}.json', 'w', encoding='utf-8') as websiteFile:
                    json.dump(websiteInput, websiteFile, indent=3)
            # run line
            elif (chartType == "line"):
                numericXValueArr = []
                for xVal, index in zip(xValueArr, range(xValueArr.size)):
                   if xVal.isnumeric():
                       numericXValueArr.append(float(xVal))
                   else:
                       #see if regex works better
                       cleanxVal = re.sub("[^\d\.]", "", xVal)
                       if len(cleanxVal) > 0:
                            numericXValueArr.append(float(cleanxVal[:4]))
                       else:
                           numericXValueArr.append(float(index))
                numericXValueArr = pd.Series(numericXValueArr)
                # determine local trends
                trendArray = []
                i = 1
                # calculate variance between each adjacent y values
                while i < (len(yValueArr)):
                    variance1 = float(yValueArr[i]) - float(yValueArr[i - 1])
                    if (variance1 > 0):
                        type1 = "positive"
                    elif (variance1 < 0):
                        type1 = "negative"
                    else:
                        type1 = "neutral"
                    trendArray.append(type1)
                    i = i + 1
                localTrends = []
                n = 0
                startIndex = 0
                trendLen = len(trendArray)
                # iterate through the variances and check for trends
                # creates dictionary containing the trend length, direction, start and end indices, and the linear regression of the trend
                while (n < trendLen):
                    currentVal = trendArray[n - 1]
                    nextVal = trendArray[n]
                    if (currentVal != nextVal or (currentVal == nextVal and n == (trendLen - 1))):
                        if (n == (trendLen - 1)):
                            endIndex = n + 1
                        else:
                            endIndex = n
                        #if endIndex - startIndex > 1:
                        xRange = list(numericXValueArr.loc[startIndex:endIndex].array)
                        yRange = list(yValueArr.loc[startIndex:endIndex].array)
                        try:
                            result = linregress(xRange, yRange)
                            np.seterr('raise')
                            slope = round(result.slope, 2)
                        except:
                            print('slope err')
                            slope = 1
                            intercept = 1
                        """
                        # normalize slope to between 0 and 1
                        minimum = min(yValueArr)
                        maximum = max(yValueArr)
                        # called feature scaling / minmaxScaling
                        scaler = preprocessing.MinMaxScaler()
                        scaledX = scaler.fit_transform(xRange.values.reshape(-1, 1))
                        scaledY = scaler.fit_transform(yRange.values.reshape(-1, 1))
                        result2 = linregress(scaledX.reshape(1, -1), scaledY.reshape(1, -1))
                        normalizedSlope = round(result2.slope, 2)
                        print(normalizedSlope)
                        if (abs(normalizedSlope) > 0.75):
                            magnitude = "extremely"
                        elif (abs(normalizedSlope) > 0.5 and abs(normalizedSlope) <= 0.75):
                            magnitude = "strongly"
                        elif (abs(normalizedSlope) > 0.25 and abs(normalizedSlope) <= 0.5):
                            magnitude = "moderately"
                        else:
                            magnitude = "slightly"
                        if type1 == "neutral":
                            magnitude = ""
                        """
                        magnitude = ""
                        intercept = round(result[1], 2)
                        trend = "y=" + str(slope) + "x+" + str(intercept)
                        trendRange = {"Length": (endIndex - startIndex + 1), "direction": currentVal, "start": startIndex,
                                      "end": endIndex, "trend": trend, "magnitude": magnitude}
                        localTrends.append(trendRange)
                        startIndex = n
                    n = n + 1
                # sort the trend dictionaries by length
                sortedTrends = sorted(localTrends, key=lambda i: i['Length'], reverse=True)
                localTrendSummary = ""
                # determine the trend length which we consider to be significant (ex: if trend is longer than 1/4 of total length)
                significanceRange = round(yValueArr.size / 8)
                significantTrendCount = 0
                significantTrendArray = []
                # iterate through array of trend dictionaries, creating a new array with the trend dictionaries which are larger than the significance range
                for trends in sortedTrends:
                    if (trends['Length'] > significanceRange):
                        significantTrendCount = significantTrendCount + 1
                        significantTrendArray.append(trends)
                    # generate the textual summary from the significant trend dictionary array at m
                if (significantTrendCount > 0):
                    m = 1
                    startVal = str(xValueArr[(significantTrendArray[0]['start'])])
                    endVal = str(xValueArr[(significantTrendArray[0]['end'])])
                    direction = str(significantTrendArray[0]['direction'])
                    magnitude = str(significantTrendArray[0]['magnitude'])
                    # execute here if more than 1 significant trend
                    similarSynonyms = ["Similarly", "Correspondingly", "Likewise", "Additionally", "Also",
                                       "In a similar manner"]
                    contrarySynonyms = ["Contrarily", "Differently", "On the other hand", "Conversely", "On the contrary",
                                        "But"]
                    if (significantTrendCount > 1):
                        extraTrends = ""
                        localTrendSentence1 = "This line chart has an x axis representing " + xLabel + " and a y axis representing " + yLabel + ", with a total of " + str(yValueArr.size) \
                                              + " data points. The chart has " + str(significantTrendCount) + " significant trends."
                        summaryArray.append(localTrendSentence1)
                        localTrendSummary = " The longest trend is " + magnitude + " " + direction + " which exists from " + startVal + " to " + endVal + "."
                        summaryArray.append(localTrendSummary)
                        while (m < significantTrendCount):
                            # append conjunction between significant trends
                            if (direction == "positive"):
                                length = len(similarSynonyms)
                                random = randint(0, length - 1)
                                synonym = similarSynonyms[random]
                                conjunction = synonym + ","
                            elif (direction == "negative"):
                                length = len(contrarySynonyms)
                                random = randint(0, length - 1)
                                synonym = contrarySynonyms[random]
                                conjunction = synonym + ","
                            startVal = str(xValueArr[(significantTrendArray[m]['start'])])
                            endVal = str(xValueArr[(significantTrendArray[m]['end'])])
                            direction = str(significantTrendArray[m]['direction'])
                            magnitude = str(significantTrendArray[m]['magnitude'])
                            extraTrends = " " + conjunction + " the next significant trend is " + magnitude + " " + direction + " which exists from " + startVal + " to " + endVal + "."
                            summaryArray.append(extraTrends)
                            m = m + 1
                    # execute here if only 1 significant trend
                    else:
                        localTrendSentence1 = "This line chart has an x axis representing " + xLabel + " and a y axis representing " + yLabel + " with a total of " + str(yValueArr.size) + " data points. The chart has one significant trend."
                        summaryArray.append(localTrendSentence1)
                        localTrendSummary = " This trend is " + magnitude + " " + direction + " which exists from " + startVal + " to " + endVal + "."
                        summaryArray.append(localTrendSummary)
                dataJson = [{xLabel: xVal, yLabel: yVal} for xVal, yVal in zip(cleanXArr, cleanYArr)]
                websiteInput = {"title": title, "xAxis": xLabel, "yAxis": yLabel,
                                "columnType": "two", \
                                "graphType": chartType, "summary": summaryArray, "trends": significantTrendArray,
                                "data": dataJson}
                with open(f'{websitePath}/{count}.json', 'w', encoding='utf-8') as websiteFile:
                    json.dump(websiteInput, websiteFile, indent=3)
        # print(summaryArray)
