import json
import re
from random import randint
from statistics import mean
from operator import itemgetter
from scipy.stats import linregress
from sklearn import preprocessing
import pandas as pd
import numpy as np

goldPath = '../data/test/testOriginalSummary.txt'
dataPath = '../data/test/testData.txt'
titlePath = '../data/test/testTitle.txt'

websitePath = '../results/aug17/generated_baseline'
onePath = '../results/aug17/generated_baseline.txt'

summaryList = []


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
    # print(value)
    if value == '-' or value == 'nan':
        return '0'
    cleanValue = re.sub('\s', '_', value)
    cleanValue = cleanValue.replace('|', '').replace(',', '').replace('%', '').replace('*', '')
    return cleanValue


def getMagnitude(normalizedSlope):
    magnitude = "slightly"
    print(normalizedSlope)
    if (abs(normalizedSlope) > 0.75):
        magnitude = "extremely"
    elif (abs(normalizedSlope) > 0.25 and abs(normalizedSlope) <= 0.75):
        magnitude = "moderately"
    else:
        mangitude = "slightly"
    return magnitude


scaler = preprocessing.MinMaxScaler()
count = 0
with open(goldPath, 'r', encoding='utf-8') as goldFile, open(dataPath, 'r', encoding='utf-8') as dataFile, \
        open(titlePath, 'r', encoding='utf-8') as titleFile, open(onePath, 'w', encoding='utf-8') as oneFile:
    # assert len(goldFile.readlines()) == len(dataFile.readlines()) == len(titleFile.readlines())
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
            # calculate top two largest categories
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
                stringLabels.insert(len(stringLabels) - 1, 'and')
                categories = ', '.join(stringLabels[1:-1]) + f' {stringLabels[-1]}'
                if rowCount > 1:
                    for category in categoricalValueArr:
                        meanCategoricalDict[category[0]] = mean(category[1:])
                    sortedCategories = sorted(meanCategoricalDict.items(), key=lambda x: x[1])
                    numerator = abs(sortedCategories[-1][1] - sortedCategories[-2][1])
                    denominator = (sortedCategories[-1][1] + sortedCategories[-2][1]) / 2
                    topTwoDelta = round((numerator / denominator) * 100, 1)

                    summary1 = f"This grouped bar chart has {rowCount} categories for the x axis of {stringLabels[0]} representing {categories}."
                    summary2 = f" The highest category is found at {sortedCategories[-1][0]} with a mean value of {round(sortedCategories[-1][1],2)}."
                    summaryArray.append(summary1)
                    summaryArray.append(summary2)
                    maxValueIndex = cleanValArr[0].index(sortedCategories[-1][0])
                    secondValueIndex = cleanValArr[0].index(sortedCategories[-2][0])
                    summary3 = f" The second highest category is found at {sortedCategories[-2][0]} with a mean value of {round(sortedCategories[-2][1],2)}."
                    summary4 = f" This represents a difference of {topTwoDelta}%."
                    summaryArray.append(summary3)
                    summaryArray.append(summary4)
                    trendsArray = [
                        {},{"2": ["0", str(maxValueIndex)], "13": [str(columnCount-1), str(maxValueIndex)]},
                           {"2": ["0", str(secondValueIndex)], "14": [str(columnCount-1), str(secondValueIndex)]},{}
                    ]
                else:
                    summary1 = f"This grouped bar chart has 1 category for the x axis of {stringLabels[0]}."
                    summary2 = f" This category is {stringLabels[1]}, with a mean value of {round(mean(categoricalValueArr[1]),2)}."
                    summaryArray.append(summary1)
                    summaryArray.append(summary2)
                    trendsArray = [{},{"3": ["0", "0"], "9": ["0", "0"]}]
                websiteInput = {"title": title.strip(), "labels": [' '.join(label) for label in labelArr],
                                "columnType": "multi",
                                "graphType": chartType, "summaryType": "baseline", "summary": summaryArray,
                                "trends": trendsArray,
                                "data": dataJson, "gold": gold}
                with open(f'{websitePath}/{count}.json', 'w', encoding='utf-8') as websiteFile:
                    json.dump(websiteInput, websiteFile, indent=3)
                oneFile.writelines(''.join(summaryArray)+'\n')
            elif (chartType == "line"):
                # clean data
                intData = []
                print(valueArr[1:])
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
                    print(len(intData))
                # calculate mean for each line
                meanLineVals = []
                assert len(stringLabels[1:]) == len(intData)
                for label, data in zip(stringLabels[1:], intData):
                    x = (label, round(mean(data), 1))
                    print(x)
                    meanLineVals.append(x)
                sortedLines = sorted(meanLineVals,key=itemgetter(1))
                # if more than 2 lines
                lineCount = len(labelArr) - 1
                maxLine = sortedLines[-1]
                index1 = stringLabels.index(maxLine[0]) - 1
                maxLineData = round(max(intData[index1]), 2)
                maxXValue = valueArr[0][intData[index1].index(maxLineData)]
                secondLine = sortedLines[-2]
                rowIndex1 = intData[index1].index(maxLineData)
                index2 = stringLabels.index(secondLine[0]) - 1
                secondLineData = round(max(intData[index2]), 2)
                secondXValue = valueArr[0][intData[index2].index(secondLineData)]
                rowIndex2 = intData[index2].index(secondLineData)
                summaryArr = [f'This line chart has {lineCount} lines.',
                              f' The line representing {maxLine[0]} had the highest values across {stringLabels[0]} with a mean value of {maxLine[1]}.',
                              f' This peaked at {maxXValue} with a value of {maxLineData}. ',
                              f' The line with the second highest values was {secondLine[0]} with a mean value of {secondLine[1]}.',
                              f' This line peaked at {secondXValue} with a value of {secondLineData}']
                trendsArray = [{},
                               {"2": ["0", str(index1)], "16": [str(rowCount - 1), str(index1)]},
                               {"1": [str(rowIndex1), str(index1)], "9": [str(rowIndex1), str(index1)]},
                               {"2": ["0", str(index2)], "15": [str(rowCount - 1), str(index2)]},
                               {"1": [str(rowIndex2), str(index2)], "10": [str(rowIndex2), str(index2)]}
                               ]
                websiteInput = {"title": title.strip(), "labels": [' '.join(label) for label in labelArr],
                                "columnType": "multi", \
                                "graphType": chartType, "summaryType": "baseline", "summary": summaryArr,
                                "trends": trendsArray,
                                "data": dataJson, "gold": gold}
                with open(f'{websitePath}/{count}.json', 'w', encoding='utf-8') as websiteFile:
                    json.dump(websiteInput, websiteFile, indent=3)
                oneFile.writelines(''.join(summaryArr)+'\n')
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
            maxValue = str(max(yValueArr))
            minValue = str(min(yValueArr))
            maxValueIndex = pd.Series(yValueArr).idxmax()
            minValueIndex = pd.Series(yValueArr).idxmin()
            # run bar
            summaryArray = []
            if (chartType == "bar"):
                summary1 = "This bar chart has " + str(len(xValueArr)) + " categories for the x axis representing " + xLabel + ", and a y axis representing " + yLabel + "."
                summary2 = " The highest category is found at " + str(
                    xValueArr[maxValueIndex]) + " where " + yLabel + " is " + str(yValueArr[maxValueIndex]) + "."
                summary3 = " The lowest category is found at " + str(
                    xValueArr[minValueIndex]) + " where " + yLabel + " is " + str(yValueArr[minValueIndex]) + "."
                summaryArray.append(summary1)
                summaryArray.append(summary2)
                summaryArray.append(summary3)
                trendsArray = [{},{"7": maxValueIndex, "12": maxValueIndex},
                                  {"7": minValueIndex, "12": minValueIndex},{}]
                dataJson = [{xLabel: xVal, yLabel: yVal} for xVal, yVal in zip(cleanXArr, cleanYArr)]
                websiteInput = {"title": title, "xAxis": xLabel, "yAxis": yLabel,
                                "columnType": "two", \
                                "graphType": chartType, "summaryType": "baseline", "summary": summaryArray,
                                "trends": trendsArray,
                                "data": dataJson, "gold": gold}
                with open(f'{websitePath}/{count}.json', 'w', encoding='utf-8') as websiteFile:
                    json.dump(websiteInput, websiteFile, indent=3)
                oneFile.writelines(''.join(summaryArray)+'\n')
            # run line
            elif (chartType == "line"):
                trendArray = []
                numericXValueArr = []
                for xVal, index in zip(xValueArr, range(len(xValueArr))):
                    if xVal.isnumeric():
                        numericXValueArr.append(float(xVal))
                    else:
                        # see if regex works better
                        cleanxVal = re.sub("[^\d\.]", "", xVal)
                        if len(cleanxVal) > 0:
                            numericXValueArr.append(float(cleanxVal[:4]))
                        else:
                            numericXValueArr.append(float(index))
                # determine local trends
                graphTrendArray = []
                i = 1
                # calculate variance between each adjacent y values
                while i < (len(yValueArr)):
                    variance1 = float(yValueArr[i]) - float(yValueArr[i - 1])
                    if (variance1 > 0):
                        type1 = "negative"
                    elif (variance1 < 0):
                        type1 = "positive"
                    else:
                        type1 = "neutral"
                    trendArray.append(type1)
                    i = i + 1
                # iterate through the variances and check for trends
                startIndex = 0
                trendLen = len(trendArray)
                # creates dictionary containing the trend length, direction, start and end indices, and the linear regression of the trend
                significanceRange = round(len(yValueArr) / 8)
                significantTrendCount = 0
                significantTrendArray = []
                for n in range(trendLen):
                    currentVal = trendArray[n - 1]
                    nextVal = trendArray[n]
                    if (currentVal != nextVal or (currentVal == nextVal and n == (trendLen - 1))):
                        if (n == (trendLen - 1)):
                            endIndex = n + 1
                        else:
                            endIndex = n
                        trendLength = endIndex - startIndex + 1
                        if trendLength > significanceRange:
                            xRange = pd.Series(numericXValueArr).loc[startIndex:endIndex]
                            yRange = pd.Series(yValueArr).loc[startIndex:endIndex]
                            result = linregress(xRange, yRange)
                            intercept = round(result[1], 2)
                            slope = round(result[0], 2)
                            trendRange = {"Length": (endIndex - startIndex + 1), "direction": currentVal,
                                          "start": startIndex, "end": endIndex, "slope": slope, "intercept":intercept}
                            significantTrendArray.append(trendRange)
                            significantTrendCount += 1
                            startIndex = n
                # sort the trend dictionaries by length
                if (significantTrendCount > 1):
                    # normalize trend slopes to get magnitudes for multi-trend charts
                    slopes = np.array([trend['slope'] for trend in significantTrendArray]).reshape(-1, 1)
                    scaler = preprocessing.MinMaxScaler()
                    scaler.fit(slopes)
                    scaledSlopes = scaler.transform(slopes)
                    print(significantTrendArray)
                    for trend, normalizedSlope in zip(significantTrendArray, scaledSlopes):
                        trend['magnitude'] = getMagnitude(normalizedSlope[0])
                    print(significantTrendArray)

                sortedTrends = sorted(significantTrendArray, key=lambda i: i['Length'], reverse=True)
                # generate the textual summary from the significant trend dictionary array at m
                if (significantTrendCount > 0):
                    startVal = str(xValueArr[(sortedTrends[0]['start'])])
                    endVal = str(xValueArr[(sortedTrends[0]['end'])])
                    direction = str(sortedTrends[0]['direction'])
                    if (significantTrendCount > 1):
                        magnitude = str(sortedTrends[0]['magnitude'])
                        m = 1
                        # execute here if more than 1 significant trend
                        similarSynonyms = ["Similarly", "Correspondingly", "Likewise", "Additionally", "Also",
                                           "In a similar manner"]
                        contrarySynonyms = ["Contrarily", "Differently", "On the other hand", "Conversely",
                                            "On the contrary",
                                            "But"]
                        extraTrends = ""
                        localTrendSentence1 = "This line chart has an x axis representing " + xLabel + " and a y axis representing " + yLabel + ", with a total of " + str(
                            len(yValueArr)) \
                                              + " data points. The chart has " + str(
                            significantTrendCount) + " significant trends."
                        summaryArray.append(localTrendSentence1)
                        graphTrendArray.append({})
                        localTrendSummary = " The longest trend is " + magnitude + " " + direction + " which exists from " + startVal + " to " + endVal + "."
                        summaryArray.append(localTrendSummary)
                        graphTrendArray.append({"1":str(xValueArr.index(startVal)), "12":str(xValueArr.index(endVal))})
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
                            startVal = str(xValueArr[(sortedTrends[m]['start'])])
                            endVal = str(xValueArr[(sortedTrends[m]['end'])])
                            direction = str(sortedTrends[m]['direction'])
                            magnitude = str(sortedTrends[m]['magnitude'])
                            extraTrends = " " + conjunction + " the next significant trend is " + magnitude + " " + direction + " which exists from " + startVal + " to " + endVal + "."
                            summaryArray.append(extraTrends)
                            graphTrendArray.append({"3":str(xValueArr.index(startVal)), "14":str(xValueArr.index(endVal))})
                            m = m + 1
                    # execute here if only 1 significant trend
                    else:
                        localTrendSentence1 = "This line chart has an x axis representing " + xLabel + " and a y axis representing " + yLabel + " with a total of " + str(
                            len(yValueArr)) + " data points. The chart has one significant trend."
                        summaryArray.append(localTrendSentence1)
                        graphTrendArray.append({})
                        localTrendSummary = " This trend is " + direction + " which exists from " + startVal + " to " + endVal + "."
                        summaryArray.append(localTrendSummary)
                        graphTrendArray.append({"3":str(xValueArr.index(startVal)), "14":str(xValueArr.index(endVal))})
                dataJson = [{xLabel: xVal, yLabel: yVal} for xVal, yVal in zip(cleanXArr, cleanYArr)]
                websiteInput = {"title": title, "xAxis": xLabel, "yAxis": yLabel,
                                "columnType": "two", \
                                "graphType": chartType, "summaryType": "baseline", "summary": summaryArray,
                                "trends": graphTrendArray,
                                "data": dataJson, "gold": gold}
                with open(f'{websitePath}/{count}.json', 'w', encoding='utf-8') as websiteFile:
                    json.dump(websiteInput, websiteFile, indent=3)
                oneFile.writelines(''.join(summaryArray) + '\n')
        # print(summaryArray)
