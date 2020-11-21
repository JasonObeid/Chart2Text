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
captionPath = '../results/aug17/templateOutput_untemplated-p80.txt'
websitePath = '../results/aug17/generated_untemplated'

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

count = 0
with open(goldPath, 'r', encoding='utf-8') as goldFile, open(dataPath, 'r', encoding='utf-8') as dataFile, \
        open(titlePath, 'r', encoding='utf-8') as titleFile, open(captionPath, 'r', encoding='utf-8') as captionFile:
    # assert len(goldFile.readlines()) == len(dataFile.readlines()) == len(titleFile.readlines())
    fileIterators = zip(goldFile.readlines(), dataFile.readlines(), titleFile.readlines(), captionFile.readlines())
    for gold, data, title, caption in fileIterators:
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
                websiteInput = {"title": title.strip(), "labels": [' '.join(label) for label in labelArr],
                                "columnType": "multi", "graphType": chartType, "summaryType": "nlp", "summary": caption,
                                "trends": {}, "data": dataJson, "gold": gold}
                with open(f'{websitePath}/{count}.json', 'w', encoding='utf-8') as websiteFile:
                    json.dump(websiteInput, websiteFile, indent=3)
            elif (chartType == "line"):
                websiteInput = {"title": title.strip(), "labels": [' '.join(label) for label in labelArr],
                                "columnType": "multi", "graphType": chartType, "summaryType": "nlp", "summary": caption,
                                "trends": {}, "data": dataJson, "gold": gold}
                with open(f'{websitePath}/{count}.json', 'w', encoding='utf-8') as websiteFile:
                    json.dump(websiteInput, websiteFile, indent=3)
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
            if (chartType == "bar"):
                dataJson = [{xLabel: xVal, yLabel: yVal} for xVal, yVal in zip(cleanXArr, cleanYArr)]
                websiteInput = {"title": title, "xAxis": xLabel, "yAxis": yLabel,
                                "columnType": "two", "graphType": chartType, "summaryType": "baseline", "summary": caption,
                                "trends": {}, "data": dataJson, "gold": gold}
                with open(f'{websitePath}/{count}.json', 'w', encoding='utf-8') as websiteFile:
                    json.dump(websiteInput, websiteFile, indent=3)
            # run line
            elif (chartType == "line"):
                dataJson = [{xLabel: xVal, yLabel: yVal} for xVal, yVal in zip(cleanXArr, cleanYArr)]
                websiteInput = {"title": title, "xAxis": xLabel, "yAxis": yLabel,
                                "columnType": "two", "graphType": chartType, "summaryType": "baseline",
                                "summary": caption,
                                "trends": {}, "data": dataJson, "gold": gold}
                with open(f'{websitePath}/{count}.json', 'w', encoding='utf-8') as websiteFile:
                    json.dump(websiteInput, websiteFile, indent=3)
        # print(summaryArray)
