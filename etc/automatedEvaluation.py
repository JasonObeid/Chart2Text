import csv
import json
from statistics import mean, stdev
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
import sys
"""
def norm_dld(s1, s2):
    return 1.0-normalized_damerau_levenshtein_distance(s1, s2)


def calc_dld(predictedRecords, goldRecords):
    # calc dld for cs score
    """"""if len(predictedRecords) < len(goldRecords):
        delta = len(goldRecords) - len(predictedRecords)
        for i in range(delta):
            predictedRecords.append('')
    if len(predictedRecords) > len(goldRecords):
        delta = len(predictedRecords) - len(goldRecords)
        for i in range(delta):
            goldRecords.append('')""""""

    #assert len(predictedRecords) == len(goldRecords)

    if len(predictedRecords) > 0:
        sumDLD = 0
        for predictedRecord, goldRecord in zip(predictedRecords, goldRecords):
            sumDLD += norm_dld(predictedRecord, goldRecord)
        meanScore = sumDLD/len(predictedRecords)
        return meanScore
    else:
        return 0
"""

labelPath = '../data/test/testSummaryLabel.txt'
goldPath = '../data/test/testOriginalSummary.txt'
summaryPath = '../data/test/testSummary.txt'

generatedPath = '../results/aug17/generated/'
untemplatedPath = '../results/aug17/generated_untemplated/'
baselinePath = '../results/aug17/generated_baseline/'

fillers = ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', ':', '-',
           'to', 'but', 'is', 'of', 'it', 'on', '.', 'at', '(', ')', ',', ';']

count = 1

generatedScores = []
baselineScores = []
untemplatedScores = []

"""generatedDLDs = []
baselineDLDs = []
untemplatedDLDs = []"""

with open(labelPath, 'r', encoding='utf-8') as labelFile, open(summaryPath, 'r', encoding='utf-8') as summaryFile, \
        open(goldPath, 'r', encoding='utf-8') as goldFile:
    for lbls, summary, gold in zip(labelFile.readlines(), summaryFile.readlines(), goldFile.readlines()):
        labelArr = lbls.split()
        summArr = summary.split()
        goldArr = gold.split()
        recordList = []
        for lab, sums, gld in zip(labelArr, summArr, goldArr):
            if lab == '1' and gld.lower() not in fillers and gld.lower() not in recordList:
                recordList.append(gld.lower())
        list1 = recordList
        list2 = recordList
        list3 = recordList
        recordLength = len(recordList)
        generatedList = []
        with open(generatedPath + str(count) + '.json') as generatedFile:
            document1 = json.loads(generatedFile.read())
            summary1 = ''.join(document1['summary'])
        for token in summary1.split():
            if token.lower() in list1:
                list1.remove(token.lower())
                generatedList.append(token.lower())

        untemplatedList = []
        with open(untemplatedPath + str(count) + '.json') as untemplatedFile:
            document2 = json.loads(untemplatedFile.read())
            summary2 = ''.join(document2['summary'])
        for token in summary2.split():
            if token.lower() in list2:
                list2.remove(token.lower())
                untemplatedList.append(token.lower())

        """baselineList = []
        with open(baselinePath + str(count) + '.json') as baselineFile:
            document3 = json.loads(baselineFile.read())
            summary3 = ''.join(document3['summary'])
        for token in summary3.split():
            if token.lower() in list3:
                list3.remove(token.lower())
                baselineList.append(token.lower())"""
        count += 1

        generatedRatio = len(generatedList) / recordLength
        #baselineRatio = len(baselineList) / recordLength
        untemplatedRatio = len(untemplatedList) / recordLength

        generatedScores.append(generatedRatio)
        #baselineScores.append(baselineRatio)
        untemplatedScores.append(untemplatedRatio)

        """
        Cant calculate precision and recall with this method of detecting records
        generatedTP = sum([1 for item in generatedList if item in baselineList])
        generatedFP = sum([1 for item in generatedList if item not in baselineList])
        generatedFN = 

        #baselineRatio = len(baselineList) / recordLength
        #untemplatedRatio = len(untemplatedList) / recordLength

        generatedPrecision.append(generatedRatio)
        baselinePrecision.append(baselineRatio)
        untemplatedPrecision.append(untemplatedRatio)

        recall = truePositives / (truePositives + falseNegatives)
        precision = truePositives / (truePositives + falsePositives)
        
        generatedDLDs.append(calc_dld(generatedList, recordList))
        baselineDLDs.append(calc_dld(baselineList, recordList))
        untemplatedDLDs.append(calc_dld(untemplatedList, recordList))

print(f'generated CO: {round(mean(generatedDLDs)*100,2)}%')
print(f'baseline CO: {round(mean(baselineDLDs)*100,2)}%')
print(f'untemplated CO: {round(mean(untemplatedDLDs)*100,2)}%')

print('\n')
"""

"""print(f'generated CS stdev: {(stdev(generatedScores))}')
#print(f'baseline CS stdev: {(stdev(baselineScores))}')
print(f'baseline CS stdev: {(stdev(untemplatedScores))}')
print()
print(f'generated CS mean: {mean(generatedScores)}')
#print(f'baseline CS mean: {mean(baselineScores)}')
print(f'baseline CS mean: {mean(untemplatedScores)}')

print('\n')"""

print(f'generated CS stdev: {round(stdev(generatedScores)*100,2)}%')
#print(f'baseline CS stdev: {round(stdev(baselineScores)*100,2)}%')
print(f'baseline CS stdev: {round(stdev(untemplatedScores)*100,2)}%')
print()
print(f'generated CS mean: {round(mean(generatedScores)*100,2)}%')
#print(f'baseline CS mean: {round(mean(baselineScores)*100,2)}%')
print(f'baseline CS mean: {round(mean(untemplatedScores)*100,2)}%')
print()
print(f'generated CS RSD: {round((stdev(generatedScores)*100) / abs(mean(generatedScores)),2)}%')
#print(f'baseline CS RSD: {round((stdev(baselineScores)*100) / abs(mean(baselineScores)),2)}%')
print(f'baseline CS RSD: {round((stdev(untemplatedScores)*100) / abs(mean(untemplatedScores)),2)}%')

labels = ['generated CS stdev', 'generated CS mean', 'generated CS RSD', 'baseline CS stdev', 'baseline CS mean', 'baseline CS RSD']
values = [f'{round(stdev(generatedScores)*100,2)}%',
          f'{round(mean(generatedScores)*100,2)}%',
          f'{round((stdev(generatedScores)*100) / abs(mean(generatedScores)),2)}%',
          f'{round(stdev(untemplatedScores)*100,2)}%',
          f'{round(mean(untemplatedScores)*100,2)}%',
          f'{round((stdev(untemplatedScores)*100) / abs(mean(untemplatedScores)),2)}%']

with open('../results/automatedEvaluation.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(labels)
    csvwriter.writerow(values)