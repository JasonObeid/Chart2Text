import json
from statistics import mean

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
        recordLength = len(recordList)
        generatedList = []
        with open(generatedPath + str(count) + '.json') as generatedFile:
            document1 = json.loads(generatedFile.read())
            summary1 = ''.join(document1['summary'])
        for token in summary1.split():
            if token.lower() in recordList:
                recordList.remove(token.lower())
                generatedList.append(token.lower())

        untemplatedList = []
        with open(untemplatedPath + str(count) + '.json') as untemplatedFile:
            document2 = json.loads(untemplatedFile.read())
            summary2 = ''.join(document2['summary'])
        for token in summary2.split():
            if token.lower() in recordList:
                recordList.remove(token.lower())
                untemplatedList.append(token.lower())

        baseLineList = []
        with open(baselinePath + str(count) + '.json') as baselineFile:
            document3 = json.loads(baselineFile.read())
            summary3 = ''.join(document3['summary'])
        for token in summary3.split():
            if token.lower() in recordList:
                recordList.remove(token.lower())
                baseLineList.append(token.lower())
        count += 1

        generatedRatio = len(generatedList) / recordLength
        baselineRatio = len(baseLineList) / recordLength
        untemplatedRatio = len(untemplatedList) / recordLength

        generatedScores.append(generatedRatio)
        baselineScores.append(baselineRatio)
        untemplatedScores.append(untemplatedRatio)

print(f'generated {round(mean(generatedScores),4)*100}%')
print(f'baseline: {round(mean(baselineScores),4)*100}%')
print(f'untemplated: {round(mean(untemplatedScores),4)*100}%')