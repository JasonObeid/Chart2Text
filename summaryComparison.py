generatedPath = 'testOutput4.16.2.txt'
goldPath = 'data/test/testSummary.txt'
dataPath = 'data/test/testData.txt'
titlePath = 'data/test/testTitle.txt'
outputPath = 'summaryComparison.txt'
count = 0
import re


def are_numbers(stringList):
    try:
        for value in stringList:
            float(value)
        return True
    except ValueError:
        return False


def fixIndex(index, array):
    if are_numbers(array):
        array = [float(i) for i in array]
        if index == 'max':
            index = array.index(max(array))
            return int(index)
        elif index == 'min':
            index = array.index(min(array))
            return int(index)
    if index == 'last':
        index = len(array) - 1
        return index
    return int(index)


with open(goldPath, 'r', encoding='utf-8') as goldFile, open(generatedPath, 'r', encoding='utf-8') as generatedFile \
        , open(dataPath, 'r', encoding='utf-8') as dataFile, open(outputPath, 'w', encoding='utf-8') as outputFile, \
        open(titlePath, 'r', encoding='utf-8') as titleFile:
    for gold, generated, data, title in zip(goldFile.readlines(), generatedFile.readlines(), dataFile.readlines(),
                                            titleFile.readlines()):
        count += 1
        xValueArr = []
        yValueArr = []
        reversedArr = []
        datum = data.split()
        xLabel = datum[0].split('|')[0].split('_')
        yLabel = datum[1].split('|')[0].split('_')
        for i in range(0, len(datum)):
            if i % 2 == 0:
                xValueArr.append(datum[i].split('|')[1])
            else:
                yValueArr.append(datum[i].split('|')[1])
        for token in gold.split():
            if 'template' in token:
                index = str(re.search(r"\[(\w+)\]", token).group(0)).replace('[', '').replace(']', '')
                if 'templateTitle' in token:
                    index = fixIndex(index, title.split())
                    replacedToken = title.split()[index]
                elif 'templateXValue' in token:
                    index = fixIndex(index, xValueArr)
                    replacedToken = xValueArr[index]
                elif 'templateYValue' in token:
                    index = fixIndex(index, yValueArr)
                    replacedToken = yValueArr[index]
                elif 'templateXLabel' in token:
                    index = fixIndex(index, xLabel)
                    replacedToken = xLabel[index]
                elif 'templateYLabel' in token:
                    index = fixIndex(index, yLabel)
                    replacedToken = yLabel[index]
            else:
                replacedToken = token
            reversedArr.append(replacedToken)
        reverse = (' ').join(reversedArr)
        print(f'count: {count}\ndata: {data}title: {title}gold: {gold}reversed: {reverse}\ngenerated: {generated}')
        outputFile.write(f'count: {count}\ndata: {data}title: {title}gold: {gold}generated: {generated}\n')
