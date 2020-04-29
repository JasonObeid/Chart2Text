generatedPath = 'templateOutput.429.1.txt'
goldPath = 'data/test/testOriginalSummary.txt'
dataPath = 'data/test/testData.txt'
titlePath = 'data/test/testTitle.txt'
outputPath = 'summaryComparison.txt'

import re


def are_numbers(stringList):
    try:
        for value in stringList:
            float(value)
        return True
    except ValueError:
        return False


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
            return 0
    elif are_numbers(array[0:len(array)-1]):
        try:
            array = [float(i) for i in array[0:len(array)-1]]
            if str(index) == 'max':
                index = array.index(max(array))
                return int(index)
            elif str(index) == 'min':
                index = array.index(min(array))
                return int(index)
        except:
            return 0
    if index == 'last':
        index = len(array) - 1
        return int(index)
    try:
        return int(index)
    except:
        return 0


def main():
    count = 0
    with open(goldPath, 'r', encoding='utf-8') as goldFile, open(generatedPath, 'r', encoding='utf-8') as generatedFile \
            , open(dataPath, 'r', encoding='utf-8') as dataFile, open(outputPath, 'w', encoding='utf-8') as outputFile, \
            open(titlePath, 'r', encoding='utf-8') as titleFile:
        for gold, generated, data, title in zip(goldFile.readlines(), generatedFile.readlines(),
                                                dataFile.readlines(), titleFile.readlines()):
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
            for token in generated.split():
                if 'template' in token:
                    index = str(re.search(r"\[(\w+)\]", token).group(0)).replace('[', '').replace(']', '')
                    if 'templateTitle' in token:
                        index = mapIndex(index, title.split())
                        try:
                            replacedToken = title.split()[index]
                        except:
                            replacedToken = 'N/A'
                    elif 'templateXValue' in token:
                        index = mapIndex(index, xValueArr)
                        try:
                            replacedToken = xValueArr[index]
                        except:
                            replacedToken = 'N/A'
                    elif 'templateYValue' in token:
                        index = mapIndex(index, yValueArr)
                        try:
                            replacedToken = yValueArr[index]
                        except:
                            replacedToken = 'N/A'
                    elif 'templateXLabel' in token:
                        index = mapIndex(index, xLabel)
                        try:
                            replacedToken = xLabel[index]
                        except:
                            replacedToken = 'N/A'
                    elif 'templateYLabel' in token:
                        index = mapIndex(index, yLabel)
                        try:
                            replacedToken = yLabel[index]
                        except:
                            replacedToken = 'N/A'
                else:
                    replacedToken = token
                reversedArr.append(replacedToken)
            reverse = (' ').join(reversedArr)
            print(f'Example {count}:\ndata: {data}title: {title}gold: {gold}generated: {reverse}\n\n')
            outputFile.write(f'Example {count}:\ndata: {data}title: {title}gold: {gold}generated: {reverse}\n\n')


#try:
main()
#except Exception as ex:
#    print('error:', ex)

""" elif are_numbers(array[0:len(array)-2]):
    array = [float(i) for i in array[0:len(array)-2]]
    if str(index) == 'max':
        index = array.index(max(array))
        return int(index)
    elif str(index) == 'min':
        index = array.index(min(array))
        return int(index)    """