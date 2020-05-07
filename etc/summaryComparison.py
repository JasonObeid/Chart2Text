generatedPath = '../results/may 06/templateOutput_506_beam=4_batch=8.txt'
goldPath = '../data/test/testOriginalSummary.txt'
goldTemplatePath = '../data/test/testSummary.txt'
dataPath = '../data/test/testData.txt'
titlePath = '../data/test/testTitle.txt'
outputPath = '../results/may 06/summaryComparison506_beam4_batch8.txt'

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
            print('num err')
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
            print('num err')
            return 0
    if index == 'last':
        index = len(array) - 1
        return int(index)
    try:
        return int(index)
    except:
        print('num err')
        return 0


#def main():
count = 0
with open(goldPath, 'r', encoding='utf-8') as goldFile, open(generatedPath, 'r', encoding='utf-8') as generatedFile \
    , open(dataPath, 'r', encoding='utf-8') as dataFile, open(outputPath, 'w', encoding='utf-8') as outputFile, \
    open(titlePath, 'r', encoding='utf-8') as titleFile, open(goldTemplatePath, 'r', encoding='utf-8') as goldTemplateFile:
    fileIterators = zip(goldFile.readlines(), goldTemplateFile.readlines(),
                        generatedFile.readlines(), dataFile.readlines(), titleFile.readlines())
    for gold, goldTemplate, generated, data, title in fileIterators:
        count += 1
        xValueArr = []
        yValueArr = []
        reversedArr = []
        datum = data.split()
        xLabel = datum[0].split('|')[0].split('_')
        yLabel = datum[1].split('|')[0].split('_')
        fillers = ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', ':', '-',
                   'to', 'but', 'is', 'of', 'it', 'on', '.', 'at', '(', ')', ',', 'with']
        # remove filler words from labels
        cleanXLabel = [xWord for xWord in xLabel if xWord.lower() not in fillers]
        cleanYLabel = [yWord for yWord in yLabel if yWord.lower() not in fillers]
        for i in range(0, len(datum)):
            if i % 2 == 0:
                xValueArr.append(datum[i].split('|')[1])
            else:
                yValueArr.append(datum[i].split('|')[1])
        tokens = generated.split()
        for token, i in zip(tokens, range(0, len(tokens))):
            if i < (len(tokens) - 1):
                if tokens[i] == tokens[i + 1]:
                    print(f'1:{tokens[i]} 2:{tokens[i + 1]}')
                    print(f'popped: {tokens[i + 1]}')
                    tokens.pop(i + 1)
            if 'template' in token:
                index = str(re.search(r"\[(\w+)\]", token).group(0)).replace('[', '').replace(']', '')
                if 'templateTitle' in token:
                    titleArr = [word for word in title.split() if word.lower() not in fillers]
                    index = mapIndex(index, titleArr)
                    try:
                        replacedToken = titleArr[index]
                    except:
                        replacedToken = ''#'titleErr'# titleArr[len(titleArr) - 1]
                elif 'templateXValue' in token:
                    index = mapIndex(index, xValueArr)
                    try:
                        replacedToken = xValueArr[index]
                    except:
                        replacedToken = ''#'xValErr'# xValueArr[len(xValueArr) - 1]
                elif 'templateYValue' in token:
                    index = mapIndex(index, yValueArr)
                    try:
                        replacedToken = yValueArr[index]
                    except:
                        replacedToken = ''#'yValErr'# yValueArr[len(yValueArr) - 1]
                elif 'templateXLabel' in token:
                    index = mapIndex(index, cleanXLabel)
                    try:
                        replacedToken = cleanXLabel[index]
                    except:
                        replacedToken = ''#'xLabelErr'# cleanXLabel[len(cleanXLabel) - 1]
                elif 'templateYLabel' in token:
                    index = mapIndex(index, cleanYLabel)
                    try:
                        replacedToken = cleanYLabel[index]
                    except:
                        replacedToken = ''#'yLabelErr'# cleanYLabel[len(cleanYLabel) - 1]
            else:
                replacedToken = token
            reversedArr.append(replacedToken)
        reverse = (' ').join(reversedArr)
        print(f'Example {count}:\ndata: {data}title: {title}\ngold: {gold}gold_template: {goldTemplate}\ngenerated_template: {generated}generated: {reverse}\n\n')
        outputFile.write(
            f'Example {count}:\ndata: {data}title: {title}\ngold: {gold}gold_template: {goldTemplate}\ngenerated_template: {generated}generated: {reverse}\n\n\n')


# try:
#main()
# except Exception as ex:
#    print('error:', ex)

""" elif are_numbers(array[0:len(array)-2]):
    array = [float(i) for i in array[0:len(array)-2]]
    if str(index) == 'max':
        index = array.index(max(array))
        return int(index)
    elif str(index) == 'min':
        index = array.index(min(array))
        return int(index)    """
