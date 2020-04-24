generatedPath = 'testOutput4.16.2.txt'
goldPath = 'data/test/testSummary.txt'
dataPath = 'data/test/testData.txt'
titlePath = 'data/test/testTitle.txt'
outputPath = 'summaryComparison.txt'
count = 0
with open(goldPath, 'r', encoding='utf-8') as goldFile, open(generatedPath, 'r', encoding='utf-8') as generatedFile\
        , open(dataPath, 'r', encoding='utf-8') as dataFile, open(outputPath, 'w', encoding='utf-8') as outputFile,\
        open(titlePath, 'r', encoding='utf-8') as titleFile:
    for gold, generated, data, title in zip(goldFile.readlines(), generatedFile.readlines(), dataFile.readlines(), titleFile.readlines()):
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
            if 'templateTitle' in token:
                index = token[len(token)-2:len(token)-1]
                replacedToken = title.split()[int(index)]
            elif 'templateXvalue' in token:
                index = int(token[len(token)-2:len(token)-1])
                replacedToken = xValueArr[index]
            elif 'templateYvalue' in token:
                index = int(token[len(token) - 2:len(token) - 1])
                replacedToken = yValueArr[index]
            elif 'templateXlabel' in token:
                index = int(token[len(token) - 2:len(token) - 1])
                replacedToken = xLabel[index]
            elif 'templateYlabel' in token:
                index = int(token[len(token) - 2:len(token) - 1])
                replacedToken = yLabel[index]
            else:
                replacedToken = token
            reversedArr.append(replacedToken)
        reverse = (' ').join(reversedArr)
        print(f'count: {count}\ndata: {data}title: {title}gold: {gold}reversed: {reverse}\ngenerated: {generated}')
        outputFile.write(f'count: {count}\ndata: {data}title: {title}gold: {gold}generated: {generated}\n')
