labelPath = '../data/test/testDataLabel.txt'
summaryPath = '../data/test/testOriginalSummary.txt'
dataPath = '../data/test/testData.txt'
count = 0


with open(labelPath, 'r', encoding='utf-8') as labelFile, open(summaryPath, 'r', encoding='utf-8') as summaryFile, \
        open(dataPath, 'r', encoding='utf-8') as dataFile:
    for lbls, summary, data in zip(labelFile.readlines(), summaryFile.readlines(), dataFile.readlines()):
        """
        columnType = datum[0].split('|')[2].isnumeric()
        if columnType:
            labelArr = []
            cleanLabelArr = []
            chartType = datum[0].split('|')[3].split('_')[0]
            values = [value.split('|')[1] for value in datum]
            # find number of columns:
            columnCount = max([int(data.split('|')[2]) for data in datum]) + 1
            # Get labels
            for i in range(columnCount):
                label = datum[i].split('|')[0].split('_')
                cleanLabel = [word for word in label if word.lower() not in fillers]
                labelArr.append(label)
                cleanLabelArr.append(cleanLabel)
            stringLabels = [' '.join(label) for label in labelArr]
            # Get values
            valueArr = [[] for i in range(columnCount)]
            cleanValArr = [[] for i in range(columnCount)]
            rowCount = round(len(datum) / columnCount)
            i = 0
            for n in range(rowCount):
                for m in range(columnCount):
                    value = values[i]
                    # print(f'col {n} row {m}: {value}')
                    cleanVal = datum[i].split('|')[1].replace('_', ' ')
                    valueArr[m].append(value)
                    cleanValArr[m].append(cleanVal)
                    i += 1
            for value, label in zip(valueArr, labelArr):
                
        else:"""
        xValueArr = []
        yValueArr = []
        xLabelArr = []
        yLabelArr = []
        print(f'{count}: {summary}')
        assert len(lbls.split(' ')) == len(data.split())
        datum = data.split()
        labels = lbls.split()
        for i in range(0, len(datum)):
            if i % 2 == 0:
                xLabelArr.append(labels[i])
                xValueArr.append(datum[i].split('|')[1])
            else:
                yLabelArr.append(labels[i])
                yValueArr.append(datum[i].split('|')[1])
        for x, y, xLabel, yLabel in zip(xValueArr,yValueArr,xLabelArr,yLabelArr):
            if xLabel == '1':
                print(f'x:{x, xLabel}')
            if yLabel == '1':
                print(f'y:{y, yLabel}')
        count += 1
        print('\n')
