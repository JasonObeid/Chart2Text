labelPath = '../data/test/testDataLabel.txt'
summaryPath = '../data/test/testOriginalSummary.txt'
dataPath = '../data/test/testData.txt'
count = 0


with open(labelPath, 'r', encoding='utf-8') as labelFile, open(summaryPath, 'r', encoding='utf-8') as summaryFile, \
        open(dataPath, 'r', encoding='utf-8') as dataFile:
    for label, summary, data in zip(labelFile.readlines(), summaryFile.readlines(), dataFile.readlines()):
        xValueArr = []
        yValueArr = []
        xLabelArr = []
        yLabelArr = []
        print(f'{count}: {summary}')
        assert len(label.split(' ')) == len(data.split())
        datum = data.split()
        labels = label.split()
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
