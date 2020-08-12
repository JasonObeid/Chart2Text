labelPath = '../data/test/testSummaryLabel.txt'
goldPath = '../data/test/testOriginalSummary.txt'
summaryPath = '../data/test/testSummary.txt'
count = 0


with open(labelPath, 'r', encoding='utf-8') as labelFile, open(summaryPath, 'r', encoding='utf-8') as summaryFile, \
        open(goldPath, 'r', encoding='utf-8') as goldFile:
    for lbls, summary, gold in zip(labelFile.readlines(), summaryFile.readlines(), goldFile.readlines()):
        labelArr = lbls.split()
        summArr = summary.split()
        goldArr = gold.split()
        for lab, sums, gld in zip(labelArr, summArr, goldArr):
            if lab == '1':
                print(sums, gld)