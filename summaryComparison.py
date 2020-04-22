generatedPath = 'testOutput2.txt'
goldPath = 'data/test/testSummary.txt'
dataPath = 'data/test/testData.txt'
count = 0
with open(goldPath, 'r', encoding='utf-8') as goldFile, open(generatedPath, 'r', encoding='utf-8') as generatedFile\
        , open(dataPath, 'r', encoding='utf-8') as dataFile:
    for gold, generated, data in zip(goldFile.readlines(), generatedFile.readlines(), dataFile.readlines()):
        count += 1
        print(f'count: {count}\ndata: {data}gold: {gold}generated: {generated}')
