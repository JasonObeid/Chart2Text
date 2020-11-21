import os
from statistics import mean

import pandas as pd

punctuation = [';',':', '-', '(', ')', ',']

captionList = os.listdir('../dataset/captions')
complexCaptionList = os.listdir('../dataset/multiColumn/captions')

captionList.sort()
complexCaptionList.sort()

dataList = os.listdir('../dataset/multiColumn/data/')
complexDataList = os.listdir('../dataset/data/')

dataList.sort()
complexDataList.sort()

captionCount = 0
wordCountList = []
sentenceCountList = []

cellCount = 0

vocab = []
for filePath in captionList:
    with open('../dataset/captions/'+filePath,'r') as captionFile:
        caption = captionFile.read()
        captionSentences = caption.split(' . ')
        sentenceCount = len(captionSentences)
        sentenceCountList.append(sentenceCount)
        captionWords = [word for word in caption.split() if word not in punctuation]
        for token in captionWords:
            if token.lower() not in vocab and token.isalpha():
                vocab.append(token.lower())
        wordCount = len(captionWords)
        wordCountList.append(wordCount)
        captionCount += 1

for filePath in complexCaptionList:
    with open('../dataset/multiColumn/captions/'+filePath,'r') as captionFile:
        caption = captionFile.read()
        captionSentences = caption.split(' . ')
        sentenceCount = len(captionSentences)
        sentenceCountList.append(sentenceCount)
        captionWords = [word for word in caption.split() if word not in punctuation]
        for token in captionWords:
            if token.lower() not in vocab and token.isalpha():
                vocab.append(token.lower())
        wordCount = len(captionWords)
        wordCountList.append(wordCount)
        captionCount += 1

averageWords = sum(wordCountList)/captionCount
print(averageWords)

averageSentences = sum(sentenceCountList)/captionCount
print(averageSentences)

print(len(vocab))
print(sum(wordCountList))

for dataPath in dataList:
    df = pd.read_csv('../dataset/multiColumn/data/' + dataPath)
    cellCount += (df.size - len(df.columns))

for complexDataPath in complexDataList:
    df = pd.read_csv('../dataset/data/' + complexDataPath)
    cellCount += (df.size - len(df.columns))

print(cellCount)

with open('../results/aug17/generated-p80.txt', encoding='utf-8') as templated, \
open('../results/aug17/generated_untemplated-p80.txt', encoding='utf-8') as untemplated, \
open('../results/aug17/generated_baseline.txt', encoding='utf-8') as baseline:
    tmp = templated.readlines()
    untmp = untemplated.readlines()
    base = baseline.readlines()
    templatedSentences = mean([len(line.split(' . ')) for line in tmp])
    templatedTokens = mean([len(line.split(' ')) for line in tmp])
    untemplatedSentences = mean([len(line.split(' . ')) for line in untmp])
    untemplatedTokens = mean([len(line.split(' ')) for line in untmp])
    baselineSentences = mean([len(line.split('. ')) for line in base])
    baselineTokens = mean([len(line.split(' ')) for line in base])
    print(f'templated: {templatedSentences}, {templatedTokens}')
    print(f'untemplated: {untemplatedSentences}, {untemplatedTokens}')
    print(f'baseline: {baselineSentences}, {baselineTokens}')