import os

captionList = os.listdir('captions')
complexCaptionList = os.listdir('multiColumn/captions')
captionCount = 0
wordCountList = []
sentenceCountList = []
vocab = []
for filePath in captionList:
    with open('captions/'+filePath,'r') as caption:
        captionSentences = caption.read().split(' . ')
        sentenceCount = len(captionSentences)
        sentenceCountList.append(sentenceCount)
        captionWords = caption.read().split()
        for token in captionWords:
            if token.lower() not in vocab:
                vocab.append(token.lower())
        wordCount = len(captionWords)
        wordCountList.append(wordCount)
        captionCount += 1

for filePath in complexCaptionList:
    with open('multiColumn/captions/'+filePath,'r') as caption:
        captionSentences = caption.read().split(' . ')
        sentenceCount = len(captionSentences)
        sentenceCountList.append(sentenceCount)
        captionWords = caption.read().split()
        for token in captionWords:
            if token.lower() not in vocab:
                vocab.append(token.lower())
        wordCount = len(captionWords)
        wordCountList.append(wordCount)
        captionCount += 1

averageWords = sum(wordCountList)/captionCount
print(averageWords)