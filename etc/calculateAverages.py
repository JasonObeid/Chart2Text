import os

punctuation = [';',':', '-', '(', ')', ',']

captionList = os.listdir('../dataset/captions')
complexCaptionList = os.listdir('../dataset/multiColumn/captions')

captionList.sort()
complexCaptionList.sort()

captionCount = 0
wordCountList = []
sentenceCountList = []

for filePath in captionList:
    with open('../dataset/captions/'+filePath,'r') as captionFile:
        caption = captionFile.read()
        captionSentences = caption.split(' . ')
        sentenceCount = len(captionSentences)
        sentenceCountList.append(sentenceCount)
        captionWords = [word for word in caption.split() if word not in punctuation]
        wordCount = len(captionWords)
        wordCountList.append(wordCount)
        captionCount += 1

for filePath in complexCaptionList:
    with open('../dataset/multiColumn/captions/'+filePath,'r') as captionFile:
        captionSentences = caption.split(' . ')
        sentenceCount = len(captionSentences)
        sentenceCountList.append(sentenceCount)
        captionWords = [word for word in caption.split() if word not in punctuation]
        wordCount = len(captionWords)
        wordCountList.append(wordCount)
        captionCount += 1

averageWords = sum(wordCountList)/captionCount
print(averageWords)

averageSentences = sum(sentenceCountList)/captionCount
print(averageSentences)