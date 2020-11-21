import utils.tokenizer as tkn
import os

def groupSequence(lst):
        res = [[lst[0]]]
        for i in range(1, len(lst)):
                if lst[i - 1] + 1 == lst[i]:
                        res[-1].append(lst[i])
                else:
                        res.append([lst[i]])
        return res


def addUnderscores(cleanTitle, uppercaseWords):
        newTitle = cleanTitle
        if len(uppercaseWords) > 1:
                indices = [cleanTitle.index(word) for word in uppercaseWords]
                consecutives = groupSequence(indices)
                tokensToPop = []
                tokensToUpdate = {}
                for consecutive in consecutives:
                        if len(consecutive) > 1:
                                consecutiveTokens = [cleanTitle[i] for i in consecutive]
                                newToken = '_'.join(consecutiveTokens)
                                firstIndex = consecutive.pop(0)
                                tokensToUpdate[firstIndex] = newToken
                                tokensToPop.append([consecutive[0], consecutive[-1] + 1])
                for index, value in zip(tokensToUpdate.keys(), tokensToUpdate.values()):
                        newTitle[index] = value
                length = len(tokensToPop)
                for i in range(0, length):
                        x = tokensToPop.pop()
                        start = x[0]
                        end = x[1]
                        del newTitle[start:end]
        return newTitle


titlePaths = os.listdir('../dataset/titles_old/')
titlePaths.sort()

for summaryPath in titlePaths:
        with open('../dataset/titles_old/'+summaryPath, 'r', encoding='utf-8') as titleFile:
                title = titleFile.read()
                cleanTitle = tkn.word_tokenize(title)
                # replace (2009 - 2016) with (2009 to 2016)
                lastTokens = cleanTitle[-3:]
                if lastTokens[1] == '-' and lastTokens[0].isnumeric() and lastTokens[2].isnumeric():
                        cleanTitle[-2] = 'to'
                newTitlePath = '../dataset/titles/' + summaryPath
                with open(newTitlePath, "w", encoding='utf-8') as outf:
                        # remove asterisks
                        cleanTitle = ' '.join(cleanTitle).replace('*', '')
                        outf.write("{}\n".format(cleanTitle))
                outf.close()
                # uppercaseWords = [word for word in cleanTitle if word[0].isupper()]
                # newTitle = addUnderscores(cleanTitle, uppercaseWords)
                # with open(newTitlePath, "w") as outf:
                #    outf.write("{}\n".format(' '.join(newTitle)))
                # outf.close()
