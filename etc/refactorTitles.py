import scripts.tokenizer as tkn
import os
# re.findall('([A-Z][a-z]+)', ' '.join(cleanTitle))
# remainingWords = cleanTitle.split()[1:]
# if 'U.S.' in cleanTitle:
# uppercaseWords.append('U.S.')

# if len(uppercaseWords) == 1:
#        if uppercaseWords[0] == firstWord:
#                x = 0
# print(f'first word is the only uppercase: {uppercaseWords} from {cleanTitle}')
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
                uppercaseWords = [word for word in cleanTitle if word[0].isupper()]
                newTitle = addUnderscores(cleanTitle, uppercaseWords)
                newTitlePath = '../dataset/titles/'+summaryPath
                with open(newTitlePath, "w") as outf:
                    outf.write("{}\n".format(' '.join(newTitle)))
                outf.close()

        """
        print(uppercaseWords)
        if indices[i - 1] == indices[i] - 1:
                index1 = indices[i - 1]
                index2 = indices[i]
                word1 = cleanTitle[index1]
                word2 = cleanTitle[index2]
                print('sequence found')
                print(index1, index2)
                print(word1, word2)
                print(indices)
                #print(uppercaseWords)
                print(cleanTitle)
                cleanTitle[index2] = f'{word1}_{word2}'
                cleanTitle.pop(index1)
                #indices.pop(i)
                print(cleanTitle)
#print(indices)
#print(cleanTitle)
#print(uppercaseWords)
"""