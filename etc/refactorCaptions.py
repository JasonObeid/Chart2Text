import scripts.tokenizer as tkn
import os

summaryPaths = os.listdir('../dataset/captions_old/')

def groupSequence(lst):
        res = [[lst[0]]]
        for i in range(1, len(lst)):
                if lst[i - 1] + 1 == lst[i]:
                        res[-1].append(lst[i])
                else:
                        res.append([lst[i]])
        return res


def addUnderscores(cleanCaption, uppercaseWords):
        newCaption = cleanCaption
        if len(uppercaseWords) > 1:
                indices = [cleanCaption.index(word) for word in uppercaseWords]
                consecutives = groupSequence(indices)
                tokensToPop = []
                tokensToUpdate = {}
                print(consecutives)
                for consecutive in consecutives:
                        if len(consecutive) > 1:
                                consecutiveTokens = [cleanCaption[i] for i in consecutive]
                                newToken = '_'.join(consecutiveTokens)
                                firstIndex = consecutive.pop(0)
                                tokensToUpdate[firstIndex] = newToken
                                tokensToPop.append([consecutive[0], consecutive[-1] + 1])
                for index, value in zip(tokensToUpdate.keys(), tokensToUpdate.values()):
                        newCaption[index] = value
                length = len(tokensToPop)
                for i in range(0, length):
                        x = tokensToPop.pop()
                        start = x[0]
                        end = x[1]
                        del newCaption[start:end]
        return newCaption

for summaryPath in summaryPaths:
        with open('../dataset/captions_old/'+summaryPath, 'r', encoding='utf-8') as summaryFile:
                summary = summaryFile.read()
                cleanCaption = tkn.word_tokenize(summary)
                #for token, i in zip(cleanCaption, range(0,len(cleanCaption))):
                        #remove anything in brackets
                        #if i < len(cleanCaption) - 2:
                        #        if cleanCaption[i] == '(' and cleanCaption[i+1].isalnum() and cleanCaption[i+2] == ')':
                        #                print(cleanCaption)
                        #                del cleanCaption[i:i+3]
                        #                print(cleanCaption)
                                        #print(f'found brackets at {cleanCaption[i], cleanCaption[i+1], cleanCaption[i+2]}')
                newSummaryPath = '../dataset/captions/'+summaryPath
                with open(newSummaryPath, "w") as outf:
                    cleanCaption = ' '.join(cleanCaption).replace('*','')
                    outf.write("{}\n".format(cleanCaption))
                outf.close()
                #newSentences = []
                #for sentence in ' '.join(cleanCaption).split(' . '):
                #        sent = sentence.split()
                #        uppercaseWords = [word for word in sent if word[0].isupper()]
                #        newSentence = ' '.join(addUnderscores(sent, uppercaseWords))
                #        newSentences.append(newSentence)
                #print(newSentences)
                #newCaption = ' . '.join(newSentences)
                #with open(newSummaryPath, "w") as outf:
                #    outf.write("{}\n".format(newCaption))
                #outf.close()
                #print(newCaption)