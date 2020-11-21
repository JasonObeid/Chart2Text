import re
from typing import List
import nltk
import os

summaryPaths = os.listdir('../dataset/captions_old/')
def word_tokenize(string: str, language: str = "english") -> List[str]:
    """tokenizes a given string into a list of substrings.

    :param string: String to tokenize.
    :param language: Language. Either one of ``english'' or ``german''.
    """
    if language not in ["english", "german"]:
        raise ValueError("language argument has to be either ``english'' or ``german''")

    # excessive whitespaces
    string = re.sub(r"\s+", " ", string)

    # some unicode characters
    string = string.replace("’", "'")
    string = string.replace("”", '"')
    string = string.replace("“", '"')

    # floating point (e.g., 1.3 => 1.3)
    string = re.sub(r"(\d+)\.(\d+)", r"\g<1>._\g<2>", string)

    # percentage (e.g., below.500 => below .500)
    string = re.sub(r"(\w+)\.(\d+)", r"\g<1> ._\g<2>", string)

    # end of quote
    string = string.replace(".``", ". ``")

    # number with apostrophe (e.g. '90)
    string = re.sub(r"\s'(\d+)", r"' \g<1>", string)

    # names with Initials (e.g. C. J. Miles)
    string = re.sub(r"(^|\s)(\w)\. (\w)\.", r"\g<1>\g<2>._ \g<3>._", string)

    # some dots
    string = string.replace("..", " ..")

    # names with apostrophe => expands temporarily
    string = re.sub(r"\w+'(?!d|s|ll|t|re|ve|\s)", r"\g<0>_", string)

    # win-loss scores (German notation seems to be XX:YY, but this is also the time format,
    # and the times are not tokenized in the original RotoWire. So we manually handle XX:YY
    # expression.
    string = re.sub(r"(\d+)-(\d+)", r"\g<1> - \g<2>", string)

    # actual tokenization
    tokenized = nltk.word_tokenize(string, language=language)

    joined = " ".join(tokenized)
    # shrink expanded name-with-apostrophe expressions
    joined = joined.replace("'_", "'")
    # shrink expanded name-with-initial expressions
    joined = joined.replace("._", ".")
    tokenized = joined.split(" ")

    return tokenized


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
                cleanCaption = word_tokenize(summary)
                newSummaryPath = '../dataset/captions/'+summaryPath
                with open(newSummaryPath, "w", encoding='utf-8') as outf:
                    cleanCaption = ' '.join(cleanCaption).replace('*','')
                    outf.write("{}\n".format(cleanCaption))
                outf.close()
