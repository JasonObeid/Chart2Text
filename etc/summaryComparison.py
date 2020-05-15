import spacy
import en_core_web_md
import re

generatedPath = '../results/may13/templateOutput_513p80_beam=4_batch=8.txt'
goldPath = '../data/test/testOriginalSummary.txt'
goldTemplatePath = '../data/test/testSummary.txt'
dataPath = '../data/test/testData.txt'
titlePath = '../data/test/testTitle.txt'
comparisonPath = '../results/may13/summaryComparison513-p80_beam4_batch8.txt'
outputPath = '../results/may13/generated-513-p80.txt'

nlp = spacy.load('en_core_web_md')
from fitbert import FitBert

# currently supported models: bert-large-uncased and distilbert-base-uncased
# this takes a while and loads a whole big BERT into memory
fb = FitBert()


def askBert(masked_string, options):
    ranked_options = fb.rank(masked_string, options)
    print(ranked_options)
    return ranked_options[0]


def getNamedEntity(title, xValueArr):
    doc = nlp(title)
    entities = {}
    entities['Subject'] = []
    entities['Date'] = []
    for X in doc.ents:
        if X.label_ == 'GPE' or X.label_ == 'ORG' or X.label_ == 'NORP' or X.label_ == 'LOC':
            cleanSubject = [word for word in X.text.split() if word.isalpha() and word not in fillers]
            if len(cleanSubject) > 0:
                entities['Subject'].append(' '.join(cleanSubject))
        elif X.label_ == 'DATE':
            if X.text.isnumeric():
                entities['Date'].append(X.text)
    if len(entities['Subject']) == 0:
        uppercaseWords = [word for word in title.split() if word[0].isupper()]
        if len(uppercaseWords) > 1:
            guessedSubject = ' '.join(uppercaseWords[1:])
        else:
            guessedSubject = uppercaseWords[0]
        entities['Subject'].append(guessedSubject)
    if len(entities['Date']) == 0:
        if xLabel[0] == 'Month':
            years = []
            for item in xValueArr:
                [years.append(year.replace('\'', '')) for year in item.split('_') if year.replace('\'', '').isnumeric()]
            if len(years) > 0:
                entities['Date'] = [min(years), max(years)]
        elif xLabel[0] == 'Year':
            years = [year for year in title.split() if year.replace('/', '').isnumeric()]
            if len(years) > 0:
                entities['Date'] = [min(years), max(years)]
        else:
            try:
                years = [year for year in title.split() if year.isnumeric()]
                if len(years) > 0:
                    entities['Date'] = [min(years), max(years)]
            except:
                p = 0
    return entities


def are_numbers(stringList):
    try:
        for value in stringList:
            float(value)
        return True
    except ValueError:
        return False


def mapIndex(index, array):
    if are_numbers(array):
        try:
            array = [float(i) for i in array]
            if str(index) == 'max':
                index = array.index(max(array))
                return int(index)
            elif str(index) == 'min':
                index = array.index(min(array))
                return int(index)
        except:
            print('num err')
            return 0
    elif are_numbers(array[0:len(array) - 1]):
        try:
            array = [float(i) for i in array[0:len(array) - 1]]
            if str(index) == 'max':
                index = array.index(max(array))
                return int(index)
            elif str(index) == 'min':
                index = array.index(min(array))
                return int(index)
        except:
            print('num err')
            return 0
    if index == 'last':
        index = len(array) - 1
        return int(index)
    try:
        return int(index)
    except:
        print('num err')
        return 0


#def main():
fillers = ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', ':', '-',
           'to', 'but', 'is', 'of', 'it', 'on', '.', 'at', '(', ')', ',', 'with']
count = 0
with open(goldPath, 'r', encoding='utf-8') as goldFile, open(generatedPath, 'r', encoding='utf-8') as generatedFile \
    , open(dataPath, 'r', encoding='utf-8') as dataFile, open(outputPath, 'w', encoding='utf-8') as outputFile, \
    open(titlePath, 'r', encoding='utf-8') as titleFile, open(goldTemplatePath, 'r', encoding='utf-8') as goldTemplateFile, \
    open(comparisonPath, 'w', encoding='utf-8') as comparisonFile:
    fileIterators = zip(goldFile.readlines(), goldTemplateFile.readlines(),
                        generatedFile.readlines(), dataFile.readlines(), titleFile.readlines())
    for gold, goldTemplate, generated, data, title in fileIterators:
        count += 1
        xValueArr = []
        yValueArr = []
        reversedArr = []
        datum = data.split()
        xLabel = datum[0].split('|')[0].split('_')
        yLabel = datum[1].split('|')[0].split('_')

        # remove filler words from labels
        cleanXLabel = [xWord for xWord in xLabel if xWord.lower() not in fillers]
        cleanYLabel = [yWord for yWord in yLabel if yWord.lower() not in fillers]

        for i in range(0, len(datum)):
            if i % 2 == 0:
                xValueArr.append(datum[i].split('|')[1])
            else:
                yValueArr.append(datum[i].split('|')[1])
        tokens = generated.split()
        entities = getNamedEntity(title, xValueArr)
        # titleEntities = [word[0].lower() for word in entities.values() if len(word) > 0]
        titleArr = [word for word in title.split() if word.lower() not in fillers]
        # titleArr = [word for word in titleArr if word.lower() not in titleEntities]
        for sentence in generated.split(' . '):
            reversedSentence = []
            sentence = sentence.split()
            maskList = []
            for token, i in zip(sentence, range(0, len(sentence))):
                if i < (len(tokens) - 1):
                    if tokens[i] == tokens[i + 1]:
                        print(f'1:{tokens[i]} 2:{tokens[i + 1]}')
                        print(f'popped: {tokens[i + 1]}')
                        tokens.pop(i + 1)
                if 'template' in token:
                    index = str(re.search(r"\[(\w+)\]", token).group(0)).replace('[', '').replace(']', '')
                    if 'templateXValue' in token:
                        index = mapIndex(index, xValueArr)
                        try:
                            replacedToken = xValueArr[index].replace('_', ' ')
                        except:
                            replacedToken = xValueArr[len(xValueArr) - 1].replace('_', ' ')
                    elif 'templateYValue' in token:
                        index = mapIndex(index, yValueArr)
                        try:
                            replacedToken = yValueArr[index].replace('_', ' ')
                        except:
                            replacedToken = yValueArr[len(yValueArr) - 1].replace('_', ' ')
                    elif 'templateTitleSubject' in token:
                        # print(entities['Subject'][int(index)], index)
                        try:
                            replacedToken = entities['Subject'][int(index)]
                        except:
                            replacedToken = entities['Subject'][len(entities['Subject']) - 1]
                    elif 'templateTitleDate' in token:
                        # print(entities['Date'][int(index)], index)
                        try:
                            replacedToken = entities['Date'][int(index)]
                        except:
                            if len(entities['Date']) > 0:
                                replacedToken = entities['Date'][len(entities['Date']) - 1]
                            else:
                                replacedToken = ''
                    elif 'templateTitle' in token:
                        replacedToken = token
                        maskList.append([titleArr, i])
                    elif 'templateXLabel' in token:
                        replacedToken = token
                        maskList.append([cleanXLabel, i])
                    elif 'templateYLabel' in token:
                        replacedToken = token
                        maskList.append([cleanYLabel, i])
                else:
                    replacedToken = token
                if i > 1:
                    # check if replaced token is a duplicate
                    if replacedToken.lower() != reversedSentence[-1].lower():
                        reversedSentence.append(replacedToken)
                    else:
                        print(f'dupe removed: {replacedToken}')
                else:
                    reversedSentence.append(replacedToken)
            reversedSentence = [item for item in reversedSentence if item != '']
            # reversedSentence = ' '.join(reversedSentence)
            for mask in maskList:
                print(reversedSentence)
                options = mask[0]
                index = mask[1]
                print(options)
                reversedSentence[index] = '***mask***'
                masked_string = ' '.join(reversedSentence)
                print(masked_string)
                newToken = askBert(masked_string, options)
                print(newToken)
                reversedSentence[index] = newToken
                print(reversedSentence)
                print('')
            reversedArr.append(' '.join(reversedSentence))
        # remove empty items
        reversedArr = [sentence for sentence in reversedArr if sentence != '']
        reverse = ' . '.join(reversedArr)
        comparison = f'Example {count}:\ntitleEntities: {entities}\ntitle: {title}X_Axis{xLabel}: {xValueArr}\nY_Axis{yLabel}: {yValueArr}\n\ngold: {gold}' \
                     f'gold_template: {goldTemplate}\ngenerated_template: {generated}generated: {reverse}\n\n'
        print(comparison)
        comparisonFile.write(comparison)
        outputFile.write(f'{reverse}\n')


# try:
#main()
# except Exception as ex:
#    print('error:', ex)

""" elif are_numbers(array[0:len(array)-2]):
    array = [float(i) for i in array[0:len(array)-2]]
    if str(index) == 'max':
        index = array.index(max(array))
        return int(index)
    elif str(index) == 'min':
        index = array.index(min(array))
        return int(index)    """
