import pandas as pd

def getMeanScore(df):
    items = []
    for col in df.iteritems():
        score = col[1].to_list().index(True) + 1
        items.append(score)
    meanScore = round(sum(items) / len(items),2)
    return meanScore

def getSentencePercentages(df, count):
    studySize = df.shape[1]
    yesList = 0
    noList = 0
    partialList = 0
    otherList = []
    for col in df.iteritems():
        if col[1].isna().any() == False:
            trueIndex = col[1].to_list().index(True)
            if trueIndex == 0:
                yesList += 1
            elif trueIndex == 1:
                noList += 1
            elif trueIndex == 2:
                partialList += 1
            elif trueIndex == 3:
                otherList.append(col[1][4])
        else:
            #shrink size if the sentence question did not exist
            studySize -= 1
    yesRatio = round(yesList / studySize, 2)
    noRatio = round(noList / studySize, 2)
    partialRatio = round(partialList / studySize, 2)
    otherRatio = round(len(otherList) / studySize, 2)
    print(f'sentence {count}')
    print(f'yes: {yesRatio}')
    print(f'no: {noRatio}')
    print(f'partial: {partialRatio}')
    print(f'other: {otherRatio}')

def readData(filePath):
    df = pd.read_csv(filePath)

    question1 = pd.DataFrame((
        df["Answer.option1d.1"], df["Answer.option2d.2"], df["Answer.option3d.3"],
        df["Answer.option4d.4"], df["Answer.option5d.5"])) #, df["Answer.explanation4"]))

    question2 = pd.DataFrame((
        df["Answer.option1c.1"], df["Answer.option2c.2"], df["Answer.option3c.3"],
        df["Answer.option4c.4"], df["Answer.option5c.5"])) #, df["Answer.explanation4"]))

    question3 = pd.DataFrame((
        df["Answer.option1a.1"], df["Answer.option2a.2"], df["Answer.option3a.3"],
        df["Answer.option4a.4"], df["Answer.option5a.5"])) #, df["Answer.explanation4"]))

    question4 = pd.DataFrame((
        df["Answer.option1b.1"], df["Answer.option2b.2"], df["Answer.option3b.3"],
        df["Answer.option4b.4"], df["Answer.option5b.5"])) #, df["Answer.explanation4"]))

    print(filePath)
    questionCount = 0
    for question in [question1, question2, question3, question4]:
        print(f'Question {questionCount}: {getMeanScore(question)}')
        questionCount += 1

    sentence1 = pd.DataFrame((df["Answer.yes0.1"], df["Answer.no0.0"],
                              df["Answer.partial0.2"], df["Answer.other0.3"], df["Answer.otherText0"]))
    getSentencePercentages(sentence1, 1)

    if 'baseline' not in filePath:
        sentence2 = pd.DataFrame((df["Answer.yes1.1"], df["Answer.no1.0"],
                                  df["Answer.partial1.2"], df["Answer.other1.3"], df["Answer.otherText1"]))
        getSentencePercentages(sentence2, 2)
        sentence3 = pd.DataFrame((df["Answer.yes2.1"], df["Answer.no2.0"],
                                  df["Answer.partial2.2"], df["Answer.other2.3"], df["Answer.otherText2"]))
        getSentencePercentages(sentence3, 3)
paths = ['../results_ours.csv', '../results_untemplated.csv', '../results_baseline.csv']

for path in paths:
    readData(path)