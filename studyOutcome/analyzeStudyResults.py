from statistics import mean

import pandas as pd
import csv
from collections import Counter
from statistics import stdev
"""Created on Aug 1, 2016
@author: skarumbaiah
Computes Fleiss' Kappa 
Joseph L. Fleiss, Measuring Nominal Scale Agreement Among Many Raters, 1971."""

def checkInput(rate, n):
    """
    Check correctness of the input matrix
    @param rate - ratings matrix
    @return n - number of raters
    @throws AssertionError
    """
    N = len(rate)
    k = len(rate[0])
    assert all(len(rate[i]) == k for i in range(k)), "Row length != #categories)"
    assert all(isinstance(rate[i][j], int) for i in range(N) for j in range(k)), "Element not integer"
    assert all(sum(row) == n for row in rate), "Sum of ratings != #raters)"

def fleissKappa(rate,n):
    """
    Computes the Kappa value
    @param rate - ratings matrix containing number of ratings for each subject per category
    [size - N X k where N = #subjects and k = #categories]
    @param n - number of raters
    @return fleiss' kappa
    """

    N = len(rate)
    k = len(rate[0])
    print("#raters = ", n, ", #subjects = ", N, ", #categories = ", k)
    checkInput(rate, n)

    #mean of the extent to which raters agree for the ith subject
    PA = sum([(sum([i**2 for i in row])- n) / (n * (n - 1)) for row in rate])/N
    print("PA = ", PA)

    # mean of squares of proportion of all assignments which were to jth category
    PE = sum([j**2 for j in [sum([rows[i] for rows in rate])/(N*n) for i in range(k)]])
    print("PE =", PE)

    kappa = -float("inf")
    try:
        kappa = (PA - PE) / (1 - PE)
        kappa = float("{:.3f}".format(kappa))
    except ZeroDivisionError:
        print("Expected agreement = 1")

    print("Fleiss' Kappa =", kappa)

    return kappa

labels = ['System' , 'Informativeness' , 'Conciseness' , 'Coherence' , 'Fluency' , 'Sentence 1 yes' , 'Sentence 1 no' , 'Sentence 1 partial' , 'Sentence 1 other' , 'Sentence 2 yes' , 'Sentence 2 no' , 'Sentence 2 partial' , 'Sentence 2 other' , 'Sentence 3 yes' , 'Sentence 3 no' , 'Sentence 3 partial' , 'Sentence 3 other']

def getMeanScore(df):
    scores = []
    #print(type(df))
    for col in df.iteritems():
        score = col[1].to_list().index(True) + 1
        scores.append(score)
    meanScore = round(sum(scores) / len(scores), 2)
    stdevScore = round(stdev(scores), 2)
    return meanScore, stdevScore, scores

#def getSentencePercentages(sentences):


def getKappaScore(scores):
    responses = []
    # 1 - 120
    for m in range(120):
        # 1 - 4
        response = [scores[n][m] for n in range(4)]
        responses.append(response)
    #print(len(responses))
    #print(responses)
    respondent1Scores = []
    respondent2Scores = []
    respondent3Scores = []
    sampleCounter = 0
    for response, i in zip(responses, range(len(responses))):
        if sampleCounter == 3:
            sampleCounter = 0
        if sampleCounter == 0:
            respondent1Scores.append(response)
        elif sampleCounter == 1:
            respondent2Scores.append(response)
        elif sampleCounter == 2:
            respondent3Scores.append(response)
        sampleCounter += 1
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    for response1, response2, response3 in zip(respondent1Scores, respondent2Scores, respondent3Scores):
        #iterate over questions:
        print(response1, response2, response3)
        agreement = []
        for i in range(4):
            question = [response1[i], response2[i], response3[i]]
            x = Counter(question)
            lists = [0, 0, 0, 0, 0]
            for k, v in x.items():
                lists[k-1] = v
            if i == 0:
                q1.append(lists)
            elif i == 1:
                q2.append(lists)
            elif i == 2:
                q3.append(lists)
            elif i == 3:
                q4.append(lists)
    print(q1)
    print(q4)
    kappa1 = fleissKappa(q1, 3)
    kappa2 = fleissKappa(q2, 3)
    kappa3 = fleissKappa(q3, 3)
    kappa4 = fleissKappa(q4, 3)
    return [kappa1,kappa2,kappa3,kappa4]

fileNames = ['OurModel','Baseline']

def readData(filePath, count):
    line = []
    line.append(fileNames[count])
    df = pd.read_csv(filePath)
    dataOrder.append(df["Input.imgPath"].tolist())

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
    items = []
    print(question1)
    for question in [question1, question2, question3, question4]:
        meanScore, stdevScore, scores = getMeanScore(question)
        line.append(meanScore)
        items.append(scores)
        print(', '.join([str(score) for score in scores]))
        print(f'Question {questionCount} mean: {meanScore}')
        print(f'Question {questionCount} stdev: {stdevScore}')
        questionCount += 1
    #print(len(items))
    kappaScores = getKappaScore(items)
    print(kappaScores)

    sentence1 = pd.DataFrame((df["Answer.yes0.1"], df["Answer.no0.0"],
                              df["Answer.partial0.2"], df["Answer.other0.3"], df["Answer.otherText0"]))
    if 'baseline' not in filePath:
        sentence2 = pd.DataFrame((df["Answer.yes1.1"], df["Answer.no1.0"],
                                  df["Answer.partial1.2"], df["Answer.other1.3"], df["Answer.otherText1"]))
        #getSentencePercentages(sentence2, 2)
        sentence3 = pd.DataFrame((df["Answer.yes2.1"], df["Answer.no2.0"],
                                  df["Answer.partial2.2"], df["Answer.other2.3"], df["Answer.otherText2"]))
        sentences = [sentence1, sentence2, sentence3]
    else:
        sentences = [sentence1]

    for sentenceDF in sentences:
        if not sentenceDF.empty:
            studySize = sentenceDF.shape[1]
            yesList = 0
            noList = 0
            partialList = 0
            otherList = []
            for col in sentenceDF.iteritems():
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
            yesRatio = f'{round((yesList / studySize)*100, 3)}%'
            noRatio = f'{round((noList / studySize)*100, 3)}%'
            partialRatio = f'{round((partialList / studySize)*100, 3)}%'
            otherRatio = f'{round((len(otherList) / studySize)*100, 3)}%'
            print(f'yes: {yesRatio}')
            print(f'no: {noRatio}')
            print(f'partial: {partialRatio}')
            print(f'other: {otherRatio}')
            line.append(yesRatio)
            line.append(noRatio)
            line.append(partialRatio)
            line.append(otherRatio)
        else:
            line.append('')
            line.append('')
            line.append('')
            line.append('')

    print(line)
    csvwriter.writerow(line)

paths = ['./results_ours.csv', './results_untemplated.csv']

dataOrder = []
with open('../studyOutcome/studyStats.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(labels)
    for path, count in zip(paths, range(len(paths))):
        readData(path, count)
    #check order is correct for kappa score validity
    assert dataOrder[0] == dataOrder[1]

