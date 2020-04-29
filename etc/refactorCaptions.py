import scripts.tokenizer as tkn
import os

summaryPaths = os.listdir('dataset/captions_old/')

for summaryPath in summaryPaths:
        with open('dataset/captions_old/'+summaryPath, 'r', encoding='utf-8') as summaryFile:
                summary = summaryFile.read()
                #print(len(tkn.word_tokenize(summary)))
                print(len(summary.split()))
                newSummary = tkn.word_tokenize(summary)
                newSummaryPath = 'dataset/captions/'+summaryPath
                with open(newSummaryPath, "w") as outf:

                    outf.write("{}\n".format(' '.join(newSummary)))
                outf.close()
                print(newSummary)