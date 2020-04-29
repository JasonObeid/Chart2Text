labelPath = 'data/test/testSummaryLabel.txt'
summaryPath = 'data/test/testSummary.txt'
count = 0
import re
import scripts.tokenizer as tkn

with open(labelPath, 'r', encoding='utf-8') as labelFile, open(summaryPath, 'r', encoding='utf-8') as summaryFile:
    for label, summary in zip(labelFile.readlines(), summaryFile.readlines()):
        assert len(label.split(' ')) == len(summary.split())
        #print(len(tkn.word_tokenize(summary)))
        count += 1
        if count < 10:
            for labels, tokens in zip(label.split(' '), tkn.word_tokenize(summary)):
                print(f'label:{labels}, token:{tokens}')
        else:
            break

s = "string. With. Punctuation?"
print(s)
s = re.sub(r'[^\w\s]','',s)
print(s)