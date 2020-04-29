caption = 'This statistic shows imports of corn in the United States from 2001 to 2019. According to the report, U.S. corn imports amounted to approximately 57 million bushels in 2016, down from 68 million bushels the previous year.'

captionTokens = caption.rstrip().split()
labelMap = []
captionMatchCount = 0
for token in captionTokens:
    if token in xValueArr:
        print(token)
        tokenBool = 1
        captionMatchCount += 1
    elif token in yValueArr:
        print(token)
        tokenBool = 1
        captionMatchCount += 1
    else:
        tokenBool = 0
    labelMap.append(str(tokenBool))