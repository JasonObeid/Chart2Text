# Sentence METEOR

# METEOR mainly works on sentence evaluation rather than corpus evaluation
# Run this file from CMD/Terminal
# Example Command: python3 sentence-meteor.py test_file_name.txt mt_file_name.txt

import sys
from nltk.translate.meteor_score import meteor_score


target_test = sys.argv[1]	#  Test file argument
target_pred = sys.argv[2]	#  MTed file argument


# Open the test dataset human translation file
with open(target_test) as test:
    refs = test.readlines()

#print("Reference 1st sentence:", refs[0])

# Open the translation file by the NMT model
with open(target_pred) as pred:
    preds = pred.readlines()

meteor_file = "meteor-" + target_pred + ".txt"

# Calculate METEOR for each sentence and save the result to a file
with open(meteor_file, "w+") as output:
    for line in zip(refs, preds):
        test = line[0]
        pred = line[1]
        #print(test, pred)

        meteor = round(meteor_score([test], pred), 2) # list of references
        #print(meteor, "\n")
        output.write(str(meteor) + "\n")

print("Done! Please check the METEOR file '" + meteor_file + "' in the same folder!")
