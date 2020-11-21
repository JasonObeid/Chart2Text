"""tokenizer.py: tokenization utility for RotoWire Parallel dataset.

This file contains the following functions:
    - word_tokenize: Tokenize a string into a list of words.
    - sent_tokenize: Tokenize a string into a list of sentences.

Important: Use Python version >=3.5.
"""
import re
from typing import List

import nltk


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


def sent_tokenize(string: str, language: str = "english") -> List[str]:
    """tokenizes a given (detokenized) string into a list of sentences.

    :param string: String to tokenize.
    :param language: Language. Either one of ``english'' or ``german''.
    """
    # Initials names
    string = re.sub(r"(^|\s)(\w)\. (\w)\.", r"\g<1>\g<2>._ \g<3>._", string)

    # 3Pt.
    string = string.replace("3Pt.", "3Pt._")

    tokenized = nltk.sent_tokenize(string, language=language)

    joined = " || ".join(tokenized)
    # shrink expanded name-with-initial expressions
    joined = joined.replace("._", ".")
    tokenized = joined.split(" || ")

    return tokenized


def detokenize(words: List[str]) -> str:
    """detokenize a list of strings into one string.

    :param words: List of words (or any strings).
    """
    try:
        from sacremoses import MosesDetokenizer
        MD = MosesDetokenizer(lang="en")
    except ModuleNotFoundError:
        print("Please install sacremosees.")
        raise

    detokenized = MD.detokenize(words)

    # Initials names
    detokenized = re.sub(r"(\w)\.(\w)\.", r"\g<1>. \g<2>.", detokenized)

    # period mistakes (e.g., win.The)
    detokenized = re.sub(r"(\w{2,})\.(\w{2,})", r"\g<1>. \g<2>", detokenized)

    return detokenized


if __name__ == "__main__":
    import sys
    language = sys.argv[1] if len(sys.argv) > 1 else "english"
    for line in sys.stdin:
        print(" ".join(word_tokenize(line, language=language)))
