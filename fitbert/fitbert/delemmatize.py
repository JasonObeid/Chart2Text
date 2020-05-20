"""
Plan:

Lemmatize word with spacy
get data/lemma_lookup table, invert
find all words that share the same lemma

maybe need to open it up a bit by stemming the word,
then inverse lookup

More acurate than ^ would be to use wordnet to find "derivationally-related terms"
see https://github.com/explosion/spaCy/issues/327
and
https://stackoverflow.com/questions/17684186/nltk-words-lemmatizing/17687095#17687095
"""

from collections import defaultdict
from typing import Dict, List

from .data.lemma_lookup import LOOKUP


class Delemmatizer:
    REVERSE_LOOKUP: Dict[str, List[str]] = defaultdict(list)
    LOOKUP: Dict[str, str] = LOOKUP

    def __init__(self):
        if not Delemmatizer.REVERSE_LOOKUP:
            """REVERSE_LOOKUP is a class property, so once initialized it will be available
            for all instances"""
            for k, v in LOOKUP.items():
                Delemmatizer.REVERSE_LOOKUP[v].append(k)

    def __call__(self, word: str) -> List[str]:
        try:
            delems = Delemmatizer.REVERSE_LOOKUP[word]
            delems.append(word)
        except KeyError:
            delems = [word]
        if len(delems) <= 1:
            try:
                word = Delemmatizer.LOOKUP[word]
                delems = Delemmatizer.REVERSE_LOOKUP[word]
                delems.append(word)
            except KeyError:
                pass
        return delems
