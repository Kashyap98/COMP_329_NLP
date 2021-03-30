from collections import Counter
from typing import List, Set


class CountVectorizer:

    def __init__(self, data: List[str], labels: List[int]):
        if len(data) != len(labels):
            raise RuntimeError("Length of input labels and data is not the same")

        self.data = data
        self.labels = labels
        self.unique_words: Set[str] = set()

        self.word_counts: Counter = Counter()

        self._get_word_counts()
        self._get_unique_words()

    def _get_word_counts(self):
        # go through each sentence and add unique words
        for sentence in self.data:
            words = sentence.split(" ")
            self.word_counts.update(words)

    def _get_unique_words(self):
        self.unique_words.update(self.word_counts.keys())




