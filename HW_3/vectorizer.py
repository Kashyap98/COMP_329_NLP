from collections import Counter
from typing import List


class CountVectorizer:

    def __init__(self, data: List[str], labels: List[int], debug_mode: bool = False):
        if len(data) != len(labels):
            raise RuntimeError("Length of input labels and data is not the same")

        self.data = data
        self.data_length = len(self.data)
        self.labels = labels
        self.unique_words: List[str] = []
        self.final_vectors: List[List[int]] = []
        self.debug_mode = debug_mode

        self.word_counts: Counter = Counter()

        self._get_word_counts()
        self._get_unique_words()

    def _get_word_counts(self):
        # go through each sentence and add unique words
        for sentence in self.data:
            words = sentence.split(" ")
            self.word_counts.update(words)

    def _get_unique_words(self):
        self.unique_words.extend(self.word_counts.keys())

    def get_vectors(self):
        unique_word_count = len(self.unique_words)
        for data_count in range(self.data_length):
            sentence = self.data[data_count]
            vector = [0] * len(self.unique_words)

            if self.debug_mode:
                print(f"Processing data entry: {data_count} / {self.data_length}")

            for i in range(unique_word_count):
                word = self.unique_words[i]
                if word in sentence:
                    vector[i] = 1

            self.final_vectors.append(vector)
        return self.final_vectors
