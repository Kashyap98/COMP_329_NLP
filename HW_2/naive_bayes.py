import collections
import math
from typing import Tuple, List


class NaiveBayesClassifier:

    def __init__(self, training_data: List[Tuple[str, int]], stop_word_filter: List[str] = None, max_words: int = None):
        """
        This Naive Bayes Classifier is made to predict values from input data based on training data provided.
        @param training_data: List of tuples (sentence: str, value: int). This data will be converted to proportions.
        @param stop_word_filter: List of stop words to remove from training data.
        @param max_words: How many words should be kept for proportion calculations.
        """
        # Add some basic error handling for the inputs
        if len(training_data) < 1:
            raise ValueError("Training data is empty.")

        if stop_word_filter is None:
            stop_word_filter = []

        if max_words is None:
            max_words = math.inf

        self.training_data = training_data
        self.max_words = max_words
        self.stop_word_filter = stop_word_filter
        self.data_word_counts: dict = collections.defaultdict(collections.Counter)
        self.word_count_by_prediction: dict = collections.defaultdict(int)
        self.data_word_proportions: dict = {}

        self.prediction_counts = collections.Counter()
        self.prediction_values = set()
        self.total_prediction_count: int = 0
        self.prediction_proportions: dict = {}

        # perform init functions
        self.get_unique_prediction_values()
        self.get_counts_for_training_data()
        self.convert_counts_to_proportions_for_training_data()

        self.total: int = 0

    @staticmethod
    def _sort_predictions(prediction_tuple: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Sort the class predications from most likely to least
        @param prediction_tuple: predicted classes and prediction weight
        @return: sorted prediction classes and weights
        """
        def tuple_sorter(item: Tuple[str, float]) -> float:
            return item[1]

        # put the highest prediction value first.
        prediction_tuple = sorted(prediction_tuple, key=tuple_sorter, reverse=True)
        return prediction_tuple

    def _filter_words(self, words: List[str]) -> List[str]:
        """
        Remove words based on stop filter
        @param words: words in the sentence
        @return: words that are not in the set filter
        """
        # filter words if necessary
        if self.stop_word_filter is not []:
            filtered_words = []
            for word in words:
                if word not in self.stop_word_filter:
                    filtered_words.append(word)
        else:
            filtered_words = words

        return filtered_words

    def get_unique_prediction_values(self):
        """
        For the purposes of the homework this is ony binary. However, this function should be able to work even when
        the prediction is not a binary result (y/n).

        This function gets the counts for each class that we will be predicting for.
        """
        prediction_values: list = [value[1] for value in self.training_data]
        self.prediction_counts.update(prediction_values)

        # add counts of each prediction to dict
        for prediction, value in self.prediction_counts.items():
            self.prediction_values.add(prediction)
            self.prediction_proportions[prediction] = value

            # add prediction count to total
            self.total_prediction_count += value

        # add prediction values to data_word_counts. This is done so that counts can be found for each different
        # prediction type
        self.data_word_counts.fromkeys(self.prediction_values)

        # convert prediction counts to proportions
        for prediction, count in self.prediction_proportions.items():
            self.prediction_proportions[prediction] = count / self.total_prediction_count

    def get_counts_for_training_data(self):
        """
        Work through training data by adding the counts of words for each classification into a Counter for that
        classification. This will be done to create the final proportion calculations for each word on each prediction.
        """
        for data in self.training_data:
            sentence, value = data[0], data[1]
            # update relevant output value counts with words from the sentence
            words = sentence.split(" ")
            filtered_words = self._filter_words(words)

            self.data_word_counts[value].update(filtered_words)
            self.word_count_by_prediction[value] = self.word_count_by_prediction[value] + len(filtered_words)

    def convert_counts_to_proportions_for_training_data(self):
        """
        Convert the counts of each word for each classification into a proportion of the total. This is done so the
        classifier can pull these values for a predication of a new value.
        """
        for value, counter in self.data_word_counts.items():
            proportions_dict = {}
            current_count = 0
            proportion_denominator = math.log(len(counter))

            for word, count in counter.items():
                if current_count >= self.max_words:
                    break
                current_count += 1

                # convert value to log to prevent float underflow
                proportions_dict[word] = math.log(count) - proportion_denominator

            self.data_word_proportions[value] = proportions_dict

    def predict_one_sentence_value(self, sentence: str) -> List[Tuple[str, float]]:
        """
        Predict value of sentence from classes based on previous proportions
        @param sentence: sentence that will be predicted on
        @return: tuple of prediction values and weights
        """
        words = sentence.split(" ")
        prediction_estimates = []

        # get an estimate for each prediction this classifier can make
        for value in self.prediction_values:
            value_proportions = self.data_word_proportions[value]
            estimate: float = math.log(self.prediction_proportions[value])

            # go through sentence and use previous proportions
            for word in words:
                # current only looking at words we have previously seen
                if word in value_proportions:
                    estimate = estimate + value_proportions[word]
                else:
                    # use laplace smoothing for words we do not have the value for.
                    estimate = estimate + \
                               (math.log(1) -
                                (math.log(self.word_count_by_prediction[value] + len(value_proportions) + 1)))

            # convert log value back to un normalized - done to prevent float underflow
            estimate = math.pow(math.e, estimate)
            prediction_estimates.append((value, estimate))

        # sort predictions from most likely to least likely
        prediction_estimates = self._sort_predictions(prediction_estimates)
        return prediction_estimates
