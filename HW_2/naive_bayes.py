import collections


class NaiveBayesClassifier:

    def __init__(self, training_data: list, features: set, predict_column: str):

        # Add some basic error handling for the inputs
        if len(training_data) < 1:
            raise ValueError("Training data is empty.")
        if len(features) < 1:
            raise ValueError("Features of interest is empty.")
        if len(predict_column) == 0:
            raise ValueError("Prediction column is empty.")

        self.training_data = training_data
        self.training_headers: dict = self.training_data[0].keys()
        self.features = features
        self.predict_column = predict_column

        self.prediction_counts = collections.Counter()
        self.prediction_values = set()
        self.total_prediction_count: int = 0
        self.prediction_proportions: dict = dict()
        self.get_unique_prediction_values()

        self.total: int = 0

    def get_unique_prediction_values(self):
        """
        For the purposes of the homework this is ony binary. However, this function should be able to work even when
        the prediction is not a binary result (y/n)
        """
        prediction_values: list = self.training_headers[self.predict_column]
        self.prediction_counts.update(prediction_values)

        # add counts of each prediction to dict
        for prediction, value in self.prediction_counts.items():
            self.prediction_values.add(prediction)
            self.prediction_proportions[prediction] = value

            # add prediction count to total
            self.total_prediction_count += value

        # convert prediction counts to proportions
        for prediction, count in self.prediction_proportions.items():
            self.prediction_proportions[prediction] = round(count / self.total_prediction_count, 4)




