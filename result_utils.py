
class Results:
    """
    This class is used to calculate the final results after classification. This is used rather than SKLearn because
    the first homework assignment did not allow for use of the SKLearn package.
    """
    def __init__(self):
        self.positive = 0
        self.negative = 0
        self.total = 0

        self.false_positive = 0
        self.false_negative = 0
        self.true_positive = 0
        self.true_negative = 0

        self.correct_predictions = 0
        self.total_predictions = 0

        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.specificity = 0
        self.f1_score = 0

    def add_result(self, is_positive: bool, should_be_positive: bool):
        if (is_positive is True) and (should_be_positive is True):
            self.true_positive += 1
        elif (is_positive is True) and (should_be_positive is False):
            self.false_positive += 1
        elif (is_positive is False) and (should_be_positive is True):
            self.false_negative += 1
        elif (is_positive is False) and (should_be_positive is False):
            self.true_negative += 1
        else:
            print("An error has occurred in analysis.")

    def add_positive(self):
        self.positive += 1
        self.total += 1

    def add_negative(self):
        self.negative += 1
        self.total += 1

    def add_false_negative(self):
        self.false_negative += 1

    def add_false_positive(self):
        self.false_positive += 1

    def add_true_positive(self):
        self.true_positive += 1

    def add_true_negative(self):
        self.true_negative += 1

    def calc_accuracy(self):
        """
        Calculate the predictions that were classified correctly
        """
        self.correct_predictions = self.true_positive + self.true_negative
        self.total_predictions = self.correct_predictions + (self.false_positive + self.false_negative)

        self.accuracy = round(self.correct_predictions / self.total_predictions, 3)

    def calc_precision(self):
        """
        Calculate how many positive classifications were actually correct (Positive Prediction Value)
        """
        self.precision = round(self.true_positive / (self.true_positive + self.false_positive), 3)

    def calc_recall(self):
        """
        Calculate how many positive classifications were correct out of how many there were
        (Sensitivity, Hit Rate, True Positive Rate)
        """
        self.recall = round(self.true_positive / (self.true_positive + self.false_negative), 3)

    def calc_specificity(self):
        """
        Calculate proportion of negatives that were identified (True Negative Rate)
        Opposite of recall
        """
        self.specificity = round(self.true_negative / (self.true_negative + self.false_positive), 3)

    def calc_f1_score(self):
        """
        Mean of precision and recall
        maximum score of 1 (perfect precision and recall) and a minimum of 0.
        """
        self.f1_score = round((2 * (self.precision * self.recall)) / (self.precision + self.recall), 2)

    def report(self):
        self.calc_accuracy()
        self.calc_precision()
        self.calc_recall()
        self.calc_specificity()
        self.calc_f1_score()

        print(f"{self.positive=}")
        print(f"{self.negative=}")
        print(f"{self.total=}")
        print()

        print(f"{self.accuracy=}")
        print(f"{self.precision=}")
        print(f"{self.recall=}")
        print(f"{self.specificity=}")
        print(f"{self.f1_score=}")
        print()

        print(f"       P       N    ")
        print(f"   - - - - - - - - -")
        print(f"P  -  {self.true_positive} -  {self.false_positive}  -")
        print(f"   - - - - - - - - -")
        print(f"N  -  {self.false_negative} -  {self.true_negative}  -")
        print(f"   - - - - - - - - -")
