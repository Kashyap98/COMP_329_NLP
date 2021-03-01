import random
from typing import List, Tuple

import gen_utils
import result_utils

from HW_2.naive_bayes import NaiveBayesClassifier


def format_input_data(positive_data: List[str], negative_data: list) -> Tuple[List[Tuple[str, int]],
                                                                              List[Tuple[str, int]]]:
    # 1 = positive, 0 = negative
    positive_input_data = []
    negative_input_data = []

    for sentence_data in positive_data:
        positive_input_data.append((sentence_data, 1))

    for sentence_data in negative_data:
        negative_input_data.append((sentence_data, 0))

    return positive_input_data, negative_input_data


def split_data(input_data: List[Tuple[str, int]], split_percentage: float = 0.70) -> Tuple[List[Tuple[str, int]],
                                                                                           List[Tuple[str, int]]]:
    input_data = set(input_data)
    training_count = int(len(input_data) * split_percentage)

    training_data = set(random.sample(input_data, training_count))
    test_data = input_data - training_data

    return list(training_data), list(test_data)


if __name__ == '__main__':
    positive_review_data = gen_utils.get_sentence_list_for_word_file(gen_utils.POSITIVES_REVIEWS_PATH)
    negative_review_data = gen_utils.get_sentence_list_for_word_file(gen_utils.NEGATIVE_REVIEWS_PATH)
    stop_words = gen_utils.get_sentence_list_for_word_file(gen_utils.STOP_WORDS_PATH)

    formatted_positive_data, formatted_negative_data = format_input_data(positive_review_data, negative_review_data)

    positive_training, positive_test = split_data(formatted_positive_data)
    negative_training, negative_test = split_data(formatted_negative_data)

    model_training_data = positive_training + negative_training
    model_test_data = positive_test + negative_test

    print(f"Training Data Count: {len(model_training_data)}")
    print(f"Testing Data Count: {len(model_test_data)}")

    classifier = NaiveBayesClassifier(training_data=model_training_data, stop_word_filter=stop_words, max_words=10000)
    results = result_utils.Results()
    for review_data in model_test_data:
        review, is_positive = review_data[0], bool(review_data[1])
        if is_positive:
            results.add_positive()
        else:
            results.add_negative()

        predictions = classifier.predict_one_sentence_value(review)
        prediction = predictions[0]
        prediction_value, confidence = bool(prediction[0]), prediction[1]

        results.add_result(is_positive=prediction_value, should_be_positive=is_positive)

    results.report()
