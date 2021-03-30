from typing import List, Tuple

import gen_utils
import result_utils

from HW_2.naive_bayes import NaiveBayesClassifier


def run_naive_bayes(training_data: List[Tuple[str, int]],
                    testing_data: List[Tuple[str, int]],
                    max_words: int) -> Tuple[result_utils.Results, List[Tuple[str, List[Tuple[str, float]]]]]:
    """
    Train and run the Naive Bayes classifier one time
    @param training_data: list of labeled training data
    @param testing_data: list of labeled testing data
    @param max_words: maximum number of words that should be used in classifier
    @return: result of the classifier
    """
    classifier = NaiveBayesClassifier(training_data=training_data,
                                      max_words=max_words)
    results = result_utils.Results()
    all_predictions = []
    for review_data in testing_data:
        review, is_positive = review_data[0], bool(review_data[1])
        if is_positive:
            results.add_positive()
        else:
            results.add_negative()

        predictions = classifier.predict_one_sentence_value(review)
        all_predictions.append((review, predictions))

        # get most likely prediction
        prediction = predictions[0]
        prediction_value, confidence = bool(prediction[0]), prediction[1]

        results.add_result(is_positive=prediction_value, should_be_positive=is_positive)

    return results, all_predictions


def output_most_certain_and_most_uncertain_predictions(predictions: List[Tuple[str, List[Tuple[str, float]]]]):
    """
    Get the top 3 most certain and uncertain predictions for the classifier run
    @param predictions: list of sentences and their predictions
    """
    def _tuple_sort(input_tuple):
        return input_tuple[1]

    predictions_to_sort = []
    for prediction_list in predictions:
        sentence, predictions_for_sentence = prediction_list[0], prediction_list[1]
        difference = abs(predictions_for_sentence[0][1] - predictions_for_sentence[1][1])
        predictions_to_sort.append((sentence, difference))

    predictions_to_sort = sorted(predictions_to_sort, key=_tuple_sort, reverse=True)
    print("Most Certain")
    for most_certain in predictions_to_sort[:3]:
        print(f"sentence={most_certain[0]} | value={most_certain[1]} ")
    print("______________________________________________________________________")

    print("Least Certain")
    for least_certain in predictions_to_sort[len(predictions_to_sort) - 3:]:
        print(f"sentence={least_certain[0]} | value={least_certain[1]} ")
    print("______________________________________________________________________")


if __name__ == '__main__':
    # gather data for the test
    positive_review_data = gen_utils.get_sentence_list_for_word_file(gen_utils.POSITIVES_REVIEWS_PATH)
    negative_review_data = gen_utils.get_sentence_list_for_word_file(gen_utils.NEGATIVE_REVIEWS_PATH)
    stop_words = gen_utils.get_sentence_list_for_word_file(gen_utils.STOP_WORDS_PATH)
    punctuation = gen_utils.get_sentence_list_for_word_file(gen_utils.PUNCTUATION_PATH)
    stop_words.extend(punctuation)

    # format data into tuples for the classifier
    formatted_positive_data, formatted_negative_data = gen_utils.format_input_data(positive_review_data,
                                                                                   negative_review_data)

    # split data into different sets -- used for dev work
    # positive_training, positive_test, positive_dev = gen_utils.split_dev_data(formatted_positive_data)
    # negative_training, negative_test, negative_dev = gen_utils.split_dev_data(formatted_negative_data)

    # split data into different sets
    positive_training, positive_test = gen_utils.split_data(formatted_positive_data)
    negative_training, negative_test = gen_utils.split_data(formatted_negative_data)

    # create final data sets and run classifier
    model_training_data = positive_training + negative_training
    model_test_data = positive_test + negative_test
    print(f"Training Data Count: {len(model_training_data)}")
    print(f"Testing Data Count: {len(model_test_data)}")

    # lines below are used to run the classifier 1 time and output most certain/least certain predictions
    result, classifier_predictions = run_naive_bayes(model_training_data, model_test_data, max_words=11000)
    result.report()
    output_most_certain_and_most_uncertain_predictions(classifier_predictions)

    # function below is used to calculate the maximum results possible as well as the amount of words used.
    # maximum_results, max_words_for_result = result_utils.get_maximum_results(iter_start=1000, iter_end=12000,
    #                                                                          iter_step=1000,
    #                                                                          model_func=run_naive_bayes,
    #                                                                          training_data=model_training_data,
    #                                                                          testing_data=model_test_data)
