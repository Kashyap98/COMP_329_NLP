from typing import Tuple, List

import pandas as pd
from sklearn import svm, metrics

import gen_utils
import result_utils


DEBUG_MODE = False


def get_hate_speech_data() -> Tuple[List[str], List[int]]:
    train_data = pd.read_csv(gen_utils.HATE_SPEECH_PATH, header=0)
    # class = 0 - hate speech 1 - offensive language 2 - neither
    labels = list(train_data["class"].array)
    text = list(train_data["tweet"].array)

    return text, labels


def use_my_count_vectorizer(text: List[str], labels: List[int]) -> List[List[int]]:
    # transform text into vectors (using my CountVectorizer)
    from HW_3.vectorizer import CountVectorizer
    vectorizer = CountVectorizer(text, labels, debug_mode=DEBUG_MODE)
    vectors = vectorizer.get_vectors()

    return vectors


def use_sklearn_count_vectorizer(text: List[str]) -> List[List[int]]:
    # using sklearn CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    vector_holder = vectorizer.fit_transform(text)
    vectors = vector_holder.toarray()

    return vectors


def transform_vectors_into_train_and_test(vectors: List[List[int]],
                                          labels: List[int]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    # split vector and labels into train/test sets
    input_array = list(zip(vectors, labels))
    train_data, test_data = gen_utils.split_data(input_array)

    return train_data, test_data


def predict_svm_and_export_results(clf, test_text_data: List[str], test_labels_data: List[int],
                                   target_names: List[str]):
    # predict and capture results
    results = result_utils.Results()
    all_predictions = []
    test_length = len(test_text_data)
    for i in range(test_length):
        # get data to test
        data, label = test_text_data[i], test_labels_data[i]
        results.add_to_total()

        if DEBUG_MODE:
            print(f"Processing result: {i} / {test_length}")

        # get prediction
        predictions = clf.predict([data])
        prediction = predictions[0]
        all_predictions.append((data, predictions))

        # add prediction to results
        results.add_multiclass_result(result=prediction, expected_result=label)

    # output results
    results.report()
    data, predicted = zip(*all_predictions)
    print(metrics.classification_report(test_labels_data, predicted,
                                        target_names=target_names))
    print(metrics.confusion_matrix(test_labels_data, predicted))


if __name__ == '__main__':
    # get input data
    hate_text, hate_labels = get_hate_speech_data()

    # use one vectorizer
    hate_vectors = use_my_count_vectorizer(hate_text, hate_labels)
    # hate_vectors = use_sklearn_count_vectorizer(hate_text)

    # get final 70/30 split of data
    v_train_data, v_test_data = transform_vectors_into_train_and_test(hate_vectors, hate_labels)
    train_text, train_labels = zip(*v_train_data)
    test_text, test_labels = zip(*v_test_data)

    # train SVM classifier
    if DEBUG_MODE:
        print("Training SVM")
    clf_classifier = svm.LinearSVC()
    clf_classifier.fit(train_text, train_labels)

    # predict test data and output results
    predict_svm_and_export_results(clf_classifier, test_text, test_labels,
                                   target_names=["hate_speech", "offensive_language", "neither"])
