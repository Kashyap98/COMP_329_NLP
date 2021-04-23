from typing import Tuple, List

import pandas as pd
from sklearn import svm

import gen_utils
import result_utils
from HW_1 import movie_review_analysis
from HW_2 import movie_nb_analysis
from HW_3 import toxic_comment_classification


def get_hotel_review_data() -> Tuple[List[str], List[int]]:
    train_data = pd.read_csv(gen_utils.HOTEL_REVIEW_DATA, header=0)

    labels = list(train_data["Rating"].array)
    text = list(train_data["Review"].array)

    return text, labels


def get_positive_and_negative_reviews(reviews: List[str], ratings: List[int], retain_ratings: bool = False) -> \
        Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    positive_reviews = []
    negative_reviews = []

    for row in tuple(zip(reviews, ratings)):
        review, rating = row[0], row[1]
        output_rating = None
        if retain_ratings:
            output_rating = rating
        if rating > 3:
            if not output_rating:
                output_rating = 1
            positive_reviews.append((review, output_rating))
        else:
            if not output_rating:
                output_rating = 0
            negative_reviews.append((review, output_rating))

    return positive_reviews, negative_reviews


def convert_ratings_to_binary(ratings: List[int]) -> List[int]:
    new_ratings = []
    for rating in ratings:
        if rating > 3:
            new_ratings.append(1)
        else:
            new_ratings.append(0)

    return new_ratings


def run_opinion_lexicon(positive_reviews: List[Tuple[str, int]], negative_reviews: List[Tuple[str, int]]):
    # lemmatizer testing didn't require training so splitting data was not necessary.
    lemmatizer_results = result_utils.Results()
    count = 0
    total_count = len(positive_reviews) + len(negative_reviews)

    for positive_review in positive_reviews:
        review, rating = positive_review[0], positive_review[1]
        lemmatizer_results.add_positive()
        is_positive = movie_review_analysis.evaluate_sentence(review)
        lemmatizer_results.add_result(is_positive, should_be_positive=True)
        count += 1
        print(f"Progress {count:,} / {total_count:,}")

    # Run through all negative review
    for negative_review in negative_reviews:
        review, rating = negative_review[0], negative_review[1]
        lemmatizer_results.add_negative()
        is_positive = movie_review_analysis.evaluate_sentence(review)
        lemmatizer_results.add_result(is_positive, should_be_positive=False)
        count += 1
        print(f"Progress {count:,} / {total_count:,}")

    lemmatizer_results.report()


def run_naive_bayes_classifier(positive_reviews: List[Tuple[str, int]], negative_reviews: List[Tuple[str, int]]):
    # split data into different sets
    positive_training, positive_test = gen_utils.split_data(positive_reviews)
    negative_training, negative_test = gen_utils.split_data(negative_reviews)

    # create final data sets and run classifier
    model_training_data = positive_training + negative_training
    model_test_data = positive_test + negative_test
    print(f"Training Data Count: {len(model_training_data)}")
    print(f"Testing Data Count: {len(model_test_data)}")

    # lines below are used to run the classifier 1 time and output most certain/least certain predictions
    result, classifier_predictions = movie_nb_analysis.run_naive_bayes(model_training_data, model_test_data,
                                                                       max_words=99999999999999999)
    result.report()
    movie_nb_analysis.output_most_certain_and_most_uncertain_predictions(classifier_predictions)


def run_svm_classifier(reviews: List[str], ratings: List[int]):
    review_vectors = toxic_comment_classification.use_sklearn_count_vectorizer(reviews)

    # get final 70/30 split of data
    v_train_data, v_test_data = toxic_comment_classification.transform_vectors_into_train_and_test(review_vectors,
                                                                                                   ratings)
    train_text, train_labels = zip(*v_train_data)
    test_text, test_labels = zip(*v_test_data)

    # train SVM classifier
    clf_classifier = svm.LinearSVC()
    clf_classifier.fit(train_text, train_labels)

    # predict test data and output results
    toxic_comment_classification.predict_svm_and_export_results(clf_classifier, test_text, test_labels,
                                                                target_names=["1", "2", "3", "4", "5"])


if __name__ == '__main__':

    print("Retrieving review data")
    hotel_reviews, review_ratings = get_hotel_review_data()

    print("Converting review data into binary classes")
    positive_review_list, negative_review_list = get_positive_and_negative_reviews(hotel_reviews, review_ratings,
                                                                                   retain_ratings=False)

    # print("Running opinion lexicon test")
    # run_opinion_lexicon(hotel_reviews, review_ratings)

    # print("Running Naive Bayes Analysis")
    # run_naive_bayes_classifier(positive_review_list, negative_review_list)

    # print("Use vectorizer to compare get SVM Analysis")
    # run_svm_classifier(hotel_reviews, review_ratings)

    print("Run K nearest neighbors classifier")

    print("Run Decision Tree")

    print("Run MLP Neural Network")
