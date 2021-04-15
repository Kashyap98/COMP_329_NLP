from typing import Tuple, List

import pandas as pd

import gen_utils
import result_utils
from HW_1 import movie_review_analysis


def get_hotel_review_data() -> Tuple[List[str], List[int]]:
    train_data = pd.read_csv(gen_utils.HOTEL_REVIEW_DATA, header=0)

    labels = list(train_data["Rating"].array)
    text = list(train_data["Review"].array)

    return text, labels


def run_opinion_lexicon(reviews: List[str], ratings: List[int]):
    # lemmatizer testing didn't require training so splitting data was not necessary.
    lemmatizer_results = result_utils.Results()
    count = 0
    total_count = len(reviews)

    for row in tuple(zip(reviews, ratings)):
        # since the lemmatizer test was binary we are transferring the ratings to a binary value.
        # 4,5 = positive while 1,2,3 is negative.

        review, rating = row[0], row[1]
        if rating > 3:
            lemmatizer_results.add_positive()
            is_positive = movie_review_analysis.evaluate_sentence(review)
            lemmatizer_results.add_result(is_positive, should_be_positive=True)
            count += 1
            print(f"Progress {count:,} / {total_count:,}")
        # Run through all negative review
        else:
            lemmatizer_results.add_negative()
            is_positive = movie_review_analysis.evaluate_sentence(review)
            lemmatizer_results.add_result(is_positive, should_be_positive=False)
            count += 1
            print(f"Progress {count:,} / {total_count:,}")

    lemmatizer_results.report()


if __name__ == '__main__':

    print("Retrieving review data")
    hotel_reviews, review_ratings = get_hotel_review_data()

    print("Running opinion lexicon test")
    run_opinion_lexicon(hotel_reviews, review_ratings)


