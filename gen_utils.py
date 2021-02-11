
import os
from collections import Counter

MAIN_DIR = os.path.join(os.getcwd(), "..")
HOMEWORK_1_FOLDER = os.path.join(MAIN_DIR, "HW_1")
HOMEWORK_1_DATA_FOLDER = os.path.join(MAIN_DIR, "data")
NEGATIVE_REVIEWS_PATH = os.path.join(HOMEWORK_1_DATA_FOLDER, 'negative_reviews.txt')
POSITIVES_REVIEWS_PATH = os.path.join(HOMEWORK_1_DATA_FOLDER, 'positive_reviews.txt')


def get_sentence_list_for_word_file(file_path: str) -> list:
    # get file data
    with open(file_path, 'r') as review_file:
        file_text = review_file.readlines()
        return file_text


def get_counter_for_word_file(file_path: str) -> Counter:
    file_text = get_sentence_list_for_word_file(file_path)

    # count each word in the sentences
    words_counter = Counter()
    for line in file_text:

        words = line.split(" ")
        words_counter.update(words)

    return words_counter
