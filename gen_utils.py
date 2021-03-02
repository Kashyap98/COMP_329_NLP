
import os
import random
from typing import Tuple, List

MAIN_DIR = os.path.join(os.getcwd(), "..")
HOMEWORK_1_FOLDER = os.path.join(MAIN_DIR, "HW_1")
DATA_FOLDER = os.path.join(MAIN_DIR, "data")
NEGATIVE_REVIEWS_PATH = os.path.join(DATA_FOLDER, 'negative_reviews.txt')
POSITIVES_REVIEWS_PATH = os.path.join(DATA_FOLDER, 'positive_reviews.txt')
STOP_WORDS_PATH = os.path.join(DATA_FOLDER, "stopwords.txt")
PUNCTUATION_PATH = os.path.join(DATA_FOLDER, "punctuation.txt")


def get_sentence_list_for_word_file(file_path: str) -> List[str]:
    """
    Takes a file path with sentences/words split by new line, returns list of sentences
    @param file_path: subject file
    @return: list of words/sentences in file
    """
    # get file data
    with open(file_path, 'r') as review_file:
        file_text = review_file.read().splitlines()
        return file_text


def format_input_data(positive_data: List[str], negative_data: list) -> Tuple[List[Tuple[str, int]],
                                                                              List[Tuple[str, int]]]:
    """
    Add tags to classes of input data. Currently only works with binary data
    @param positive_data: list of sentences/words, data will get a value of 1
    @param negative_data: list of sentences/words, data will get a value of 0
    @return: tuple of labeled positive/negative data
    """
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
    """
    Convert data into training and test data
    @param input_data: list of labeled input data
    @param split_percentage: percentage of data that will be test/training
    @return: input data split by input percentages
    """
    input_data = set(input_data)
    training_count = int(len(input_data) * split_percentage)

    training_data = set(random.sample(input_data, training_count))
    test_data = input_data - training_data

    return list(training_data), list(test_data)


def split_dev_data(input_data: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int]],
                                                               List[Tuple[str, int]],
                                                               List[Tuple[str, int]]]:
    """
    Split data into 3 different groups. Same as above but creates a dev group of data as well.
    @param input_data: list of labeled data
    @return: tuple of 3 lists as split labeled input data
    """
    training_data, test_data = split_data(input_data)

    # split test data in half to test on
    dev_data = set(random.sample(test_data, int(len(test_data) / 2)))
    test_data = set(test_data) - set(dev_data)

    return list(training_data), list(test_data), list(test_data)
