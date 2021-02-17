import gen_utils

from HW_2.naive_bayes import NaiveBayesClassifier


if __name__ == '__main__':
    positive_review_data = gen_utils.get_sentence_list_for_word_file(gen_utils.POSITIVES_REVIEWS_PATH)
    negative_review_data = gen_utils.get_sentence_list_for_word_file(gen_utils.NEGATIVE_REVIEWS_PATH)
    stop_words = gen_utils.get_sentence_list_for_word_file(gen_utils.STOP_WORDS_PATH)

    # 1 = positive, 0 = negative
    classifier_input_data = []

    for sentence_data in positive_review_data:
        classifier_input_data.append((sentence_data, 1))

    for sentence_data in negative_review_data:
        classifier_input_data.append((sentence_data, 0))

    classifier = NaiveBayesClassifier(training_data=classifier_input_data, stop_word_filter=stop_words)

    test1 = classifier.predict_one_sentence_value('simplistic , silly and tedious .')
    test1 = classifier.predict_one_sentence_value('exploitative and largely devoid of the depth or sophistication that would make watching such a graphic treatment of the crimes bearable .')
    test1 = classifier.predict_one_sentence_value('the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .')
    test1 = classifier.predict_one_sentence_value('in its ragged , cheap and unassuming way , the movie works .')
    print("hello")
