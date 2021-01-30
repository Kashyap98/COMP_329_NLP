from nltk import WordNetLemmatizer
from nltk.corpus import opinion_lexicon
import gen_utils
import result_utils


def evaluate_sentence(sentence: str) -> bool:
    positive_count = 0
    negative_count = 0
    # lemmatizer = WordNetLemmatizer()

    # lemmatization is commented out for submission for speed and improved evaluation
    # new_sentence = ""
    # for word in sentence.split(" "):
    #     new_sentence += lemmatizer.lemmatize(word)
    # sentence = new_sentence

    for positive_word in opinion_lexicon.positive():
        # positive_word = lemmatizer.lemmatize(positive_word)

        if positive_word in sentence:
            positive_count += 1

    for negative_word in opinion_lexicon.negative():
        # negative_word = lemmatizer.lemmatize(negative_word)

        if negative_word in sentence:
            negative_count += 1

    if positive_count >= negative_count:
        is_sentence_positive = True
    else:
        is_sentence_positive = False

    return is_sentence_positive


results = result_utils.Results()

positive_review_data = gen_utils.get_sentence_list_for_word_file(gen_utils.POSITIVES_REVIEWS_PATH)
negative_review_data = gen_utils.get_sentence_list_for_word_file(gen_utils.NEGATIVE_REVIEWS_PATH)

# progress is commented out for final submission
# total_count = len(positive_review_data) + len(negative_review_data)
# count = 0

# Run through all positive review
for positive_review in positive_review_data:
    results.add_positive()
    is_positive = evaluate_sentence(positive_review)
    results.add_result(is_positive, should_be_positive=True)
    # count += 1
    # print(f"Progress {count:,} / {total_count:,}")

# Run through all negative review
for negative_review in negative_review_data:
    results.add_negative()
    is_positive = evaluate_sentence(negative_review)
    results.add_result(is_positive, should_be_positive=False)
    # count += 1
    # print(f"Progress {count:,} / {total_count:,}")

results.report()
