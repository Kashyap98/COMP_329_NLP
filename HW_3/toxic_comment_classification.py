import pandas as pd
from sklearn import svm, metrics

import gen_utils
import result_utils
from HW_3.vectorizer import CountVectorizer


train_data = pd.read_csv(gen_utils.HATE_SPEECH_PATH, header=0)
# class = 0 - hate speech 1 - offensive language 2 - neither
labels = list(train_data["class"].array)
text = list(train_data["tweet"].array)

# transform text into vectors
vectorizer = CountVectorizer(text, labels, debug_mode=True)
vectors = vectorizer.get_vectors()

# split vector and labels into train/test sets
input_array = list(zip(vectors, labels))
train_data, test_data = gen_utils.split_data(input_array)
train_text, train_labels = zip(*train_data)
test_text, test_labels = zip(*test_data)

# train SVM
print("Training SVM")
clf = svm.LinearSVC()
clf.fit(train_text, train_labels)

# predict and capture results
results = result_utils.Results()
all_predictions = []
test_length = len(test_text)
for i in range(test_length):
    # get data to test
    data, label = test_text[i], test_labels[i]
    results.add_to_total()
    print(f"Processing result: {i} / {test_length}")

    # get prediction
    predictions = clf.predict([data])
    prediction = predictions[0]
    all_predictions.append((data, predictions))

    # add prediction to results
    results.add_multiclass_result(result=prediction, expected_result=label)

results.report()

data, predicted = zip(*all_predictions)
print(metrics.classification_report(test_labels, predicted, target_names=["hate_speech", "offensive_language", "neither"]))
print(metrics.confusion_matrix(test_labels, predicted))
