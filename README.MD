# COMP 329 - NLP Homework Assignments
<hr>

## HW 1: Movie Review Data
<hr>
The purpose of this assignment is to determine if a movie review is positive or negative. The caveat was that we 
were not allowed to use any machine learning at all, and the classifier must be rule based. I chose to use the NLTK package
and their opinion lexicon based on the Hu Liu Opinion Dataset in order to determine the sentiment of the review in question.

I used the opinion lexicon and compared lemmatization of different aspects of the input to determine its effectiveness in this 
example. I either used no lemmatization, lemmatized the opinion lexicon, lemmatized the input sentence, or both inputs. 
The overall reviews were scored on if they had more positive or more negative words based on the 
lexicon dataset. The results can be found in the results.md file as well as the eventual report that will be written.

**The report, results, data, and evaluation script can be found in the HW_1 folder.**
<hr>

## HW 2: Naive Bayes Classifier
<hr>
Homework 2 focused on our own implementation of the Naive Bayes classifier that uses the Bayes Theorem to predict the class of 
input data based on prior probabilities. This type of classifier is good for data that will likely have repeated data. In addition,
this classifier is useful for less powerful machines because it can be quickly trained and optimized.

My implementation of Naive Bayes seem to average around 76-77% accuracy for the movie review data set used in HW 1. This is and increase
of about ~15% over the previous method comparing lemmatization and a previous opinion lexicon. I noticed no major difference in accuracy
when comparing a classifier using stop words to one that filters them out. I also performed all probability calculations in log space
to prevent float underflow and used Laplace smoothing to correct for data not found in either class. 

**The report, results, data, and evaluation script can be found in the HW_2 folder.**
<hr>

## HW 3: Vectorizer with Sklearn SVM Classifier
<hr>
Homework 3 involved the creation of a vectorizer to convert text data into numeric features. I created a CountVectorizer similar to that of the sklearn CountVectorizer that  transformed tweets into arrays of 1/0s depending on the presence of a word (feature). As evident by the multiple test runs in the results.md file, this vectorizer performs extremely similarly to the one included in the sklearn package. <br>

This repository (Reference 1) is hosted on Github and provides a curated set of hate speech and offensive language with regular comments. This labeled data is classified by measuring the number of people who voted for each classification per comment. It could be classified as hate speech, offensive language, and neither. <br>

According to the results of my vectorizer, I am averaging 88% accuracy when determining what class each comment should belong to, as mentioned above. The precision, recall, and f1-score of the hate speech class is much lower than the others. I think this is due to the fact that the dataset is heavily skewed towards offensive language and neither classes, so the classifier has less training in hate speech recognition. <br>

The pipeline I have created is supposed to be extensible should I have time to implement the TfidfVectorizer as well. In this case however, the vectorizer takes the training and test set (to include all possible words) and captures all unique words found in the data set. This set of words is then generated into features that represent an array for each sentence. If a word in the array is found in the sentence the value flips from 0 to 1 to represent its presence. These sentences are now represented in numeric form so that they can be inputted into classifiers such as a Support Vector Machine (SVM). <br>

I chose to use a LinearSVC in the sklearn package over a regular SVM and logistic regression models. I chose to disregard logistic regression because of points made by this article on Medium (Reference 2). Since this problem had 3 classes, rather than a binary 2 I believed that logistic regression would be a weaker model to choose. The article also recommends using LinearSVM when training on data that has a high number of features and samples. The linear method helps prevent too much overfitting when working with the data. I also noticed that training times were much faster on larger amounts of data as computational time and resources grew faster in a regular SVM. I also used the default parameters for each model as it was giving me the best performance. <br>

Therefore, I think that this classifier is pretty successful at finding offensive language in comments according to this dataset. I think the choice of using an SVM over Logistic Regression was the correct one due to the multitude of classes used in the dataset. If I had more time I would have loved to test the TfidfVectorizer, stemming, lemmatization, forcing all characters to lowercase, and tokenizing. Below, I have pasted output from 1 test run of the classifier. <br>
