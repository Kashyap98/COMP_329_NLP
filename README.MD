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

My implementation of Naive Bayes seem to average around 75-76% accuracy for the movie review data set used in HW 1. This is and increase
of about ~15% over the previous method comparing lemmatization and a previous opinion lexicon. I noticed no major difference in accuracy
when comparing a classifier using stop words to one that filters them out. I also performed all probability calculations in log space
to prevent float underflow and used Laplace smoothing to correct for data not found in either class. 

**The report, results, data, and evaluation script can be found in the HW_2 folder.**
<hr>