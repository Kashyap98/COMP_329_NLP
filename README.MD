# COMP 329 - NLP Homework Assignments

## HW 1: Movie Review Data

The purpose of this assignment is to determine if a movie review is positive or negative. The caveat was that we 
were not allowed to use any machine learning at all, and the classifier must be rule based. I chose to use the NLTK package
and their opinion lexicon based on the Hu Liu Opinion Dataset in order to determine the sentiment of the review in question.

I used the opinion lexicon and compared lemmatization of different aspects of the input to determine its effectiveness in this 
example. I either used no lemmatization, lemmatized the opinion lexicon, lemmatized the input sentence, or both inputs. 
The overall reviews were scored on if they had more positive or more negative words based on the 
lexicon dataset. The results can be found in the results.md file as well as the eventual report that will be written.