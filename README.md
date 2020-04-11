# Vectorization-effect-on-text-classification

We cannot work with text directly when using machine learning algorithms.

Instead, we'd like to convert the text to numbers.

We might want to perform classification of documents, so each document is an “input” and a category label is that the “output” for our predictive algorithm. Algorithms take vectors of numbers as input, therefore we'd like to convert documents to fixed-length vectors of numbers.

A simple and effective model for brooding about text documents in machine learning is named the Bag-of-Words Model, or BoW.

## CountVectorizer
The CountVectorizer provides an easy thanks to both tokenize a set of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.

You can use it as follows:

Create an instance of the CountVectorizer class.
Call the fit() function so as to find out a vocabulary from one or more documents.
Call the transform() function on one or more documents as required to encode each as a vector.
An encoded vector is returned with a length of the whole vocabulary and an integer count for the amount of times each word appeared within the document.

Because these vectors will contain tons of zeros, we call them sparse. Python provides an efficient way of handling sparse vectors within the scipy.sparse package.

The vectors returned from a call to transform() are going to be sparse vectors, and you'll transform them back to numpy arrays to seem and better understand what's happening by calling the toarray() function.

Below is an example of using the CountVectorizer to tokenize, build a vocabulary, then encode a document.


## TfidfVectorizer

Word counts are an honest start line , but are very basic.

One issue with simple counts is that some words like “the” will appear repeatedly and their large counts won't be very meaningful within the encoded vectors.

An alternative is to calculate word frequencies, and far and away the foremost popular method is named TF-IDF. this is often an acronym than stands for “Term Frequency – Inverse Document” Frequency which are the components of the resulting scores assigned to every word.

Term Frequency: This summarizes how often a given word appears within a document.
Inverse Document Frequency: This downscales words that appear tons across documents.
Without going into the maths , TF-IDF are word frequency scores that attempt to highlight words that are more interesting, e.g. frequent during a document but not across documents.

The TfidfVectorizer will tokenize documents, learn the vocabulary and inverse document frequency weightings, and permit you to encode new documents. Alternately, if you have already got a learned CountVectorizer, you'll use it with a TfidfTransformer to only calculate the inverse document frequencies and begin encoding documents.

The same create, fit, and transform process is employed like the CountVectorizer.


## HashingVectorizer
Counts and frequencies are often very useful, but one limitation of those methods is that the vocabulary can become very large.

This, in turn, would require large vectors for encoding documents and impose large requirements on memory and hamper algorithms.

A clever work around is to use a 1 way hash of words to convert them to integers. The clever part is that no vocabulary is required and you'll choose an arbitrary-long fixed length vector. A downside is that the hash may be a one-way function so there's no thanks to convert the encoding back to a word (which might not matter for several supervised learning tasks).

The HashingVectorizer class implements this approach which will be wont to consistently hash words, then tokenize and encode documents as required .

The example below demonstrates the HashingVectorizer for encoding one document.
