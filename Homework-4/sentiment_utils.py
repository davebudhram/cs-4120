# FIRST: RENAME THIS FILE TO sentiment_utils.py

# YOUR NAMES HERE: Dave Budhram and Akshay Dupuguntla


"""
Felix Muzny
CS 4/6120
Homework 4
Fall 2023

Utility functions for HW 4, to be imported into the corresponding notebook(s).

Add any functions to this file that you think will be useful to you in multiple notebooks.
"""
# fancy data structures
from collections import defaultdict, Counter
# for tokenizing and precision, recall, f_measure, and accuracy functions
import nltk
from nltk.metrics.scores import (precision, recall, f_measure, accuracy)
# for plotting
import matplotlib.pyplot as plt
# so that we can indicate a function in a type hint
from typing import Callable
nltk.download('punkt')


def generate_tuples_from_file(training_file_path: str) -> list:
    """
    Generates tuples from file formated like:
    id\ttext\tlabel
    id\ttext\tlabel
    id\ttext\tlabel
    Parameters:
        training_file_path - str path to file to read in
    Return:
        a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    """
    # PROVIDED
    f = open(training_file_path, "r", encoding="utf8")
    X = []
    y = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.strip().split("\t")
        if len(dataInReview) != 3:
            continue
        else:
            t = tuple(dataInReview)
            if (not t[2] == '0') and (not t[2] == '1'):
                print("WARNING")
                continue
            X.append(nltk.word_tokenize(t[1]))
            y.append(int(t[2]))
    f.close()
    return X, y


"""
NOTE: for all of the following functions, we have prodived the function signature and docstring, *that we used*, as a guide.
You are welcome to implement these functions as they are, change their function signatures as needed, or not use them at all.
Make sure that you properly update any docstrings as needed.
"""


def get_prfa(dev_y: list, preds: list, verbose=False) -> tuple:
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        dev_y: list of labels
        preds: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    refsets = defaultdict(set)  # true
    testsets = defaultdict(set)  # pred
    # Create list of tuples pairing predicated value and true values
    labels = [(pred, true) for pred, true in zip(preds, dev_y)]
    for i, (pred, true) in enumerate(labels):
        refsets[true].add(i)
        testsets[pred].add(i)
    # accuracy_val = accuracy(preds, dev_y)
    # precision_val = precision(refsets[1], testsets[1])
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for pred, true in zip(preds, dev_y):
        if pred == 1 and true == 1:
            true_positives += 1
        elif pred == 0 and true == 0:
            true_negatives += 1
        elif pred == 1 and true == 0:
            false_positives += 1
        elif pred == 0 and true == 1:
            false_negatives += 1
    accuracy_value = (true_positives + true_negatives) / \
        (true_positives + true_negatives + false_positives + false_negatives)
    precision_value = (true_positives) / (true_positives + false_positives)
    recall_value = (true_positives) / (true_positives + false_negatives)
    f1_score = 2 * ((precision_value * recall_value) /
                    (precision_value + recall_value))
    return (precision_value, recall_value, f1_score, accuracy_value)


def create_training_graph(metrics_fun: Callable, train_feats: list, dev_feats: list, kind: str, savepath: str = None, verbose: bool = False) -> None:
    """
    Create a graph of the classifier's performance on the dev set as a function of the amount of training data.
    Args:
        metrics_fun: a function that takes in training data and dev data and returns a tuple of metrics
        train_feats: a list of training data in the format [(feats, label), ...]
        dev_feats: a list of dev data in the format [(feats, label), ...]
        kind: the kind of model being used (will go in the title)
        savepath: the path to save the graph to (if None, the graph will not be saved)
        verbose: whether to print the metrics
    """
    # TODO: implement this function
    pass


def create_index(all_train_data_X: list) -> list:
    """
    Given the training data, create a list of all the words in the training data.
    Args:
        all_train_data_X: a list of all the training data in the format [[word1, word2, ...], ...]
    Returns:
        vocab: a list of all the unique words in the training data
    """
    # figure out what our vocab is and what words correspond to what indices
    # TODO: implement this function
    pass


def featurize(vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data.
    Args:
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    # using a Counter is essential to having this not take forever
    # TODO: implement this function
    pass


def convert2DArray(documents: [[str]]):
    """
    Flattens a 2D array of strings to a 1D array of strings with " " between each word
    """
    result = []
    for list in documents:
        list_text = ""
        for word in list:
            list_text += (word + " ")
        result.append(list_text)
    return result


class CustomCountVectorizer:
    """
    Class with similar behavior to sklearn CountVectorizer. Assigns words in a vocab to an index, 
    and transforms list of documents to a matrix where each row in the matrix is a vector with the 
    either number of times a word appeared in that document of 1 or 0 if the word is in the document 
    or not.
    """

    def __init__(self, binary=False):
        self.vocab = {}
        self.binary = binary

    def fit(self, documents: [[str]]):
        """
        Assigns all the words in documents an index
        """
        idx = 0
        for document in documents:
            for word in set(document):
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1

    def transform(self, documents: [[str]]):
        """
        Transforms list of documents to a matrix where each row in the matrix is a vector with the 
        either number of times a word appeared in that document of 1 or 0 if the word is in the document 
        or not.
        """
        matrix = []
        for document in documents:
            vector = [0] * len(self.vocab)
            for word in document:
                if word in self.vocab:
                    if self.binary:  # Seen the word in the document
                        vector[self.vocab[word]] = 1
                    else:  # Want the count of the word in the document
                        vector[self.vocab[word]] += 1
            matrix.append(vector)
        return matrix
