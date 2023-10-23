from collections import Counter
import numpy as np
import math

"""
CS 4/6120, Fall 2023
Homework 3 - starter code
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"


# UTILITY FUNCTIONS

def create_ngrams(tokens: list, n: int) -> list:
  """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
  # STUDENTS IMPLEMENT 
  ngrams = []
  for i in range(len(tokens) - n + 1):
      ngram = tuple(tokens[j] for j in range(i, i + n))
      ngrams.append(ngram)
  return ngrams

def read_file(path: str) -> list:
  """
  Reads the contents of a file in line by line.
  Args:
    path (str): the location of the file to read

  Returns:
    list: list of strings, the contents of the file
  """
  # PROVIDED
  f = open(path, "r", encoding="utf-8")
  contents = f.readlines()
  f.close()
  return contents

def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize a single string. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
  # PROVIDED
  inner_pieces = None
  if by_char:
    inner_pieces = list(line)
  else:
    # otherwise split on white space
    inner_pieces = line.split()

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  # always count the unigrams
  return tokens


def tokenize(data: list, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize each line in a list of strings. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - all lines tokenized as one large list
  """
  # PROVIDED
  total = []
  # also glue on sentence begin and end items
  for line in data:
    line = line.strip()
    # skip empty lines
    if len(line) == 0:
      continue
    tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
    total += tokens
  return total

def count_dict(tokens: list) -> dict:
    """
    Count the number of occurrences of each token in a list of tokens.
    Do not use the list.count() method. Do not use a Counter.
    If you would like to use any method/function other than len, it must be
    okay'd with the teaching staff.

    Args:
    tokens: list of items (often tokens)

    return:
    dict (mapping from token to count)
    """
    # YOUR CODE HERE
    token_dict = {}
    for token in tokens:
        if token not in token_dict:
            token_dict[token] = 1
        else:
            token_dict[token] +=1
    return token_dict


class LanguageModel:

  def __init__(self, n_gram):
    """Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
    # STUDENTS IMPLEMENT
    # if n_gram > 2:
    #   return ValueError('Language model only works on unigrams and bigrams')
    self.n_gram = n_gram
    self.n_gram_counts = []


  
  def train(self, tokens: list, verbose: bool = False) -> None:
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """
    # replace all tokens that appeared once with <UNK>
    tokens_count = count_dict(tokens) 
    for i in range(len(tokens)):
      if tokens_count[tokens[i]] == 1:
        tokens[i] = '<UNK>'
    self.tokens = tokens
    self.vocab = set(create_ngrams(tokens, 1)) # make the vocab use tuple for conveince

    # unigram case
    if self.n_gram == 1:
      n_grams = create_ngrams(self.tokens, self.n_gram)
      self.n_gram_counts  = count_dict(n_grams) 
      self.score_probabilities = {}
      for token in set(self.n_gram_counts):
        self.score_probabilities[token] = (self.n_gram_counts[token] + 1) / (len(self.tokens) + len(self.vocab)) 
    else:
      # create (n-1)-grams
      n_minus_1_grams = create_ngrams(self.tokens, self.n_gram - 1)
      self.n_gram_minus_one_count = count_dict(n_minus_1_grams) 
      # create n-grams
      n_grams = create_ngrams(tokens, self.n_gram)
      self.n_gram_counts = count_dict(n_grams) 
      self.score_probabilities = {}
      for curr_n_gram in set(self.n_gram_counts):
        numerator = self.n_gram_counts[curr_n_gram] + 1
        denominator = self.n_gram_minus_one_count[curr_n_gram[:self.n_gram-1]] + len(self.vocab)
        self.score_probabilities[curr_n_gram] = (numerator / denominator)
    
  def score(self, sentence_tokens: list) -> float:
    """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
      
    Returns:
      float: the probability value of the given tokens for this model
    """
    # STUDENTS IMPLEMENT
    for i in range(len(sentence_tokens)):
      # print(sentence_tokens[i])
      if (sentence_tokens[i],) not in self.vocab:
        sentence_tokens[i] = UNK # replace words not in the vocab to unknown
    sentence_n_grams = create_ngrams(sentence_tokens, self.n_gram) 
    probability = 1
   
    for n_gram in sentence_n_grams:
      # When the n-gram is 1 we need to check if the vocab has any unknown words
      if self.n_gram == 1: 
        if n_gram == (UNK,) and (UNK,) not in self.vocab:
          n_gram_probability = 1 / len(self.vocab) # if we don't have unknown words in our dictionary
        else :
          n_gram_probability = self.score_probabilities[n_gram] 
      # when the n-gram is greater than 1 we need to check if we have seen this n-gram. If not, then
      # check if we have seen the first n-1 tokens of the n-gram. If we have not, then use 
      else:
        if n_gram not in self.n_gram_counts: # never seen n gram before
            if n_gram[:self.n_gram-1] not in self.n_gram_minus_one_count: # never seen (n-1) gram before
              n_gram_probability = 1 / len(self.vocab)
            else:
              n_gram_probability = 1 / (self.n_gram_minus_one_count[n_gram[:self.n_gram-1]] + len(self.vocab))
        else:
          n_gram_probability = self.score_probabilities[n_gram] 
      probability *= n_gram_probability
    return probability


  def generate_sentence(self) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    # STUDENTS IMPLEMENT
    sentence = [] # empty sentence
    if self.n_gram == 1:
      sentence.append(SENTENCE_BEGIN) # always start with sentence start
      while True:
        if sentence[-1] == SENTENCE_END: # when we add the end of the sentence break
          break
        count = 0
        options = []
        probabilities = []
        for key in self.n_gram_counts:
          if key != ('<s>',):
            count += self.n_gram_counts[key]
            options.append(key[0])
            probabilities.append(self.n_gram_counts[key])
        probabilities = list(map(lambda x: x/count, probabilities))
        nextWord = np.random.choice(options, p=probabilities) # get new word based on probabilities
        sentence.append(nextWord)
    else:
      # Get the starting n-gram that begins with <s>
      count = 0
      options = []
      probabilities = []
      for key in self.n_gram_minus_one_count:
        if key[0] == SENTENCE_BEGIN:
          for i in range(self.n_gram_minus_one_count[key]):
            options.append(key)
          count += self.n_gram_minus_one_count[key]
      probabilities = list(map(lambda x: x/count, probabilities))
      sentenceStart = options[np.random.choice(len(options))]
      sentence.extend(list(sentenceStart))
      # Add words based on probabilities until sentence is over
      while True:
        if sentence[-1] == SENTENCE_END:
          break
        last_tokens = tuple(sentence[-self.n_gram + 1:]) # last token we have  | need the number of n-grams that begin it
        count = 0
        options = []
        probabilities = []
        for key in self.n_gram_counts:
          if key[:self.n_gram - 1] == last_tokens: # found ngram with matching last tokens
            count += self.n_gram_counts[key]
            options.append(key[-1])
            probabilities.append(self.n_gram_counts[key])
        probabilities = list(map(lambda x: x/count, probabilities))
        nextWord = np.random.choice(options, p=probabilities) # get new word based on probabilities
        sentence.append(nextWord)
    return sentence

  def generate(self, n: int) -> list:
    """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
    # PROVIDED
    return [self.generate_sentence() for i in range(n)]


  def perplexity(self, sequence: list) -> float:
    """Calculates the perplexity score for a given sequence of tokens.
    Args:
      sequence (list): a tokenized sequence to be evaluated for perplexity by this model
      
    Returns:
      float: the perplexity value of the given sequence for this model
    """
    # 6120 IMPLEMENTS
    pass
  
# not required
if __name__ == '__main__':
  print("if having a main is helpful to you, do whatever you want here, but please don't produce too much output :)")