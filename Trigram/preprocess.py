import tensorflow as tf
import numpy as np
from functools import reduce

def get_data(train_file='data/train.txt', test_file='data/test.txt'):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.
    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """

    # TODO: load and concatenate training data from training file.
    with open(train_file, 'r') as f:
        train_corpus = f.read().split()
    # TODO: load and concatenate testing data from testing file.
    with open(test_file, 'r') as f:
        test_corpus = f.read().split()
        full_corpus = train_corpus + test_corpus

   

    vocab_corpus = set(full_corpus)
    len_v_corpuse=len(vocab_corpus)

    vocabval = list(range(0,len_v_corpuse))
    vocabkey = []
    
    for i in (vocab_corpus):
        vocabkey.append(i)

    vocab_dict = dict(zip(vocabkey, vocabval)) 


    trainid = []
    testid = []
    for word in train_corpus:
        trainid.append(vocab_dict[word])
    
    for word in test_corpus:
        testid.append(vocab_dict[word])

    return (trainid, testid, vocab_dict)