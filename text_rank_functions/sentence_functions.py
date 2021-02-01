import io

import gensim
import numpy as np
import pandas as pd
from gensim.models.phrases import Phraser
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.tokenize import sent_tokenize

stop_words = set(stopwords.words("english"))

'''
tags to keep in sentences to select important words

NN = singular name
NNS = plural name
NNP = proper name
JJ = adjective
JJR = comparative adjective
JJS = superlative adjective
'''
tags = {"NN", "NNS", "NNP", "JJ", "JJR", "JJS"}


def filter_sentences(sentences):
    """
    filters phrases: to lower, stopwords removal, selection of important words, stemmatization
    :param sentences: sentences to process
    :return: filtered sentences
    """
    normalized_sentences = [s.lower() for s in sentences]
    filtered_sentences = [_clean_select_sentence(sent) for sent in normalized_sentences]

    return [_stem_sentence(sent) for sent in filtered_sentences]


def _clean_select_sentence(sentence):
    """removes stopwords from sentence and uninteresting tagged words"""
    filtered_sentence = []
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag in tags and word.lower() not in stop_words:
            filtered_sentence.append(word)

    return filtered_sentence


def load_data(path):
    """
    loading data: reads the document and divides it into sentences
    :param path: Path of the file to summarize
    :return: array of sentences
    """
    sentences = []
    with io.open(path, encoding='utf-8') as f:
        for line in f:
            for sent in sent_tokenize(line):
                sentences.append(sent)
    return sentences


def _stem_sentence(sentence):
    """stemmatization of the sentence"""
    stemmer = porter.PorterStemmer()
    return [stemmer.stem(word) for word in sentence]


def _remove_stopwords(sentence):
    """remove stopwords"""
    clean_sentence = " ".join([word for word in sentence if word not in stop_words])
    return clean_sentence


def _clean_sentence_for_glove_and_w2v(sentences):
    """removes stopwords from the sentence, lowercase and removes numbers and special characters"""
    # removes punctuation, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # turns everything into lower
    clean_sentences = [s.lower() for s in clean_sentences]

    # remove stopwords
    clean_sentences = [_remove_stopwords(sentence.split()) for sentence in clean_sentences]

    return clean_sentences


def _word_embeddings_glove():
    """word embeddings with glove (word vector): extract word vectors"""
    word_embeddings = {}
    f = open('../resources/glove.6B/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        word_embeddings[word] = np.asarray(values[1:], dtype='float32')
    f.close()
    return word_embeddings


def get_sentence_vectors(sentences):
    """
    :param sentences: text to process
    :return: sentences vector
    """
    word_embeddings = _word_embeddings_glove()
    sentences_vector = []
    for i in sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentences_vector.append(v)
    return sentences_vector


def get_sentence_vectors_w2v(sentences):
    """
    :param sentences: text to process
    :return: sentences vector
    """
    # word2vec model taken from google's pre-trained networks
    model = gensim.models.KeyedVectors.load_word2vec_format('../resources/GoogleNews-vectors-negative300.bin',
                                                            binary=True)
    # sentences preparation
    clean_sentences = _clean_sentence_for_glove_and_w2v(sentences)
    # calculation of sentence vectors
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([model[w] for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return sentence_vectors
