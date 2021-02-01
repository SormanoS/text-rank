import numpy as np
from nltk.cluster.util import cosine_distance


def build_similarity_matrix(sentences, distance):
    """
    builds the similarity matrix
    :param sentences: sentences to process
    :param distance: type of distance to use (jaccard or cosine similarity)
    :return: similarity matrix
    """
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                if distance == 0:
                    similarity_matrix[i][j] = _jaccard_similarity(sentences[i], sentences[j])
                else:
                    similarity_matrix[i][j] = _cosine_similarity(sentences[i], sentences[j])

    return _normalize_matrix(similarity_matrix)


def get_best_sentences(sentence_ranks, sentences, k):
    """
    Returns the k% highest-ranking sentences
    :param sentence_ranks: dictionary (sentence index -> rank)
    :param sentences: sentences to process
    :param k: percentage number of sentences to extract
    :return: k% highest-ranking sentences
    """
    k = k * len(sentences) // 100  # Number of sentences to extract
    indexes = list(reversed(sentence_ranks.argsort()))[:k]
    return [sentences[i] for i in indexes]


def _normalize_matrix(similarity_matrix):
    """
    normalizes the matrix to avoid having rows with all zeros
    :param similarity_matrix: sentences similarity matrix
    :return: normalized similarity matrix
    """
    for i in range(len(similarity_matrix)):
        if similarity_matrix[i].sum() == 0:
            similarity_matrix[i] = np.ones(len(similarity_matrix))
        similarity_matrix[i] = similarity_matrix[i] / similarity_matrix[i].sum()

    return similarity_matrix


def _cosine_similarity(sent1, sent2):
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # vector first sentence
    for word in sent1:
        vector1[all_words.index(word)] += 1

    # vector second sentence
    for word in sent2:
        vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)


def _jaccard_similarity(sent1, sent2):
    overlap = len(set(sent1).intersection(set(sent2)))

    if overlap == 0:
        return 0

    return overlap / (np.log10(len(sent1)) + np.log10(len(sent2)))


def pagerank(score_matrix, eps=0.0001, d=0.85):
    """
    PageRank algorithm to rank sentences.
    :param score_matrix: similarity matrix
    :param eps: The algorithm will consider the calculation as complete if the difference of PageRank values between
                iterations change less than this value for every node.
    :param d: damping factor: with a probability of 1-d this algorithm holds the sentence ignoring the score
    :return: dictionary (sentence index -> rank)
    """
    rank = np.ones(len(score_matrix))

    while True:
        r = np.ones(len(score_matrix)) * (1 - d) + d * score_matrix.T.dot(rank)
        if abs(r - rank).sum() <= eps:
            return r
        rank = r


def build_similarity_matrix_glove(sentences, sentence_vectors):
    """
        Calculate the similarity matrix using GloVe
        :param sentences: sentences to process
        :param sentence_vectors: sentences GloVe
        :return: similarity matrix
        """
    similarity_matrix = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = \
                    _cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

    return similarity_matrix


def get_sentences(sentences, ranks, k):
    """
    Returns the k highest rank phrases
    :param sentences: sentences to process
    :param ranks: dictionary (sentence index -> rank)
    :param k: percentage number of sentences to extract
    :return: k highest rank phrases
    """
    summary = []
    ranked_sentences = sorted(((ranks[i], s) for i, s in enumerate(sentences)), reverse=True)
    k = k * len(sentences) // 100  # Number of sentences to extract
    for i in range(k):
        summary.append(ranked_sentences[i][1])
    return summary


def build_similarity_matrix_w2v(sentences, sentence_vectors):
    """
    Calculate the similarity matrix using word2vec
    :param sentences: sentences to process
    :param sentence_vectors: sentences word2vec
    :return: similarity matrix
    """
    similarity_matrix = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = \
                    _cosine_similarity(sentence_vectors[i].reshape(1, 300), sentence_vectors[j].reshape(1, 300))[0, 0]

    return similarity_matrix
