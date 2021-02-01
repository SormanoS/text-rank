import argparse

import networkx as nx

from text_rank_functions.matrix_functions import build_similarity_matrix
from text_rank_functions.matrix_functions import build_similarity_matrix_glove
from text_rank_functions.matrix_functions import build_similarity_matrix_w2v
from text_rank_functions.matrix_functions import get_best_sentences
from text_rank_functions.matrix_functions import get_sentences
from text_rank_functions.matrix_functions import pagerank
from text_rank_functions.sentence_functions import filter_sentences
from text_rank_functions.sentence_functions import get_sentence_vectors
from text_rank_functions.sentence_functions import get_sentence_vectors_w2v
from text_rank_functions.sentence_functions import load_data


def summarize(sentences, k, option, distance=0):
    """
    This method summarize the sentences.
    GloVe: Global Vectors for Word Representation
    :param sentences: sentences to summarize
    :param k: percentage number of sentences to extract
    :param distance: works only if option is equals to 0.
     0 = cosine similarity (only without GloVe and word2vec); 1 = jaccard similarity (only without GloVe and word2vec)
    :param option: 0 = algorithm works in the naive way; 1 = algorithm works with GloVe; 2 = algorithm works with word2vec
    :return: summarized sentences
    """
    if option == 0:  # algorithm doesn't use GloVe
        # cleaning sentences: stopword removal, stemmatization, choice of significant words
        filtered_sentences = filter_sentences(sentences)
        similarity_matrix = build_similarity_matrix(filtered_sentences, distance)
        ranks = pagerank(similarity_matrix)
        return get_best_sentences(ranks, sentences, k)

    elif option == 1:  # algorithm uses GloVe
        sentence_vectors = get_sentence_vectors(sentences)
        similarity_matrix_glove = build_similarity_matrix_glove(sentences, sentence_vectors)
        # graph creation
        nx_graph = nx.from_numpy_array(similarity_matrix_glove)
        # rank calculation, returns a dictionary: sentence index -> rank
        ranks = nx.pagerank(nx_graph)
        return get_sentences(sentences, ranks, k)

    elif option == 2:  # algorithm uses word2vec
        sentence_vectors = get_sentence_vectors_w2v(sentences)
        similarity_matrix = build_similarity_matrix_w2v(sentences, sentence_vectors)
        # graph creation
        nx_graph = nx.from_numpy_array(similarity_matrix)
        # rank calculation, returns a dictionary: sentence index -> rank
        ranks = nx.pagerank(nx_graph)
        return get_sentences(sentences, ranks, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path of the file to summarize")
    parser.add_argument("-k", "--perc", type=int, default=70, help="Percentage of phrases to extract")
    parser.add_argument("-d", "--distance", type=int, default=0, help="works only if option is equals to 0."
                                                                      "0 = cosine similarity (only without GloVe and "
                                                                      "word2vec); "
                                                                      "1 = jaccard similarity (only without GloVe and "
                                                                      "word2vec)")
    parser.add_argument("-o", "--option", type=int, help="0 = algorithm works using jaccard or cosine similarity; "
                                                         "1 = algorithm works using GloVe; "
                                                         "2 = algorithm works using word2vec")

    args = parser.parse_args()

    sentences = load_data(args.path)

    print("\n\n Text to summarize \n\n")
    for sentence in sentences:
        print(sentence)

    print("\n\n Summary \n\n")
    summary = summarize(sentences, args.perc, args.option, args.distance)
    for sentence in summary:
        print(sentence)


if __name__ == '__main__':
    main()
