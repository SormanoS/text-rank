# TextRank

TextRank is a graph-based ranking model for text processing which can be used in order to find the most relevant sentences in text and also to find keywords. The algorithm is explained in detail in the paper at https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf.

In this project the TextRank algorithm has been implemented using python 3.8, making available different methods for the calculation of the similarity matrix:
- Using jaccard or cosine similarity;
- Using GloVe
- Using word2vec

## Resources to download

Download the following resources and place them in text-rank/resources directory (you have to create it):
- word2vec model: https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
- GloVe data: https://nlp.stanford.edu/projects/glove/

## How to run the project

To run the project just run the text_rank.py file specifying some parameters:
- **-p:** path of the file to summarize;
- **-k:** percentage of phrases to extract. This parameter is optional, if nothing is specified, the summarie will be carried out with 70% of the sentences;
- **-d:** distance with which to calculate the similarity matrix, it only works if you don't use word2vec or GloVe:
    - 0 = cosine similarity
    - 1 = jaccard similarity
- **-o:** setting this parameter, you can choose to calculate the matrix of similarity using jaccard or cosine similarity, GloVe or word2vec:
    - 0 = algorithm works in the naive way;
    - 1 = algorithm works with GloVe;
    - 2 = algorithm works with word2vec.