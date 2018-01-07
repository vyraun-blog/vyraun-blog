---
title: On the Geometry of Word Embeddings
updated: 2015-12-01 15:56
---

> Real-valued, low dimensional word vectors have become the basic building blocks for several NLP tasks and pre-trained word vectors are used ubiquitously in several applications. This post covers a few tricks, inspired by the geometric properties of the embedding space that help in getting a better performance from the embeddings and in doing effective dimensionality reduction.

## Geometric Properties of the Embedding Space

Word vectors, across all representations such as glove, word2vec etc., have a few common properties:

1. They have a large mean vector.
2. Most of their energy resides in a low-dimensional subspace (of, say 8 dimensions).

Since, the common mean vectors and the top dominant directions impact all the word vectors in the same way, subtracting the common mean vector and projecting the vectors away from the top dimensions will render them more discriminative. This is the idea behind the post-processing algorithm proposed in "Simple and Effective Post-processing for Word Representations". And it works really well.

## Application to Dimensionality Reduction

It turns out that even if you apply the post-processing algorithm and do a PCA based dimensionality reduction, the geometric properties re-emerge. This directly motivates a multi-step dimensionality reduction algorithm that combines PCA with the post-processing algorithm. This is the idea behind the paper, "Simple and Effective Dimensionality Reduction for Word Embeddings". And using it you can get 50% size reduction without any appreciable decline in performance.

... To be continued.. 
