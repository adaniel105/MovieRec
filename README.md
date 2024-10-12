# MovieRec

A recommendation system that recommends movies to users based on a similarity to movies which they already like (item-item similarity). 
This similarity or "nearness in distance" is calculated using the cosine similarity metric, which is a useful tool for measuring the angle between two vectors defined in a multi-dimsensional space.

## Dataset 

The dataset used is a subset of the MovieLens dataset, consisting of 100,000 ratings of movies (of varying genres) by users.

## Eval scores on test set

LightFM

- Precision@k: 0.15

- AUC score: 0.95
