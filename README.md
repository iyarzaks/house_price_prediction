# house_price_prediction

Pricing property is a complicated task based heavily on got-feeling. 
Sellers and buyers have a need for data-driven approach helping them to solve that estimation problem.
The standard approach for this problem is using a basic machine learning tools to estimate the price of property using his features (characterizes).
We would like to use also the spatial information to make the estimation more accurate and stable.

Comparing performance between:
1)	Classical classification models (like: logistic regression and multi class SVM).
2)	Probabilistic graph model MRF. We will model each house as node and add weighted edges between nodes (houses). For the inference part we will implement a version of an algorithm we learned in the course (Belief propagation or Variable elimination). There is two options to weighted edges:
A)	Geographic proximity between two nodes.
B)	Similarity of two nodes (houses) according to certain features.

