# Machine-Learning

Machine Learning Algorithms implementations from scratch

## Decision Tree Learning Algorithm and Prediction
Data is assumed to contain attributes with binary values and the output is also binary. Mutual Information is used as the splitting criteria. The attribute with highest mutual information value is used as the node to split the data. Mutual information less than 0 are not considered. The node becomes the terminal node for making decision.

## Logistic Regression
Logistic Regression algorithm (lr.py) used with feature engineering (feature.py). feature.py modifies the movie review (data) to a sparse representation using the dictionary provided. lr.py performs binary logistic regression on the formatted data and predicts positive or negative review for the test data. 

## Neural Network

Neural Network (with single hidden layer) implementation from scratch. Output size is assumed to be 10 (Implementation supports any number of output).
