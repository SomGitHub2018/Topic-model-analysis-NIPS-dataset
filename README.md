## Abstract

We intend to build a model to find topics from unlabeled document collection. In this growing age of digital data, labelling documents/data by human is not feasible. Automatic topic modelling is the key to organize huge amount of unstructured semantic data. Topic model analysis helps companies or organizations find structure in data and organize based on discovered topics and suggest customers based on his interests or usage patterns. We’ve chosen NIPS dataset, which is a collection of all papers from NIPS Machine learning conference between 1987-2016. Once we discover topics among all the documents, we intend to analyze how topics evolved over time and predict future trends for topics. This gives us information on how subjects have been discussed and when new topics began taking shape among scientific community.

## Run the code in following order

Create_Model.py - This creates a model and saves it under output directory
get_labels_1.py, rerank.py - This gets labels based on saved model
Topic Evolution.py - This takes in saved model and analyses data, plots evolution of topics over time
Prediction.py - This takes in earlier evolution graph, and extrapolates for futher years
Evaluation.py - This evaluates the model

** Refer Report in repo, for details 
