# K-Means Clustering


K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. The results of the K-means clustering algorithm are:

The centroids of the K clusters, which can be used to label new data Labels for the training data (each data point is assigned to a single cluster) Rather than defining groups before looking at the data, clustering allows you to find and analyze the groups that have formed organically. The "Choosing K" section below describes how the number of groups can be determined.

Each centroid of a cluster is a collection of feature values which define the resulting groups. Examining the centroid feature weights can be used to qualitatively interpret what kind of group each cluster represents.

# Database
This data consist of 

RowNumber
CustomerId
Surname
CreditScore
Geography
Gender
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
Exited

Dataset can be found at:https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling

We will apply churn modeling using K-means.

# Churn Problem
Customer churn is a costly toll on businesses. It’s not simply the lost revenue to consider, but also the added marketing expenses needed to rectify the loss. Retaining customers has historically been a lagging business indicator—it’s only after businesses lost customers that they could assess what went wrong—but with increasing access to data, several companies have developed predictive churn analysis capabilities to identify at-risk customers and proactively prevent them from leaving.
