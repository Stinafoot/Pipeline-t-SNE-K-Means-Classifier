# Pipeline-t-SNE-K-Means-Classifier

# Research question: How effective is a pipeline combining t-SNE, K-Means clustering, and supervised learning for classifying wine quality?

We are going to build a pipeline as part of this assignment.

Pipeline: t-SNE + K-Means + Classifier
Raw data → Standardization → t-SNE embedding → K-Means on the t-SNE space → Augment t-SNE features with cluster labels → Supervised classifier

Supervised Techniques for Comparison
Use the following classifiers:

Logistic Regression
Support Vector Machine
Random Forest
Steps:

1. Load and Prepare the Kaggle Dataset

Download the Wine Quality dataset https://www.kaggle.com/datasets/yasserh/wine-quality-datasetLinks to an external site. 
Remove duplicate entries.
Handle missing values, if any.
Define the target variable.
Convert the quality label into a binary variable:
quality ≥ 6 → good wine
quality < 6 → bad wine

2. Split Features and Labels
Separate the dataset into:
: input features
: target labels

4. Standardize the Features
Apply StandardScaler.
This step is necessary because t-distributed Stochastic Neighbor Embedding and K-Means are sensitive to feature scaling.

6. Build the Pipeline: t-SNE Representation
Apply t-SNE to the standardized data.
Use 2 components for visualization and downstream tasks.
Set appropriate parameters, for example:
perplexity = 30
random_state = 42
init = "pca"

8. Choose 
 for K-Means Using Silhouette Score (t-SNE Space)
Select candidate values for 
, e.g., 2, 3, 4, 5, 6.
For each value of 
:
Fit K-Means on the t-SNE embedding.
Compute the silhouette score.
Select the value of 
 that yields the highest silhouette score.
Rationale:
The silhouette score evaluates clustering quality by measuring:

Cohesion (how close points are within the same cluster)
Separation (how distinct clusters are from each other)
6. Cluster in t-SNE Space
Fit K-Means using the selected 
.
Obtain cluster labels.
Augment the t-SNE features with the cluster labels.
The resulting feature set consists of:

t-SNE component 1
t-SNE component 2
Cluster label features
7. Train Supervised Models
Train the following classifiers on the t-SNE-based features:

Logistic Regression
Support Vector Machine (SVM)
Random Forest

8. Evaluation
Evaluate each model using:

Accuracy
F1-score
ROC-AUC
9. Visualization
Generate a scatter plot of the t-SNE embedding.
Color the points by:
True class labels, and/or
Cluster assignments
