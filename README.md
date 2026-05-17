# Pipeline: t-SNE + K-Means + Supervised Classification (Wine Quality)

## Project Overview

This project explores the effectiveness of a hybrid machine learning pipeline that combines **t-SNE dimensionality reduction**, **K-Means clustering**, and **supervised learning models** for predicting wine quality.

The main research question is:

> How effective is a pipeline combining t-SNE, K-Means clustering, and supervised learning for classifying wine quality?

---

## Pipeline Architecture

The full pipeline follows these steps:


Raw Data
↓
Data Cleaning & Preprocessing
↓
Standardization (StandardScaler)
↓
t-SNE Embedding (2D)
↓
K-Means Clustering (on t-SNE space)
↓
Feature Augmentation (cluster labels + t-SNE features)
↓
Supervised Classification


---

## Dataset

- Source: Kaggle Wine Quality Dataset  
- Link: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

### Preprocessing Steps:
- Removed duplicate entries
- Handled missing values (if any)
- Converted target variable:
  - `quality ≥ 6 → Good Wine (1)`
  - `quality < 6 → Bad Wine (0)`

---

## Feature Engineering

### Standardization
All features were standardized using `StandardScaler` because:
- t-SNE is sensitive to scale
- K-Means relies on distance metrics

### t-SNE Transformation
- Reduced data to **2 components**
- Parameters:
  - perplexity = 30
  - random_state = 42
  - init = "pca"

---

## Clustering (K-Means)

- Applied K-Means on t-SNE embeddings
- Tested values of K: **2 to 6**
- Selected optimal K using **Silhouette Score**

### Silhouette Score Rationale:
- Measures cluster quality based on:
  - Cohesion (intra-cluster similarity)
  - Separation (inter-cluster distance)

---

## Supervised Learning Models

The following models were trained using the augmented feature space:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

---

## Evaluation Metrics

Each model was evaluated using:

- Accuracy
- F1-score
- ROC-AUC

---

## Visualization

A 2D t-SNE scatter plot was generated showing:

- Data points in reduced feature space
- Color-coded by:
  - True wine quality labels
  - K-Means cluster assignments

This helps visualize:
- Class separability
- Cluster structure in reduced space

---

## Key Idea

This project investigates whether combining:
- **t-SNE (nonlinear embedding)**
- **K-Means (unsupervised clustering)**
- **Supervised classifiers**

can improve classification performance by introducing meaningful structure into the feature space.

---

## Tech Stack

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib / Seaborn

---

## How to Run

bash
pip install -r requirements.txt
python main.py

---

## Future Improvements
Try UMAP instead of t-SNE
Use GridSearchCV for classifier tuning
Compare with deep learning models
Add cross-validation pipeline

---

## Author

Machine Learning Project — Feature Engineering & Clustering Pipeline


---
