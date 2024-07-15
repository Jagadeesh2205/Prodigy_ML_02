
# Mall Customer Segmentation

This project demonstrates customer segmentation using the K-Means clustering algorithm on the Mall Customer dataset. The dataset includes customer information such as `Annual Income (k$)` and `Spending Score (1-100)`. The goal is to segment customers into distinct groups based on their annual income and spending score.

## Table of Contents
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Code Explanation](#code-explanation)
- [Elbow Method](#elbow-method)
- [Clustering](#clustering)
- [Results](#results)
- [Usage](#usage)

## Dataset
The dataset used in this project is `Mall_Customers.csv`, which contains the following columns:
- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

## Dependencies
To run this project, you need the following Python libraries:
- pandas
- numpy
- matplotlib
- scikit-learn

You can install these dependencies using the following command:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## Code Explanation

1. **Import Libraries**:
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans 
    ```

2. **Load and Inspect Data**:
    ```python
    data = pd.read_csv('Mall_Customers.csv')
    print(data.head())
    print(data.isnull().sum())
    ```

3. **Select Features for Clustering**:
    ```python
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    ```

4. **Determine Optimal Number of Clusters using Elbow Method**:
    ```python
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

5. **Apply K-Means Clustering**:
    ```python
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    data['Cluster'] = y_kmeans
    ```

6. **Visualize Clusters**:
    ```python
    plt.figure(figsize=(12, 8))
    plt.scatter(X.iloc[y_kmeans == 0, 0], X.iloc[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(X.iloc[y_kmeans == 1, 0], X.iloc[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(X.iloc[y_kmeans == 2, 0], X.iloc[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(X.iloc[y_kmeans == 3, 0], X.iloc[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(X.iloc[y_kmeans == 4, 0], X.iloc[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    ```

## Elbow Method
The elbow method is used to determine the optimal number of clusters for K-means clustering. It plots the Within-Cluster Sum of Squares (WCSS) against the number of clusters. The point where the WCSS starts to diminish significantly is considered the elbow point, indicating the optimal number of clusters.

## Clustering
The K-means algorithm is applied with the optimal number of clusters (5 in this case). The resulting clusters are visualized, showing distinct groups of customers based on their annual income and spending score.

## Results
The clustering results in five distinct customer segments, visualized using a scatter plot. Each cluster is represented by a different color, and the cluster centroids are highlighted.

## Usage
1. Ensure you have the required dependencies installed.
2. Download the `Mall_Customers.csv` dataset and place it in the same directory as the script.
3. Run the script to perform customer segmentation and visualize the results:
    ```bash
    python mall_customer_segmentation.py
    ```

---
