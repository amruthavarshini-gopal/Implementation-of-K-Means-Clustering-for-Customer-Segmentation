# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## Aim:

To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Start by importing the required libraries (pandas, matplotlib.pyplot, KMeans from sklearn.cluster).

2.Load the Mall_Customers.csv dataset into a DataFrame.

3.Check for missing values in the dataset to ensure data quality.

4.Select the features Annual Income (k$) and Spending Score (1-100) for clustering.

5.Use the Elbow Method by running KMeans for cluster counts from 1 to 10 and record the Within-Cluster Sum of Squares (WCSS).

6.Plot the WCSS values against the number of clusters to determine the optimal number of clusters (elbow point).

7.Fit the KMeans model to the selected features using the chosen number of clusters (e.g., 5).

8.Predict the cluster label for each data point and assign it to a new column called cluster.

9.Split the dataset into separate clusters based on the predicted labels.

10.Visualize the clusters using a scatter plot, and optionally mark the cluster centroids.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Amruthavarshini Gopal
RegisterNumber:  212223230013
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of. clusters")
plt.ylabel("wcss")
plt.title("Elbow method")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
*/
```

## Output:

### df.head() and df.isnull().sum()

![Screenshot 2025-05-22 140556](https://github.com/user-attachments/assets/acd8e3e8-3988-4924-b499-fe42dbede7da)

### df.info()

![Screenshot 2025-05-22 140625](https://github.com/user-attachments/assets/6e57773e-a30f-45c7-a2d0-07a6e7f202d8)


### Kmeans

![Screenshot 2025-05-22 140647](https://github.com/user-attachments/assets/5f68e84e-38e3-436e-8a3b-9d9ba802b9a4)

### Elbow Method

![Screenshot 2025-05-22 140714](https://github.com/user-attachments/assets/cf22af1c-a5bb-4c77-ba84-c4fc28b99da5)

### y_pred

![Screenshot 2025-05-22 140731](https://github.com/user-attachments/assets/d6b8cfba-1fea-4152-b293-e20e9d2dfb6f)
![Screenshot 2025-05-22 140756](https://github.com/user-attachments/assets/367a67da-f872-4096-ba12-6272e27a6cc5)

## Result:

Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
