import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.cluster import Birch

data = pd.read_csv(r'C:\Users\User\Desktop\spring2022\pythonProject1\Cust_segmentation\Mall_Customers.csv')

print(data.head())


# We will need to transform categorical variable Gender into a numeric one
data['Gender'].replace(['Male', 'Female'],[0, 1], inplace=True)


# K-means clustering
def kmeans_clustering(data):
    kmeans = KMeans(n_clusters=4).fit(data)
    centroids = kmeans.cluster_centers_
    print(centroids)

    kmeans = KMeans(n_clusters=4, init='k-means++')
    kmeans.fit(data)
    print(silhouette_score(data, kmeans.labels_, metric='euclidean'))

    clusters = kmeans.fit_predict(data.iloc[:, 1:])
    data["label"] = clusters

    fig = plt.figure(figsize=(21, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data.Age[data.label == 0], data["Annual Income (k$)"][data.label == 0],
               data["Spending Score (1-100)"][data.label == 0], c='blue', s=60)
    ax.scatter(data.Age[data.label == 1], data["Annual Income (k$)"][data.label == 1],
               data["Spending Score (1-100)"][data.label == 1], c='red', s=60)
    ax.scatter(data.Age[data.label == 2], data["Annual Income (k$)"][data.label == 2],
               data["Spending Score (1-100)"][data.label == 2], c='green', s=60)
    ax.scatter(data.Age[data.label == 3], data["Annual Income (k$)"][data.label == 3],
               data["Spending Score (1-100)"][data.label == 3], c='orange', s=60)

    ax.view_init(30, 185)
    return ax

kmeans_clustering(data)


def age_vs_spending_score(data):
    plt.figure(figsize=(10,6))
    plt.scatter(data['Age'],data['Spending Score (1-100)'], marker='o');
    plt.xlabel('Age')
    plt.ylabel('Spending Score')
    plt.title('Scatter plot between Age and Spending Score')
    return plt

age_vs_spending_score(data)


# The lower the age the higher the spending score is, because the correlation is negative
data['Spending Score (1-100)'].corr(data['Age'])


def age_vs_annualincome(data):
    plt.figure(figsize=(10,6))
    plt.scatter(data['Age'],data['Annual Income (k$)'], marker='o');
    plt.xlabel('Age')
    plt.ylabel('Annual Income')
    plt.title('Scatter plot between Age and Annual Income')
    return plt

age_vs_annualincome(data)


def annualincome_vs_spendingscore(data):
    plt.figure(figsize=(10,6))
    plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'], marker='o');
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.title('Scatter plot between Annual Income and Spending Score')
    return plt

annualincome_vs_spendingscore(data)


def gender_vs_spendingscore(data):
    plt.figure(figsize=(10,6))
    plt.scatter(data['Gender'],data['Spending Score (1-100)'], marker='o');
    plt.xlabel('Gender')
    plt.ylabel('Spending Score')
    plt.title('Scatter plot between Gender and Spending Score')
    return plt

gender_vs_spendingscore(data)


def corr(data):
    fig_dims = (7, 7)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.heatmap(data.corr(), annot=True, cmap='viridis')
    return sns

corr(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])


def clustering(data):
    x = data.copy()
    kmeans = KMeans(3)
    kmeans.fit(x)
    clusters = x.copy()

    clusters['cluster_pred']=kmeans.fit_predict(x)

    plt.scatter(clusters['Annual Income (k$)'],clusters['Spending Score (1-100)'],c=clusters['cluster_pred'],cmap='rainbow')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    return plt

clustering(data)


