import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


data = pd.read_csv(r'/Users\User\Desktop\spring2022\pythonProject1/Cust_segmentation/Mall_Customers.csv')


# We will need to transform categorical variable Gender into a numeric one
data['Gender'].replace(['Male', 'Female'],[0, 1], inplace=True)


def age_vs_spending_score(data):
    """Creates a scatter plot between Age and Spending score of a customer

    returns a scatter plot
    """
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
    """Creates a scatter plot between age and annual income

    returns a scatter plot
    """
    plt.figure(figsize=(10,6))
    plt.scatter(data['Age'],data['Annual Income (k$)'], marker='o');
    plt.xlabel('Age')
    plt.ylabel('Annual Income')
    plt.title('Scatter plot between Age and Annual Income')
    return plt

age_vs_annualincome(data)


def annualincome_vs_spendingscore(data):
    """Creates a scatter plot between income and spending score

    Returns a scatter plot
    """
    plt.figure(figsize=(10,6))
    plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'], marker='o');
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.title('Scatter plot between Annual Income and Spending Score')
    return plt

annualincome_vs_spendingscore(data)


def gender_vs_spendingscore(data):
    """Creates a scatter plot between gender and spending score

    Returns a scatter plot
    """
    plt.figure(figsize=(10,6))
    plt.scatter(data['Gender'],data['Spending Score (1-100)'], marker='o');
    plt.xlabel('Gender')
    plt.ylabel('Spending Score')
    plt.title('Scatter plot between Gender and Spending Score')
    return plt

gender_vs_spendingscore(data)


def corr(data):
    """Plots a correlation heatmap between variables

    Returns a correlation heatmap"""
    fig_dims = (7, 7)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.heatmap(data.corr(), annot=True, cmap='viridis')
    return sns

corr(data)



scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])


def elbow(data):
    """Returns optimal number of clusters data should be divided to

    Returns a plot"""
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=42)
        x = data.copy()
        kmeans.fit(x)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)

    plt.figure(figsize=(10, 5))
    no_clusters = range(1, 11)
    plt.plot(no_clusters, wcss, marker="o")
    plt.title('The elbow method', fontweight="bold")
    plt.xlabel('Number of clusters(K)')
    plt.ylabel('within Clusters Sum of Squares(WCSS)')
    return plt
elbow(data)



# depending on the optimal number of clusters we got on elbow method, we will use that to do clustering

def clustering_new(data):
    """Returns mall customers divided to 5 clusters based on annual income and spending score

    Returns a scatter plot"""

    x = data.copy()
    kmeans_new = KMeans(5)
    kmeans_new.fit(x)

    clusters_new = x.copy()
    clusters_new['cluster_pred'] = kmeans_new.fit_predict(x)
    gender= {0:'Male',1:'Female'}
    clusters_new['Gender']= clusters_new['Gender'].map(gender)
    plt.figure(figsize=(6,6))
    plt.scatter(clusters_new['Annual Income (k$)'],clusters_new['Spending Score (1-100)'],c=clusters_new['cluster_pred'],cmap='rainbow')
    plt.title("Clustering customers based on Annual Income and Spending score", fontsize=15,fontweight="bold")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    return plt
clustering_new(data)


def barplot_age(data):
    x = data.copy()
    kmeans_new = KMeans(5)
    kmeans_new.fit(x)
    clusters_new = x.copy()
    clusters_new['cluster_pred'] = kmeans_new.fit_predict(x)
    avg_data = clusters_new.groupby(['cluster_pred'], as_index=False).mean()
    sns.barplot(x='cluster_pred',y='Age',palette="plasma",data=avg_data)
    return sns
barplot_age(data)

def barplot_annualincome(data):
    x = data.copy()
    kmeans_new = KMeans(5)
    kmeans_new.fit(x)
    clusters_new = x.copy()
    clusters_new['cluster_pred'] = kmeans_new.fit_predict(x)
    avg_data = clusters_new.groupby(['cluster_pred'], as_index=False).mean()
    sns.barplot(x='cluster_pred', y='Annual Income (k$)', palette="plasma", data=avg_data)
    return sns
barplot_annualincome(data)

def barplot_spendingscore(data):
    x = data.copy()
    kmeans_new = KMeans(5)
    kmeans_new.fit(x)
    clusters_new = x.copy()
    clusters_new['cluster_pred'] = kmeans_new.fit_predict(x)
    avg_data = clusters_new.groupby(['cluster_pred'], as_index=False).mean()
    sns.barplot(x='cluster_pred',y='Spending Score (1-100)',palette="plasma",data=avg_data)
    return sns
barplot_spendingscore(data)
