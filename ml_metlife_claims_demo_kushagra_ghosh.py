import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import ipywidgets as widgets
from IPython.display import display

df = pd.read_excel('EMEA Claims History Data set - Test.xlsx')
#Claims data from excel
print(df)

#Created new dataframe that groups based on each unique insured ID number and then aggregates conditional number of claims depending on the type of claim for each column.
claimsdf = df.groupby(['INSURED ID NUMBER']).apply(lambda x: pd.Series(dict(
    Disease_Claims=(x.CATEGORY == "D").sum(),
    Accident_Claims=(x.CATEGORY == "A").sum(),
    Outpatient_Care_Claims=(x.CATEGORY == "OC").sum(),
    Ordinary_Death_Claims=(x.CATEGORY == "OD").sum(),
    Credit_Life_Claims=(x.CATEGORY == "Y").sum(),
    Accident_Death_Claims=(x.CATEGORY == "AD").sum(),
    Travel_Insurance_Claims=(x.CATEGORY == "T").sum(),
    Past_Yr_Claims=(x["EVENT DATE"]>20210000).sum(),
    Lifetime_Claims=(x.CATEGORY != "").sum(),
)))
claimsdf

#To find the unique categories to group for the dataframe above
test = df.groupby(['INSURED ID NUMBER', 'CATEGORY']).size()
new_df = test.to_frame(name='Num Of Claims').reset_index()
new_df
new_df['CATEGORY'].unique() # 99, 'AD', 'D', 'OD', 'Y', 'OC', 'A', 'T'

#Clustering method to identify patterns within customers based on their claims and coverage. Understand customer base more.
useful_claimsdf = claimsdf[ (claimsdf['Disease_Claims']>=0) & (claimsdf['Accident_Claims']>=0) & (claimsdf['Disease_Claims']<200) & (claimsdf['Accident_Claims']<200)]
print(useful_claimsdf.iloc[7])
X = useful_claimsdf[['Disease_Claims', 'Accident_Claims']].values #2 features for easier visualization of clusters (1 axis per feature)
X

#Using elbow method to find optimal number of clusters for K-Means Algo
from sklearn.cluster import KMeans  #Run K-means algo several times with 10 different num of clusters
wcss = []   #Within Cluster Sum of Squares: sum of the squared distances between each observation pt of the cluster and the centroid of the cluster (for each of the num of clusters)
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) #Creating 10 different K-Mean objects for each cluster
    kmeans.fit(X) #Training K-means algo with i num of clusters
    wcss.append(kmeans.inertia_) #Get wcss value
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42) #Creating 10 different K-Mean objects for each cluster
y_kmeans = kmeans.fit_predict(X) #Create dependent variable clusters
print(y_kmeans)

#Visualizing the clusters
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='orange', label='Cluster 4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='cyan', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Disease claims')
plt.ylabel('Accident Claims')
plt.legend()
plt.show()

def updateClusterTables(c):
    numCluster = c
    filtered = useful_claimsdf[y_kmeans==numCluster-1]
    print('Cluster '+str(numCluster)+" customers")
    display(filtered)
    
    joined = pd.merge(filtered, df, on="INSURED ID NUMBER", how='inner')
    joined = joined[pd.notnull(joined['Disease_Claims'])][df.columns]
    print('All claims from the cluster '+str(numCluster)+" customers")
    display(joined)

amp = widgets.IntSlider(min=1, max=5, description="Cluster: ")
widgets.interact(updateClusterTables, c=amp)

disease_claims = kmeans.cluster_centers_[:, 0].tolist()
accident_claims = kmeans.cluster_centers_[:, 1].tolist()
index = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
barplotdf = pd.DataFrame({"Average Number of Disease Claims": disease_claims, "Average Number of Accident Claims": accident_claims}, index=index)
axes = barplotdf.plot.bar(rot=0, subplots=True)
axes[1].legend(loc=2)
barplotdf

from platform import python_version
print(python_version())

