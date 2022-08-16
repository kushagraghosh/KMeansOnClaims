# KMeansOnClaims
Used Python 3.9 (Pandas, MatplotLib, Numpy, IPyWidgets) for K-means clustering (ML algorithm) to iteratively find ‘K’ number of MetLife customer groups from a dataset of 45,017 insurance claims. [Jupyter Notebook](ML_MetLife_Claims_Demo_Kushagra_Ghosh.ipynb). [Python code](ml_metlife_claims_demo_kushagra_ghosh.py).

I created a project where I could predict outcomes based on training data related to the MetLife business. I worked with insurance claims history data to create predictive modeling using AI/ML Modeling, so I downloaded Anaconda with Jupyter Notebook and Python version 3.9.12. After using the Anaconda Prompt to open Jupyter, I imported the Pandas, MatplotLib, and Numpy packages for data analysis and then read the claims data from the excel format.

With this data, I used K-means clustering, which is an unsupervised ML algorithm for iteratively finding ‘K’ number of clusters in an unlabeled dataset based on feature similarity. This is useful for identifying unknown groups from large and versatile datasets, which is useful for identifying behavioral segmentation of various MetLife customers and claimants (based on Disease Claims and Accident Claims in my example).

In my code, I created the ‘claimsdf’ dataframe that grouped every claim based on each unique insured ID number (for each unique person) and then aggregated the number of each type of for each person. I then used the ‘elbow method’ to find optimal number of clusters (5) for the K-Means Algorithm so that my model could accurately identify patterns within customers based on their claims and coverage. I trained the K-Means model on the dataset and then visualized the clusters with a Matplotlib scatterplot with the generated clusters and their centroids (averages) plotted.

Using these clusters, I created a IPyWidget to show each cluster’s unique customers and all claims within the cluster. I also created bar plots and tables to indicate profiles of MetLife customers for each cluster that could be used to inform business about their assumptions for each customer/policyholder.
![alt text](ClaimsWidgetsPlots.png?raw=true)
