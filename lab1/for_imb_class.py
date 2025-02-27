import sklearn
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import pandas as  pd
import numpy as np 
from typing import Union
from imblearn.over_sampling import SMOTE
    
def get_clustered_data(data: Union[pd.DataFrame,np.array],n_clusters = 6000):
    '''
    Takes in the majority class and get cluster for undersampling
    The idea is to get N cluster = N sample of minority class
    Silhouette score is just to examine (score near 0 means overlapping clusters,
    which is what we want since these samples are from the same class)
    '''
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(data)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    centers = clusterer.cluster_centers_
    return centers


#this didn't work for some reason
# def get_smote_data(X,y,ratio):
#     '''
#     takes data of big and small class
#     'ration' is defined as N_mi_new/N_ma (while N_mi_new is number of  sample in minor class after smote
#     and N_ma is the number of samples in majority class )'''
    
#     X_weed,y_fentanyl = SMOTE(sampling_strategy = ratio,random_state = 4122004).fit_resample(X,y)
#     return X_weed,y_fentanyl