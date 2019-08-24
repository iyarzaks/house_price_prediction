
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import math
import statistics
import numpy


def get_neighbors_dict(df, unobserved_size):
    distance_dict1 = {}
    cluster_labels = pd.Series(df['cluster'])
    for cluster_label in cluster_labels.unique():
        current_cluster = df[df.cluster == cluster_label]
        for i in current_cluster.index:
            if i >= unobserved_size:
                break
            for j in current_cluster.index:
                if i < j:
                    distance = math.sqrt(math.pow(df.iloc[i]['lat_un_norm']-df.iloc[j]['lat_un_norm'],2)+math.pow(df.iloc[i]['long_un_norm']-df.iloc[j]['long_un_norm'],2))
                    if distance < 0.015:
                        distance_dict1[(i, j)] = distance
        # if i % 10 == 0:
        #     print(i)
        print("finish cluster " + str(cluster_label))
    return distance_dict1


def build_potentials(neighbors_dict, df):
    couple_dict = {}
    for couple in neighbors_dict:
        potential_couple_df = pd.DataFrame(index=range(5),columns=range(5))
        for i in range(5):
            for j in range(5):
                potential_couple_df.at[i,j] = df.iloc[couple[0]][i]*df.iloc[couple[1]][j]*(math.pow(0.5, abs(i - j)))
                #((5 - (abs(i - j))) / 5)
        couple_dict[couple] = potential_couple_df
    return couple_dict




#numpy.quantile(l,0.005)
