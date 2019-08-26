
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
    hist = {0 : 0, 1:0 ,2:0, 3:0 ,4:0}
    for cluster_label in cluster_labels.unique():
        current_cluster = df[df.cluster == cluster_label]
        for i in current_cluster.index:
            if i >= unobserved_size:
                break
            for j in current_cluster.index:
                if i < j:
                    distance = math.sqrt(math.pow(df.iloc[i]['lat_un_norm']-df.iloc[j]['lat_un_norm'],2)+math.pow(df.iloc[i]['long_un_norm']-df.iloc[j]['long_un_norm'],2))
                    if distance < 0.02:
                        hist[abs(df.iloc[i]['price']-df.iloc[j]['price'])] = hist[abs(df.iloc[i]['price']-df.iloc[j]['price'])]+1
                        distance_dict1[(i, j)] = distance
            if i % 10 == 0:
                print(i)
        print("finish cluster " + str(cluster_label))
    print (hist)
    return distance_dict1


def potentential_factor(i,j):
    if abs(i - j) == 1:
        return 0.95
    else:
        return (math.pow(0.5, abs(i - j)))


def build_potentials(neighbors_dict, df):
    couple_dict = {}
    for couple in neighbors_dict:
        potential_couple_df = pd.DataFrame(index=range(5),columns=range(5))
        for i in range(5):
            for j in range(5):
                potential_couple_df.at[i,j] = df.iloc[couple[0]][i]*df.iloc[couple[1]][j]*potentential_factor(i,j)
        couple_dict[couple] = potential_couple_df
    return couple_dict




#numpy.quantile(l,0.005)
