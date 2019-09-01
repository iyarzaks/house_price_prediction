
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import math
import statistics
import numpy


def get_edges_probs(df, unobserved_size, max_dist=0.0175):
    count_dict= {
        (0,0): 0,(0,1): 0, (0,2): 0, (0,3): 0, (0,4): 0,
        (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 0,
        (2, 2): 0, (2, 3): 0, (2, 4): 0,
        (3, 3): 0, (3, 4): 0, (4,4): 0,
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0
    }
    cluster_labels = pd.Series(df['cluster'])
    for cluster_label in cluster_labels.unique():
        current_cluster = df[df.cluster == cluster_label]
        for i in reversed(current_cluster.index):
            if i < unobserved_size:
                break
            for j in reversed(current_cluster.index):
                if i > j > unobserved_size:
                    distance = math.sqrt(math.pow(df.iloc[i]['lat_un_norm']-df.iloc[j]['lat_un_norm'],2)+math.pow(df.iloc[i]['long_un_norm']-df.iloc[j]['long_un_norm'],2))
                    if distance < max_dist:
                        label_i=df.iloc[i]['price']
                        label_j = df.iloc[j]['price']
                        if label_i<label_j:
                            count_dict[(label_i, label_j)] += 1
                        else:
                            count_dict[(label_j, label_i)] += 1
                        count_dict[label_i] += 1
                        count_dict[label_j] += 1
            if i % 10 == 0:
                print(i)
        print("finish cluster " + str(cluster_label))
    result = {}
    for couple in count_dict:
        if isinstance(couple,tuple):
            result[couple] = (1+count_dict[couple])/(2+(count_dict[couple[0]] + count_dict[couple[1]]))
    return result



def get_neighbors_dict(df, unobserved_size, max_dist=0.0175):
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
                    similarity = find_house_similarity(df.iloc[i], df.iloc[j])
                    if similarity > 0.95:
                    #if distance < max_dist:
                        hist[abs(df.iloc[i]['price']-df.iloc[j]['price'])] = hist[abs(df.iloc[i]['price']-df.iloc[j]['price'])]+1
                        distance_dict1[(i, j)] = distance
            if i % 10 == 0:
                print(i)
        print("finish cluster " + str(cluster_label))
    return distance_dict1


def potentential_factor(i,j,similarity, edges_probabilities):
    if not edges_probabilities:
        if abs(i - j) == 1:
            return 0.95
        else:
            return math.pow(0.5, abs(i - j))
    else:
        #max_value = max(edges_probabilities.values())
        if i < j:
            return edges_probabilities[(i, j)]
        else:
            return edges_probabilities[(j, i)]



def cosine_similarity(x, y):
    """ return cosine similarity between two lists """

    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 5)


def square_rooted(x):
    """ return 3 rounded square rooted value """

    return round(math.sqrt(sum([a * a for a in x])), 5)

def find_house_similarity(house1,house2):
    features_for_sim = ['lat','sqft_above','sqft_basement',
                        'sqft_living15','condition','bedrooms',
                        'sqft_living','grade','waterfront','floors',
                        'yr_renovated','yr_built','bathrooms','sqft_lot15',
                        'sqft_lot','view']
    house1 = house1[features_for_sim]
    house2 = house2[features_for_sim]
    similarity = cosine_similarity(house1.values, house2.values)
    return similarity


def build_potentials(neighbors_dict, df , edges_probabilities):
    couple_dict = {}
    for couple in neighbors_dict:
        similarity = find_house_similarity(df.iloc[couple[0]],df.iloc[couple[1]])
        potential_couple_df = pd.DataFrame(index=range(5),columns=range(5))
        for i in range(5):
            for j in range(5):
                potential_couple_df.at[i,j] = df.iloc[couple[0]][i]*df.iloc[couple[1]][j]*potentential_factor(i,j,similarity, edges_probabilities)
        couple_dict[couple] = potential_couple_df
    return couple_dict




#numpy.quantile(l,0.005)
