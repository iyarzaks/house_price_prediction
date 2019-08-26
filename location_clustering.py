import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import mrf
import pickle
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from hunga_bunga import HungaBungaClassifier, HungaBungaRegressor
import numpy as np
import time
from datetime import timedelta
from sklearn.cluster import KMeans


def find_clusters(update_df):
    #location_df = update_df[['long_un_norm','lat_un_norm']]
    fetures_df = update_df.drop(columns=['long_un_norm','lat_un_norm','long','lat','price',0,1,2,3,4])
    k_means = KMeans(n_clusters=4, random_state=0).fit(fetures_df)
    out = pd.value_counts(pd.Series(k_means.labels_))
    update_df ['cluster'] = pd.Series(k_means.labels_)
    return update_df


def main():
    find_clusters()


if __name__ == '__main__':
    main()


