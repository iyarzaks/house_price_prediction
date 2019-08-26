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
from location_clustering import find_clusters
import math
"""
#removed_features = ['date', 'id']
# year built to age
#yr_renovated to bin
#zip code to categorial
#normalize all columns
#price to categorial
"""

def year_to_bin(year):
    if year != 0:
        return 1
    else:
        return 0


def price_to_class(price):
    if price<250000:
        return 0
    elif price<500000:
        return 1
    elif price<750000:
        return 2
    elif price<1000000:
        return 3
    else:
        return 4


def prepare_for_model(df):
    new_df_columns = list(set(df.columns) - set(['date', 'id']))
    long_lat_df = df[['long', 'lat']]
    long_lat_df.rename(columns={'long':'long_un_norm','lat':'lat_un_norm'}, inplace=True)
    df = df[new_df_columns]
    scaler = MinMaxScaler()
    normalize_df = scaler.fit_transform(df)
    res_df = pd.DataFrame(normalize_df, columns=df.columns)
    long_lat_df.index = res_df.index
    res_df = pd.concat([res_df, long_lat_df],axis=1)
    return res_df


def data_preprocessing(test_split,total_size):
    update_df = pd.read_csv('kc_house_data.csv')
    print(total_size)
    update_df = update_df.sample(total_size)
    update_df['yr_built'] = update_df['yr_built'].apply(lambda x: 2015-x)
    update_df['yr_renovated'] = update_df['yr_renovated'].apply(year_to_bin)
    update_df = pd.get_dummies(update_df,columns=['zipcode'])
    x = update_df.loc[:, update_df.columns != 'price']
    x = prepare_for_model(x)
    Y = update_df['price']
    Y=Y.apply(price_to_class)
    X_train, X_test, y_train, y_test = train_test_split(x, Y, random_state=0, test_size=test_split)
    return X_train, X_test, y_train, y_test

def result(svm_model_linear,X_test, y_test):
    accuracy = svm_model_linear.score(X_test, y_test)
    svm_predictions = svm_model_linear.predict(X_test)
    # creating a confusion matrix
    cm = confusion_matrix(y_test, svm_predictions)
    print(accuracy)
    print(cm)


def change_train_to_class_df(y_train):
    rows = []
    for val in y_train:
        dict_to_fill = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0 , 'price' : val}
        dict_to_fill[val] = 1
        rows.append(dict_to_fill)
    y_train_df = pd.DataFrame(rows)
    return y_train_df


def write_to_pkl(var, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(var, file)


def read_from_pkl(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


def print_results(y_test,y_pred):
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test,y_pred))


def create_neighbors_dict_and_data(test_split,total_size):
    unobserved_size = math.ceil(test_split * total_size)
    s_time = time.time()
    X_train, X_test, y_train, y_test = data_preprocessing(test_split,total_size)
    X_train_for_model = X_train.drop('long_un_norm', axis=1)
    X_train_for_model = X_train_for_model.drop('lat_un_norm', axis=1)
    svm_model_linear = LogisticRegressionCV(cv=5,random_state=42, solver='lbfgs',multi_class='multinomial').fit(X_train_for_model, y_train)
    X_test_for_model = X_test.drop('long_un_norm', axis=1)
    X_test_for_model = X_test_for_model.drop('lat_un_norm', axis=1)
    print_results (y_test,svm_model_linear.predict(X_test_for_model))
    svm_predictions = pd.DataFrame(svm_model_linear.predict_proba(X_test_for_model),index = X_test.index)
    y_test.index = X_test.index
    x_test_with_probs = pd.concat([X_test,svm_predictions,y_test], axis=1)
    y_train = change_train_to_class_df(y_train)
    y_train.set_index(X_train.index,inplace=True)
    x_train_with_probs = pd.concat([X_train,y_train], axis=1)
    all_data = pd.concat([x_test_with_probs, x_train_with_probs], sort=True, axis=0)
    all_data = find_clusters(all_data)
    columns_for_mrf = ['long_un_norm','lat_un_norm',0,1,2,3,4,'price','cluster']
    all_data = all_data[columns_for_mrf]
    all_data['old_index'] = all_data.index
    all_data.index = range(len(all_data))
    #all_data = find_clusters(all_data)
    neigh_dict = mrf.get_neighbors_dict(all_data, unobserved_size)
    write_to_pkl(neigh_dict, 'neighbors_dict_lr_2.pkl')
    write_to_pkl(all_data, 'single_potentials_all_nodes_lr_2.pkl')
    total = time.time() - s_time
    str(timedelta(seconds=total))


def create_pairwise_potentials():
    neigh_dict = read_from_pkl('neighbors_dict_lr_2.pkl')
    print (len(neigh_dict))
    all_data = read_from_pkl('single_potentials_all_nodes_lr_2.pkl')
    potentials = mrf.build_potentials(neigh_dict, all_data)
    write_to_pkl(potentials, 'potentials_dict_all_nodes_lr_2.pkl')


def main():
    create_neighbors_dict_and_data(test_split=0.25, total_size=2000)
    create_pairwise_potentials()






if __name__ == '__main__':
    main()
