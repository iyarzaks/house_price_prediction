import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import mrf
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
    new_df_columns = list(set(df.columns) - set(['date', 'id', 'long', 'lat']))
    df = df[new_df_columns]
    scaler = MinMaxScaler()
    normalize_df = scaler.fit_transform(df)
    df = pd.DataFrame(normalize_df)
    return df


def data_preprocessing():
    update_df = pd.read_csv('kc_house_data.csv')
    update_df = update_df.sample(20)
    update_df['yr_built'] = update_df['yr_built'].apply(lambda x: 2015-x)
    update_df['yr_renovated'] = update_df['yr_renovated'].apply(year_to_bin)
    update_df = pd.get_dummies(update_df,columns=['zipcode'])
    x = update_df.loc[:, update_df.columns != 'price']
    Y = update_df['price']
    Y=Y.apply(price_to_class)
    X_train, X_test, y_train, y_test = train_test_split(x, Y, random_state=0)
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
        dict_to_fill = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        dict_to_fill[val] = 1
        rows.append(dict_to_fill)
    y_train_df = pd.DataFrame(rows)
    return y_train_df


def main():
    X_train, X_test, y_train, y_test = data_preprocessing()
    X_train_for_model = prepare_for_model(X_train)
    svm_model_linear = SVC(kernel='linear', C=1 ,probability=True).fit(X_train_for_model, y_train)
    X_test_for_pred = prepare_for_model(X_test)
    svm_predictions = pd.DataFrame(svm_model_linear.predict_proba(X_test_for_pred),index = X_test.index)
    x_test_with_probs = pd.concat([X_test,svm_predictions], axis=1)
    y_train = change_train_to_class_df(y_train)
    y_train.set_index(X_train.index,inplace=True)
    x_train_with_probs = pd.concat([X_train,y_train], axis=1)
    all_data = pd.concat([x_test_with_probs,x_train_with_probs], axis=0)
    columns_for_mrf = ['long','lat',0,1,2,3,4]
    all_data = all_data[columns_for_mrf]
    all_data['old_index'] = all_data.index
    all_data.index = range(len(all_data))
    neigh_dict = mrf.get_neighbors_dict(all_data)
    potentials = mrf.build_potentials(neigh_dict, all_data)
    print(potentials[(0,1)])
    print (potentials.keys())


if __name__ == '__main__':
    main()
